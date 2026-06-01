from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
from pathlib import Path
import types
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import pyshmem
import pyshmem._shared as pyshmem_shared


pytestmark = pytest.mark.cpu
WINDOWS_SHARED_MEMORY_IS_EPHEMERAL = sys.platform == "win32"
TEST_SRC_PATH = str(Path(__file__).resolve().parents[1] / "src")


def _run_python_child(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH")
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join((TEST_SRC_PATH, pythonpath))
    else:
        env["PYTHONPATH"] = TEST_SRC_PATH
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _read_cpu_payload(name: str, queue) -> None:
    shm = pyshmem.open(name)
    payload = shm.read()
    queue.put(
        {
            "shape": shm.shape,
            "dtype": str(shm.dtype),
            "size": shm.size,
            "values": payload.tolist(),
        }
    )
    shm.close()


def _hold_lock(name: str, queue, hold_seconds: float) -> None:
    shm = pyshmem.open(name)
    with shm.locked(timeout=1.0):
        queue.put("locked")
        time.sleep(hold_seconds)
    shm.close()


def _crash_while_holding_lock(name: str, event) -> None:
    shm = pyshmem.open(name)
    shm.acquire(timeout=1.0)
    event.set()
    os._exit(0)


def _create_write_and_exit(name: str, event) -> None:
    shm = pyshmem.create(name, shape=(2, 2), dtype=np.float32)
    shm.write(np.full((2, 2), 7.0, dtype=np.float32))
    event.set()
    shm.close()


def test_import_exposes_public_api():
    assert callable(pyshmem.create)
    assert callable(pyshmem.open)
    assert pyshmem.SharedMemory is not None


def test_instance_does_not_expose_module_factory_methods(shm_name):
    shm = pyshmem.create(shm_name, shape=(1,), dtype=np.float32)

    assert not hasattr(shm, "create")
    assert not hasattr(shm, "open")

    shm.close()


def test_create_write_read_round_trip_cpu(shm_name):
    shm = pyshmem.create(shm_name, shape=(2, 3), dtype=np.float32)

    assert shm.name == shm_name
    assert shm.shape == (2, 3)
    assert shm.dtype == np.dtype(np.float32)
    assert shm.size == 24
    assert shm.gpu_device is None

    payload = np.arange(6, dtype=np.float32).reshape(2, 3)
    shm.write(payload)
    received = shm.read()

    assert isinstance(received, np.ndarray)
    np.testing.assert_array_equal(received, payload)

    shm.close()


def test_open_reports_clear_error_for_missing_name(shm_name):
    with pytest.raises(FileNotFoundError, match="does not exist") as exc_info:
        pyshmem.open(shm_name)

    assert f"pyshmem.create({shm_name!r}, ...)" in str(exc_info.value)


def test_open_reconstructs_shape_dtype_and_contents(shm_name):
    writer = pyshmem.create(shm_name, shape=(4,), dtype=np.int32)
    payload = np.array([1, 2, 3, 4], dtype=np.int32)
    writer.write(payload)

    reader = pyshmem.open(shm_name)

    assert reader.name == shm_name
    assert reader.shape == (4,)
    assert reader.dtype == np.dtype(np.int32)
    assert reader.size == payload.nbytes
    assert reader.gpu_device is None
    np.testing.assert_array_equal(reader.read(), payload)

    reader.close()
    writer.close()


def test_read_new_waits_for_next_write(shm_name):
    writer = pyshmem.create(shm_name, shape=(2, 2), dtype=np.float32)
    reader = pyshmem.open(shm_name)
    writer.write(np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(
        reader.read(), np.zeros((2, 2), dtype=np.float32)
    )

    def delayed_write():
        time.sleep(0.05)
        writer.write(np.ones((2, 2), dtype=np.float32))

    thread = threading.Thread(target=delayed_write)
    thread.start()
    start = time.monotonic()
    received = reader.read_new(timeout=1.0)
    elapsed = time.monotonic() - start
    thread.join()

    assert elapsed >= 0.04
    np.testing.assert_array_equal(received, np.ones((2, 2), dtype=np.float32))

    reader.close()
    writer.close()


def test_read_new_times_out_when_no_new_write_arrives(shm_name):
    writer = pyshmem.create(shm_name, shape=(1,), dtype=np.float32)
    reader = pyshmem.open(shm_name)

    with pytest.raises(TimeoutError):
        reader.read_new(timeout=0.05)

    reader.close()
    writer.close()


def test_safe_reads_stay_consistent_during_concurrent_writes(shm_name):
    writer = pyshmem.create(shm_name, shape=(32, 32), dtype=np.float32)
    reader = pyshmem.open(shm_name)
    stop_event = threading.Event()
    failures: list[str] = []

    def write_loop() -> None:
        for value in range(1, 200):
            writer.write(np.full((32, 32), value, dtype=np.float32))
        stop_event.set()

    thread = threading.Thread(target=write_loop)
    thread.start()

    while not stop_event.is_set():
        snapshot = reader.read()
        if not np.all(snapshot == snapshot[0, 0]):
            failures.append("inconsistent snapshot")
            break

    thread.join()
    assert failures == []

    reader.close()
    writer.close()


def test_cpu_stream_can_be_opened_in_another_process(shm_name):
    writer = pyshmem.create(shm_name, shape=(2, 3), dtype=np.float32)
    payload = np.arange(6, dtype=np.float32).reshape(2, 3)
    writer.write(payload)

    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_read_cpu_payload, args=(shm_name, queue))
    process.start()
    process.join(timeout=20)

    assert process.exitcode == 0
    message = queue.get(timeout=5)
    assert message["shape"] == (2, 3)
    assert message["dtype"] == "float32"
    assert message["size"] == payload.nbytes
    assert message["values"] == payload.tolist()

    writer.close()


@pytest.mark.skipif(
    WINDOWS_SHARED_MEMORY_IS_EPHEMERAL,
    reason=(
        "Windows destroys named shared memory when the last handle closes, "
        "so attach persistence after another process exits is not meaningful"
    ),
)
def test_attached_process_exit_keeps_stream_attachable(shm_name):
    writer = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)
    payload = np.array([1.0, 2.0], dtype=np.float32)
    writer.write(payload)

    child = _run_python_child(
        "import pyshmem; "
        f"shm = pyshmem.open({shm_name!r}); "
        "print(shm.read().tolist())"
    )

    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == str(payload.tolist())
    assert "resource_tracker" not in child.stderr

    reopened = pyshmem.open(shm_name)
    np.testing.assert_array_equal(reopened.read(), payload)

    reopened.close()
    writer.close()


def test_unregister_uses_platform_expected_shared_memory_name(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_unregister(name: str, rtype: str) -> None:
        calls.append((name, rtype))

    monkeypatch.setattr(
        pyshmem_shared.resource_tracker, "unregister", fake_unregister
    )

    shm = types.SimpleNamespace(_name="/ps_test_meta")

    pyshmem_shared._unregister(shm)

    if os.name == "nt":
        assert calls == []
    else:
        assert calls == [("/ps_test_meta", "shared_memory")]


def test_cross_process_lock_blocks_explicit_acquire_until_release(shm_name):
    writer = pyshmem.create(shm_name, shape=(4,), dtype=np.float32)
    payload = np.arange(4, dtype=np.float32)
    writer.write(payload)
    reader = pyshmem.open(shm_name)

    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_hold_lock, args=(shm_name, queue, 0.2))
    process.start()

    assert queue.get(timeout=5) == "locked"
    start = time.monotonic()
    reader.acquire(timeout=1.0)
    elapsed = time.monotonic() - start
    received = reader.read()
    reader.release()

    process.join(timeout=20)
    assert process.exitcode == 0
    assert elapsed >= 0.15
    np.testing.assert_array_equal(received, payload)

    reader.close()
    writer.close()


def test_unsafe_read_requires_explicit_lock(shm_name):
    writer = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)
    writer.write(np.array([1.0, 2.0], dtype=np.float32))

    with pytest.raises(RuntimeError, match="safe=False requires"):
        writer.read(safe=False)

    with writer.locked():
        raw = writer.read(safe=False)
        np.testing.assert_array_equal(
            raw, np.array([1.0, 2.0], dtype=np.float32)
        )

    writer.close()


def test_clear_resets_contents_to_zero(shm_name):
    shm = pyshmem.create(shm_name, shape=(2, 2), dtype=np.float32)
    shm.write(np.ones((2, 2), dtype=np.float32))

    initial_count = shm.count
    shm.clear()

    np.testing.assert_array_equal(
        shm.read(), np.zeros((2, 2), dtype=np.float32)
    )
    assert shm.count == initial_count + 1

    shm.close()


def test_delete_alias_unlinks_shared_memory(shm_name):
    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)
    shm.write(np.array([1.0, 2.0], dtype=np.float32))

    shm.delete()

    with pytest.raises(FileNotFoundError):
        pyshmem.open(shm_name)


def test_closed_handle_operations_raise_clear_errors(shm_name):
    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)
    shm.close()

    operations = (
        lambda: shm.read(),
        lambda: shm.read_new(timeout=0.01),
        lambda: shm.write(np.array([1.0, 2.0], dtype=np.float32)),
        lambda: shm.acquire(),
        lambda: shm.release(),
        lambda: shm.clear(),
        lambda: shm.count,
        lambda: shm.write_time,
        lambda: shm.write_sequence,
    )

    for operation in operations:
        with pytest.raises(RuntimeError, match="closed shared memory"):
            operation()


def test_close_is_idempotent(shm_name):
    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)

    shm.close()
    shm.close()


def test_process_crash_releases_lock(shm_name):
    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)
    shm.write(np.array([1.0, 2.0], dtype=np.float32))

    context = mp.get_context("spawn")
    event = context.Event()
    process = context.Process(
        target=_crash_while_holding_lock,
        args=(shm_name, event),
    )
    process.start()

    assert event.wait(timeout=5)
    process.join(timeout=20)
    assert process.exitcode == 0

    start = time.monotonic()
    shm.acquire(timeout=1.0)
    elapsed = time.monotonic() - start
    payload = shm.read()
    shm.release()

    assert elapsed < 0.5
    np.testing.assert_array_equal(
        payload, np.array([1.0, 2.0], dtype=np.float32)
    )

    shm.close()


@pytest.mark.skipif(
    WINDOWS_SHARED_MEMORY_IS_EPHEMERAL,
    reason=(
        "Windows destroys named shared memory when the last handle closes, "
        "so a segment cannot outlive its creator when no other handle is open"
    ),
)
def test_creator_exit_leaves_shared_memory_usable(shm_name):
    context = mp.get_context("spawn")
    event = context.Event()
    process = context.Process(
        target=_create_write_and_exit,
        args=(shm_name, event),
    )
    process.start()

    assert event.wait(timeout=5)
    process.join(timeout=20)
    assert process.exitcode == 0

    shm = pyshmem.open(shm_name)
    np.testing.assert_array_equal(
        shm.read(), np.full((2, 2), 7.0, dtype=np.float32)
    )
    shm.delete()


def test_release_without_acquire_reports_unlocked_state(shm_name):
    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)

    with pytest.raises(RuntimeError, match="unlocked"):
        shm.release()

    shm.close()


def test_reentrant_acquire_requires_balanced_release(shm_name):
    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)

    shm.acquire()
    shm.acquire()
    shm.write(np.array([3.0, 4.0], dtype=np.float32))
    np.testing.assert_array_equal(
        shm.read(safe=False), np.array([3.0, 4.0], dtype=np.float32)
    )
    shm.release()
    np.testing.assert_array_equal(
        shm.read(safe=False), np.array([3.0, 4.0], dtype=np.float32)
    )
    shm.release()

    with pytest.raises(RuntimeError, match="unlocked"):
        shm.release()

    shm.close()


@pytest.mark.skipif(
    WINDOWS_SHARED_MEMORY_IS_EPHEMERAL,
    reason=(
        "Windows destroys named shared memory when the last handle closes, "
        "so close-then-reopen is not possible without another live handle"
    ),
)
def test_close_and_reopen_preserves_repr_and_metadata(shm_name):
    writer = pyshmem.create(shm_name, shape=(3, 3), dtype=np.float32)
    writer.write(np.ones((3, 3), dtype=np.float32))
    writer.close()

    reopened = pyshmem.open(shm_name)

    assert repr(reopened) == (
        "SharedMemory(name='{}', shape=(3, 3), dtype='float32', "
        "gpu_device=None)"
    ).format(shm_name)
    assert reopened.shape == (3, 3)
    assert reopened.dtype == np.dtype(np.float32)
    assert reopened.count == 1

    reopened.close()


def test_create_rejects_mismatched_size(shm_name):
    with pytest.raises(ValueError, match="size does not match"):
        pyshmem.create(shm_name, shape=(2, 2), dtype=np.float32, size=12)


def test_create_reports_clear_error_for_existing_name(shm_name):
    writer = pyshmem.create(shm_name, shape=(2, 2), dtype=np.float32)

    with pytest.raises(FileExistsError, match="already exists") as exc_info:
        pyshmem.create(shm_name, shape=(2, 2), dtype=np.float32)

    assert f"use pyshmem.open({shm_name!r})" in str(exc_info.value)

    writer.close()


# ---------------------------------------------------------------------------
# GPU_SUPPORTED_DTYPES (#4)
# ---------------------------------------------------------------------------


def test_gpu_supported_dtypes_is_public_frozenset():
    assert isinstance(pyshmem.GPU_SUPPORTED_DTYPES, frozenset)


def test_gpu_supported_dtypes_contains_torch_mapped_types():
    for dtype in (
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
    ):
        assert np.dtype(dtype) in pyshmem.GPU_SUPPORTED_DTYPES


def test_gpu_supported_dtypes_excludes_non_gpu_integer_types():
    # uint16 / uint32 / uint64 have no direct torch equivalent.
    for dtype in (np.uint16, np.uint32, np.uint64):
        assert np.dtype(dtype) not in pyshmem.GPU_SUPPORTED_DTYPES


# ---------------------------------------------------------------------------
# list_streams (#5)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="list_streams is not supported on Windows",
)
def test_list_streams_returns_list():
    result = pyshmem.list_streams()
    assert isinstance(result, list)
    assert all(isinstance(s, str) for s in result)


@pytest.mark.skipif(
    sys.platform in ("win32", "darwin"),
    reason="list_streams requires /dev/shm (Linux only)",
)
def test_list_streams_includes_data_segment_of_created_stream(shm_name):
    expected = pyshmem_shared._segment_base_name(shm_name)

    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)
    try:
        assert expected in pyshmem.list_streams()
    finally:
        shm.close()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="list_streams is not supported on Windows",
)
def test_list_streams_segment_absent_after_unlink(shm_name):
    expected = pyshmem_shared._segment_base_name(shm_name)

    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)
    shm.unlink()

    assert expected not in pyshmem.list_streams()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="list_streams is not supported on Windows",
)
def test_list_streams_does_not_include_meta_or_gpu_segments(shm_name):
    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)
    try:
        streams = pyshmem.list_streams()
        assert not any(s.endswith("_meta") for s in streams)
        assert not any(s.endswith("_gpu") for s in streams)
    finally:
        shm.close()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="list_streams is not supported on Windows",
)
def test_list_streams_returns_sorted_results(shm_name):
    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)
    try:
        streams = pyshmem.list_streams()
        assert streams == sorted(streams)
    finally:
        shm.close()


# ---------------------------------------------------------------------------
# auto_unlink / stream() context manager (#6)
# ---------------------------------------------------------------------------


def test_create_auto_unlink_true_destroys_stream_on_context_exit(shm_name):
    with pyshmem.create(
        shm_name, shape=(2,), dtype=np.float32, auto_unlink=True
    ):
        pass

    with pytest.raises(FileNotFoundError):
        pyshmem.open(shm_name)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows named shared memory is freed when the last handle closes",
)
def test_create_auto_unlink_false_only_closes_on_context_exit(shm_name):
    with pyshmem.create(shm_name, shape=(2,), dtype=np.float32):
        pass  # auto_unlink defaults to False

    # Stream should still be accessible (POSIX keeps it alive until unlink).
    reopened = pyshmem.open(shm_name)
    reopened.close()


def test_stream_context_manager_unlinks_on_normal_exit(shm_name):
    with pyshmem.stream(shm_name, shape=(3,), dtype=np.float32) as shm:
        shm.write(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    with pytest.raises(FileNotFoundError):
        pyshmem.open(shm_name)


def test_stream_context_manager_unlinks_on_exception(shm_name):
    with pytest.raises(ValueError):
        with pyshmem.stream(shm_name, shape=(2,), dtype=np.float32):
            raise ValueError("deliberate")

    with pytest.raises(FileNotFoundError):
        pyshmem.open(shm_name)


def test_stream_context_manager_exposes_writable_shm(shm_name):
    payload = np.array([9.0, 8.0, 7.0], dtype=np.float32)

    with pyshmem.stream(shm_name, shape=(3,), dtype=np.float32) as shm:
        shm.write(payload)
        np.testing.assert_array_equal(shm.read(), payload)


# ---------------------------------------------------------------------------
# write_locked (#8)
# ---------------------------------------------------------------------------


def test_write_locked_writes_payload_when_lock_held(shm_name):
    shm = pyshmem.create(shm_name, shape=(3,), dtype=np.float32)

    with shm.locked():
        shm.write_locked(np.array([7.0, 8.0, 9.0], dtype=np.float32))

    np.testing.assert_array_equal(shm.read(), [7.0, 8.0, 9.0])
    shm.close()


def test_write_locked_increments_count(shm_name):
    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)
    baseline = shm.count

    with shm.locked():
        shm.write_locked(np.array([1.0, 2.0], dtype=np.float32))

    assert shm.count == baseline + 1
    shm.close()


def test_write_locked_raises_without_active_lock(shm_name):
    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)

    with pytest.raises(RuntimeError, match="write_locked"):
        shm.write_locked(np.array([1.0, 2.0], dtype=np.float32))

    shm.close()


def test_write_locked_rejects_wrong_shape(shm_name):
    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)

    with shm.locked():
        with pytest.raises(ValueError, match="shape"):
            shm.write_locked(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    shm.close()


def test_write_locked_is_visible_to_safe_read(shm_name):
    writer = pyshmem.create(shm_name, shape=(4,), dtype=np.float32)
    reader = pyshmem.open(shm_name)

    with writer.locked():
        writer.write_locked(np.arange(4, dtype=np.float32))

    np.testing.assert_array_equal(reader.read(), [0.0, 1.0, 2.0, 3.0])

    reader.close()
    writer.close()


# ---------------------------------------------------------------------------
# read(out=) (#9)
# ---------------------------------------------------------------------------


def test_read_with_out_writes_data_into_supplied_buffer(shm_name):
    shm = pyshmem.create(shm_name, shape=(4,), dtype=np.float32)
    payload = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    shm.write(payload)

    buf = np.zeros(4, dtype=np.float32)
    shm.read(out=buf)

    np.testing.assert_array_equal(buf, payload)
    shm.close()


def test_read_with_out_returns_the_same_buffer_object(shm_name):
    shm = pyshmem.create(shm_name, shape=(4,), dtype=np.float32)
    shm.write(np.zeros(4, dtype=np.float32))

    buf = np.zeros(4, dtype=np.float32)
    result = shm.read(out=buf)

    assert result is buf
    shm.close()


def test_read_with_out_stays_consistent_during_concurrent_writes(shm_name):
    writer = pyshmem.create(shm_name, shape=(16,), dtype=np.float32)
    reader = pyshmem.open(shm_name)
    stop_event = threading.Event()
    buf = np.zeros(16, dtype=np.float32)
    failures: list[str] = []

    def write_loop() -> None:
        for value in range(1, 200):
            writer.write(np.full(16, value, dtype=np.float32))
        stop_event.set()

    thread = threading.Thread(target=write_loop)
    thread.start()

    while not stop_event.is_set():
        snapshot = reader.read(out=buf)
        if not np.all(snapshot == snapshot[0]):
            failures.append("inconsistent")
            break

    thread.join()
    assert failures == []

    reader.close()
    writer.close()


# ---------------------------------------------------------------------------
# Lock directory isolation (#11)
# ---------------------------------------------------------------------------


def test_lock_path_default_includes_uid_prefix():
    uid = getattr(os, "getuid", lambda: 0)()
    path = pyshmem_shared._lock_path("any_stream")
    assert f"pyshmem-locks-{uid}" in path


def test_lock_path_uses_pyshmem_lock_dir_env_var(monkeypatch, tmp_path):
    custom_dir = str(tmp_path / "custom_locks")
    monkeypatch.setenv("PYSHMEM_LOCK_DIR", custom_dir)
    path = pyshmem_shared._lock_path("any_stream")
    assert path.startswith(custom_dir + os.sep)


def test_stream_still_lockable_with_custom_lock_dir(
    shm_name, monkeypatch, tmp_path
):
    custom_dir = str(tmp_path / "locks")
    monkeypatch.setenv("PYSHMEM_LOCK_DIR", custom_dir)

    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)
    shm.write(np.array([1.0, 2.0], dtype=np.float32))

    with shm.locked():
        result = shm.read(safe=False)
    np.testing.assert_array_equal(result, [1.0, 2.0])

    shm.close()


# ---------------------------------------------------------------------------
# describe (#13)
# ---------------------------------------------------------------------------


def test_describe_returns_string_containing_all_expected_fields(shm_name):
    shm = pyshmem.create(shm_name, shape=(3,), dtype=np.float32)
    shm.write(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    desc = shm.describe()

    assert isinstance(desc, str)
    for field in (
        "name:",
        "shape:",
        "dtype:",
        "size:",
        "gpu_enabled:",
        "gpu_device:",
        "cpu_mirror:",
        "count:",
        "write_time:",
        "write_seq:",
        "owner:",
    ):
        assert field in desc, f"missing field {field!r}"
    assert shm_name in desc

    shm.close()


def test_describe_reflects_current_write_count(shm_name):
    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)
    shm.write(np.zeros(2, dtype=np.float32))

    desc = shm.describe()
    assert "count:        1" in desc

    shm.close()


def test_describe_raises_on_closed_stream(shm_name):
    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)
    shm.close()

    with pytest.raises(RuntimeError, match="closed shared memory"):
        shm.describe()


# ---------------------------------------------------------------------------
# read_new write_sequence check (#3)
# ---------------------------------------------------------------------------


def test_read_new_waits_for_in_progress_write_to_complete(shm_name):
    """read_new must not return early when write_sequence is odd.

    (write in progress)
    """
    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)
    shm.write(np.zeros(2, dtype=np.float32))

    # Manually start a write without finishing it — sequence is now odd.
    shm._mark_write_started()
    assert shm.write_sequence % 2 == 1

    def finish_write_after_delay() -> None:
        time.sleep(0.05)
        np.copyto(shm._array, np.array([3.0, 4.0], dtype=np.float32))
        shm._finish_write()

    thread = threading.Thread(target=finish_write_after_delay)
    thread.start()

    start = time.monotonic()
    result = shm.read_new(timeout=1.0)
    elapsed = time.monotonic() - start
    thread.join()

    assert elapsed >= 0.04, "read_new returned before write completed"
    np.testing.assert_array_equal(result, [3.0, 4.0])

    shm.close()


# ---------------------------------------------------------------------------
# read_new_async (#15)
# ---------------------------------------------------------------------------


def test_read_new_async_returns_new_payload(shm_name):
    writer = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)
    reader = pyshmem.open(shm_name)
    writer.write(np.zeros(2, dtype=np.float32))

    async def _run():
        def delayed_write() -> None:
            time.sleep(0.05)
            writer.write(np.array([5.0, 6.0], dtype=np.float32))

        thread = threading.Thread(target=delayed_write)
        thread.start()
        start = time.monotonic()
        result = await reader.read_new_async(timeout=1.0)
        elapsed = time.monotonic() - start
        thread.join()

        assert elapsed >= 0.04
        np.testing.assert_array_equal(result, [5.0, 6.0])

    asyncio.run(_run())

    reader.close()
    writer.close()


def test_read_new_async_times_out_when_no_write_arrives(shm_name):
    shm = pyshmem.create(shm_name, shape=(1,), dtype=np.float32)

    async def _run():
        with pytest.raises(TimeoutError):
            await shm.read_new_async(timeout=0.05)

    asyncio.run(_run())

    shm.close()


def test_read_new_async_raises_on_closed_stream(shm_name):
    shm = pyshmem.create(shm_name, shape=(1,), dtype=np.float32)
    shm.close()

    async def _run():
        with pytest.raises(RuntimeError, match="closed shared memory"):
            await shm.read_new_async(timeout=0.01)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# to_config / create_from_config (#16)
# ---------------------------------------------------------------------------


def test_to_config_contains_expected_keys_and_values(shm_name):
    shm = pyshmem.create(shm_name, shape=(2, 3), dtype=np.float64)

    cfg = shm.to_config()

    assert cfg["name"] == shm_name
    assert cfg["shape"] == [2, 3]
    assert cfg["dtype"] == "float64"
    assert cfg["gpu_device"] is None
    assert cfg["cpu_mirror"] is True

    shm.close()


def test_to_config_dtype_is_plain_string(shm_name):
    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.int32)

    cfg = shm.to_config()

    assert isinstance(cfg["dtype"], str)
    assert cfg["dtype"] == "int32"

    shm.close()


def test_create_from_config_round_trip(shm_name):
    shm = pyshmem.create(shm_name, shape=(4,), dtype=np.int32)
    cfg = shm.to_config()
    shm.unlink()

    recreated = pyshmem.SharedMemory.create_from_config(cfg)

    assert recreated.name == shm_name
    assert recreated.shape == (4,)
    assert recreated.dtype == np.dtype(np.int32)
    assert recreated.gpu_device is None

    recreated.close()


def test_create_from_config_defaults_dtype_to_float32(shm_name):
    shm = pyshmem.SharedMemory.create_from_config(
        {"name": shm_name, "shape": [3]}
    )

    assert shm.dtype == np.dtype(np.float32)

    shm.close()


def test_create_from_config_write_read_round_trip(shm_name):
    original = pyshmem.create(shm_name, shape=(3,), dtype=np.float32)
    cfg = original.to_config()
    original.unlink()

    payload = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    shm = pyshmem.SharedMemory.create_from_config(cfg)
    shm.write(payload)
    np.testing.assert_array_almost_equal(shm.read(), payload)

    shm.close()


# ---------------------------------------------------------------------------
# CLI (#12)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="CLI list requires /dev/shm (Linux only)",
)
def test_cli_list_shows_created_stream_segment(shm_name, capsys):
    import pyshmem._cli as cli

    expected = pyshmem_shared._segment_base_name(shm_name)
    shm = pyshmem.create(shm_name, shape=(2,), dtype=np.float32)
    try:
        exit_code = cli.main(["list"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert expected in captured.out
    finally:
        shm.close()


def test_cli_unlink_destroys_named_stream(shm_name):
    import pyshmem._cli as cli

    pyshmem.create(shm_name, shape=(2,), dtype=np.float32).close()

    exit_code = cli.main(["unlink", shm_name])
    assert exit_code == 0

    with pytest.raises(FileNotFoundError):
        pyshmem.open(shm_name)


def test_cli_unlink_multiple_names(shm_name):
    import pyshmem._cli as cli

    name_a = f"{shm_name}_a"
    name_b = f"{shm_name}_b"
    pyshmem.create(name_a, shape=(2,), dtype=np.float32).close()
    pyshmem.create(name_b, shape=(2,), dtype=np.float32).close()

    try:
        exit_code = cli.main(["unlink", name_a, name_b])
        assert exit_code == 0

        with pytest.raises(FileNotFoundError):
            pyshmem.open(name_a)
        with pytest.raises(FileNotFoundError):
            pyshmem.open(name_b)
    finally:
        pyshmem.unlink(name_a)
        pyshmem.unlink(name_b)


def test_cli_no_command_exits_nonzero(capsys):
    import pyshmem._cli as cli

    exit_code = cli.main([])
    assert exit_code == 1
