from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading
import time

import numpy as np
import pytest

import pyshare


pytestmark = pytest.mark.cpu
WINDOWS_SHARED_MEMORY_IS_EPHEMERAL = sys.platform == "win32"


def _read_cpu_payload(name: str, queue) -> None:
    shm = pyshare.open(name)
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
    shm = pyshare.open(name)
    with shm.locked(timeout=1.0):
        queue.put("locked")
        time.sleep(hold_seconds)
    shm.close()


def _crash_while_holding_lock(name: str, event) -> None:
    shm = pyshare.open(name)
    shm.acquire(timeout=1.0)
    event.set()
    os._exit(0)


def _create_write_and_exit(name: str, event) -> None:
    shm = pyshare.create(name, shape=(2, 2), dtype=np.float32)
    shm.write(np.full((2, 2), 7.0, dtype=np.float32))
    event.set()
    shm.close()


def test_import_exposes_public_api():
    assert callable(pyshare.create)
    assert callable(pyshare.open)
    assert pyshare.SharedMemory is not None


def test_instance_does_not_expose_module_factory_methods(shm_name):
    shm = pyshare.create(shm_name, shape=(1,), dtype=np.float32)

    assert not hasattr(shm, "create")
    assert not hasattr(shm, "open")

    shm.close()


def test_create_write_read_round_trip_cpu(shm_name):
    shm = pyshare.create(shm_name, shape=(2, 3), dtype=np.float32)

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
        pyshare.open(shm_name)

    assert f"pyshare.create({shm_name!r}, ...)" in str(exc_info.value)


def test_open_reconstructs_shape_dtype_and_contents(shm_name):
    writer = pyshare.create(shm_name, shape=(4,), dtype=np.int32)
    payload = np.array([1, 2, 3, 4], dtype=np.int32)
    writer.write(payload)

    reader = pyshare.open(shm_name)

    assert reader.name == shm_name
    assert reader.shape == (4,)
    assert reader.dtype == np.dtype(np.int32)
    assert reader.size == payload.nbytes
    assert reader.gpu_device is None
    np.testing.assert_array_equal(reader.read(), payload)

    reader.close()
    writer.close()


def test_read_new_waits_for_next_write(shm_name):
    writer = pyshare.create(shm_name, shape=(2, 2), dtype=np.float32)
    reader = pyshare.open(shm_name)
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
    writer = pyshare.create(shm_name, shape=(1,), dtype=np.float32)
    reader = pyshare.open(shm_name)

    with pytest.raises(TimeoutError):
        reader.read_new(timeout=0.05)

    reader.close()
    writer.close()


def test_safe_reads_stay_consistent_during_concurrent_writes(shm_name):
    writer = pyshare.create(shm_name, shape=(32, 32), dtype=np.float32)
    reader = pyshare.open(shm_name)
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
    writer = pyshare.create(shm_name, shape=(2, 3), dtype=np.float32)
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


def test_cross_process_lock_blocks_explicit_acquire_until_release(shm_name):
    writer = pyshare.create(shm_name, shape=(4,), dtype=np.float32)
    payload = np.arange(4, dtype=np.float32)
    writer.write(payload)
    reader = pyshare.open(shm_name)

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
    writer = pyshare.create(shm_name, shape=(2,), dtype=np.float32)
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
    shm = pyshare.create(shm_name, shape=(2, 2), dtype=np.float32)
    shm.write(np.ones((2, 2), dtype=np.float32))

    initial_count = shm.count
    shm.clear()

    np.testing.assert_array_equal(
        shm.read(), np.zeros((2, 2), dtype=np.float32)
    )
    assert shm.count == initial_count + 1

    shm.close()


def test_delete_alias_unlinks_shared_memory(shm_name):
    shm = pyshare.create(shm_name, shape=(2,), dtype=np.float32)
    shm.write(np.array([1.0, 2.0], dtype=np.float32))

    shm.delete()

    with pytest.raises(FileNotFoundError):
        pyshare.open(shm_name)


def test_closed_handle_operations_raise_clear_errors(shm_name):
    shm = pyshare.create(shm_name, shape=(2,), dtype=np.float32)
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
    shm = pyshare.create(shm_name, shape=(2,), dtype=np.float32)

    shm.close()
    shm.close()


def test_process_crash_releases_lock(shm_name):
    shm = pyshare.create(shm_name, shape=(2,), dtype=np.float32)
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

    shm = pyshare.open(shm_name)
    np.testing.assert_array_equal(
        shm.read(), np.full((2, 2), 7.0, dtype=np.float32)
    )
    shm.delete()


def test_release_without_acquire_reports_unlocked_state(shm_name):
    shm = pyshare.create(shm_name, shape=(2,), dtype=np.float32)

    with pytest.raises(RuntimeError, match="unlocked"):
        shm.release()

    shm.close()


def test_reentrant_acquire_requires_balanced_release(shm_name):
    shm = pyshare.create(shm_name, shape=(2,), dtype=np.float32)

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
    writer = pyshare.create(shm_name, shape=(3, 3), dtype=np.float32)
    writer.write(np.ones((3, 3), dtype=np.float32))
    writer.close()

    reopened = pyshare.open(shm_name)

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
        pyshare.create(shm_name, shape=(2, 2), dtype=np.float32, size=12)


def test_create_reports_clear_error_for_existing_name(shm_name):
    writer = pyshare.create(shm_name, shape=(2, 2), dtype=np.float32)

    with pytest.raises(FileExistsError, match="already exists") as exc_info:
        pyshare.create(shm_name, shape=(2, 2), dtype=np.float32)

    assert f"use pyshare.open({shm_name!r})" in str(exc_info.value)

    writer.close()
