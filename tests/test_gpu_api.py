from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import pyshmem


torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu


CUDA_AVAILABLE = pyshmem.gpu_available()
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


def _read_gpu_payload(name: str, queue) -> None:
    shm = pyshmem.open(name, gpu_device="cuda:0")
    payload = shm.read()
    queue.put(
        {
            "device": payload.device.type,
            "gpu_device": shm.gpu_device,
            "shape": tuple(payload.shape),
            "dtype": str(shm.dtype),
            "size": shm.size,
            "values": payload.detach().cpu().tolist(),
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
    shm = pyshmem.create(
        name,
        shape=(2, 2),
        dtype=np.float32,
        gpu_device="cuda:0",
        cpu_mirror=True,
    )
    shm.write(np.full((2, 2), 7.0, dtype=np.float32))
    event.set()
    shm.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_create_write_read_round_trip_gpu(shm_name):
    shm = pyshmem.create(
        shm_name, shape=(2, 2), dtype=np.float32, gpu_device="cuda:0"
    )
    payload = np.arange(4, dtype=np.float32).reshape(2, 2)

    assert shm.name == shm_name
    assert shm.shape == (2, 2)
    assert shm.dtype == np.dtype(np.float32)
    assert shm.size == payload.nbytes
    assert shm.gpu_device == "cuda:0"
    assert shm.cpu_mirror is False

    shm.write(payload)
    received = shm.read()

    assert isinstance(received, torch.Tensor)
    assert received.device.type == "cuda"
    assert tuple(received.shape) == (2, 2)
    assert torch.equal(received.cpu(), torch.from_numpy(payload))

    shm.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_stream_without_cpu_mirror_requires_gpu_attachment_for_reads(
    shm_name,
):
    writer = pyshmem.create(
        shm_name, shape=(2, 2), dtype=np.float32, gpu_device="cuda:0"
    )
    writer.write(np.ones((2, 2), dtype=np.float32))
    reader = pyshmem.open(shm_name)

    assert reader.cpu_mirror is False

    with pytest.raises(RuntimeError, match="cpu_mirror=True"):
        reader.read()

    reader.close()
    writer.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_stream_with_cpu_mirror_can_be_read_without_gpu_attachment(
    shm_name,
):
    writer = pyshmem.create(
        shm_name,
        shape=(2, 2),
        dtype=np.float32,
        gpu_device="cuda:0",
        cpu_mirror=True,
    )
    payload = np.arange(4, dtype=np.float32).reshape(2, 2)
    writer.write(payload)

    reader = pyshmem.open(shm_name)

    assert reader.cpu_mirror is True
    received = reader.read()

    assert isinstance(received, np.ndarray)
    np.testing.assert_array_equal(received, payload)

    reader.close()
    writer.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_open_reports_clear_error_for_missing_name(shm_name):
    with pytest.raises(FileNotFoundError, match="does not exist") as exc_info:
        pyshmem.open(shm_name)

    assert f"pyshmem.create({shm_name!r}, ...)" in str(exc_info.value)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_open_reconstructs_shape_dtype_and_contents(shm_name):
    writer = pyshmem.create(
        shm_name, shape=(4,), dtype=np.int32, gpu_device="cuda:0"
    )
    payload = np.array([1, 2, 3, 4], dtype=np.int32)
    writer.write(payload)

    reader = pyshmem.open(shm_name, gpu_device="cuda:0")

    assert reader.name == shm_name
    assert reader.shape == (4,)
    assert reader.dtype == np.dtype(np.int32)
    assert reader.size == payload.nbytes
    assert reader.gpu_device == "cuda:0"
    received = reader.read()

    assert isinstance(received, torch.Tensor)
    assert received.device.type == "cuda"
    assert torch.equal(received.cpu(), torch.from_numpy(payload))

    reader.close()
    writer.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_read_new_waits_for_next_write(shm_name):
    writer = pyshmem.create(
        shm_name, shape=(2, 2), dtype=np.float32, gpu_device="cuda:0"
    )
    reader = pyshmem.open(shm_name, gpu_device="cuda:0")
    writer.write(np.zeros((2, 2), dtype=np.float32))
    assert torch.equal(
        reader.read().cpu(), torch.zeros((2, 2), dtype=torch.float32)
    )

    def delayed_write() -> None:
        time.sleep(0.05)
        writer.write(np.ones((2, 2), dtype=np.float32))

    thread = threading.Thread(target=delayed_write)
    thread.start()
    start = time.monotonic()
    received = reader.read_new(timeout=1.0)
    elapsed = time.monotonic() - start
    thread.join()

    assert elapsed >= 0.04
    assert torch.equal(received.cpu(), torch.ones((2, 2), dtype=torch.float32))

    reader.close()
    writer.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_read_new_times_out_when_no_new_write_arrives(shm_name):
    writer = pyshmem.create(
        shm_name, shape=(1,), dtype=np.float32, gpu_device="cuda:0"
    )
    reader = pyshmem.open(shm_name)

    with pytest.raises(TimeoutError):
        reader.read_new(timeout=0.05)

    reader.close()
    writer.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_safe_reads_stay_consistent_during_concurrent_writes(shm_name):
    writer = pyshmem.create(
        shm_name,
        shape=(16, 16),
        dtype=np.float32,
        gpu_device="cuda:0",
        cpu_mirror=True,
    )
    reader = pyshmem.open(shm_name, gpu_device="cuda:0")
    stop_event = threading.Event()
    failures: list[str] = []

    def write_loop() -> None:
        for value in range(1, 100):
            writer.write(np.full((16, 16), value, dtype=np.float32))
        stop_event.set()

    thread = threading.Thread(target=write_loop)
    thread.start()

    while not stop_event.is_set():
        snapshot = reader.read().cpu()
        if not torch.all(snapshot == snapshot[0, 0]):
            failures.append("inconsistent snapshot")
            break

    thread.join()
    assert failures == []

    reader.close()
    writer.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_stream_can_be_opened_in_another_process(shm_name):
    writer = pyshmem.create(
        shm_name, shape=(2, 2), dtype=np.float32, gpu_device="cuda:0"
    )
    payload = np.arange(4, dtype=np.float32).reshape(2, 2)
    writer.write(payload)

    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_read_gpu_payload, args=(shm_name, queue))
    process.start()
    process.join(timeout=20)

    assert process.exitcode == 0
    message = queue.get(timeout=5)
    assert message["device"] == "cuda"
    assert message["gpu_device"] == "cuda:0"
    assert message["shape"] == (2, 2)
    assert message["dtype"] == "float32"
    assert message["size"] == payload.nbytes
    assert message["values"] == payload.tolist()

    writer.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_attached_process_exit_keeps_stream_attachable(shm_name):
    writer = pyshmem.create(
        shm_name,
        shape=(2, 2),
        dtype=np.float32,
        gpu_device="cuda:0",
        cpu_mirror=True,
    )
    payload = np.full((2, 2), 3.0, dtype=np.float32)
    writer.write(payload)

    child = _run_python_child(
        "import pyshmem; "
        f"shm = pyshmem.open({shm_name!r}, gpu_device='cuda:0'); "
        "print(shm.read().detach().cpu().tolist())"
    )

    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == str(payload.tolist())
    assert "resource_tracker" not in child.stderr

    reopened = pyshmem.open(shm_name, gpu_device="cuda:0")
    assert torch.equal(reopened.read().cpu(), torch.from_numpy(payload))

    reopened.close()
    writer.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_cross_process_lock_blocks_explicit_acquire_until_release(shm_name):
    writer = pyshmem.create(
        shm_name, shape=(4,), dtype=np.float32, gpu_device="cuda:0"
    )
    payload = np.arange(4, dtype=np.float32)
    writer.write(payload)
    reader = pyshmem.open(shm_name, gpu_device="cuda:0")

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
    assert torch.equal(received.cpu(), torch.from_numpy(payload))

    reader.close()
    writer.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_unsafe_read_requires_explicit_lock(shm_name):
    writer = pyshmem.create(
        shm_name, shape=(2,), dtype=np.float32, gpu_device="cuda:0"
    )
    writer.write(np.array([1.0, 2.0], dtype=np.float32))

    with pytest.raises(RuntimeError, match="safe=False requires"):
        writer.read(safe=False)

    with writer.locked():
        raw = writer.read(safe=False)
        assert isinstance(raw, torch.Tensor)
        assert torch.equal(raw.cpu(), torch.tensor([1.0, 2.0]))

    writer.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_clear_resets_contents_to_zero(shm_name):
    shm = pyshmem.create(
        shm_name, shape=(2, 2), dtype=np.float32, gpu_device="cuda:0"
    )
    shm.write(np.ones((2, 2), dtype=np.float32))

    initial_count = shm.count
    shm.clear()

    assert torch.equal(
        shm.read().cpu(), torch.zeros((2, 2), dtype=torch.float32)
    )
    assert shm.count == initial_count + 1

    shm.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_delete_alias_unlinks_shared_memory(shm_name):
    shm = pyshmem.create(
        shm_name, shape=(2,), dtype=np.float32, gpu_device="cuda:0"
    )
    shm.write(np.array([1.0, 2.0], dtype=np.float32))

    shm.delete()

    with pytest.raises(FileNotFoundError):
        pyshmem.open(shm_name)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_closed_handle_operations_raise_clear_errors(shm_name):
    shm = pyshmem.create(
        shm_name, shape=(2,), dtype=np.float32, gpu_device="cuda:0"
    )
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


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_close_is_idempotent(shm_name):
    shm = pyshmem.create(
        shm_name, shape=(2,), dtype=np.float32, gpu_device="cuda:0"
    )

    shm.close()
    shm.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_process_crash_releases_lock(shm_name):
    shm = pyshmem.create(
        shm_name, shape=(2,), dtype=np.float32, gpu_device="cuda:0"
    )
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
    assert torch.equal(payload.cpu(), torch.tensor([1.0, 2.0]))

    shm.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
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


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_release_without_acquire_reports_unlocked_state(shm_name):
    shm = pyshmem.create(
        shm_name, shape=(2,), dtype=np.float32, gpu_device="cuda:0"
    )

    with pytest.raises(RuntimeError, match="unlocked"):
        shm.release()

    shm.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_reentrant_acquire_requires_balanced_release(shm_name):
    shm = pyshmem.create(
        shm_name, shape=(2,), dtype=np.float32, gpu_device="cuda:0"
    )

    shm.acquire()
    shm.acquire()
    shm.write(np.array([3.0, 4.0], dtype=np.float32))
    assert torch.equal(shm.read(safe=False).cpu(), torch.tensor([3.0, 4.0]))
    shm.release()
    assert torch.equal(shm.read(safe=False).cpu(), torch.tensor([3.0, 4.0]))
    shm.release()

    with pytest.raises(RuntimeError, match="unlocked"):
        shm.release()

    shm.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_close_and_reopen_preserves_repr_and_metadata(shm_name):
    writer = pyshmem.create(
        shm_name, shape=(3, 3), dtype=np.float32, gpu_device="cuda:0"
    )
    writer.write(np.ones((3, 3), dtype=np.float32))
    writer.close()

    reopened = pyshmem.open(shm_name, gpu_device="cuda:0")

    assert repr(reopened) == (
        "SharedMemory(name='{}', shape=(3, 3), dtype='float32', "
        "gpu_device='cuda:0')"
    ).format(shm_name)
    assert reopened.shape == (3, 3)
    assert reopened.dtype == np.dtype(np.float32)
    assert reopened.count == 1
    assert reopened.gpu_device == "cuda:0"

    reopened.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_create_rejects_mismatched_size(shm_name):
    with pytest.raises(ValueError, match="size does not match"):
        pyshmem.create(
            shm_name,
            shape=(2, 2),
            dtype=np.float32,
            size=12,
            gpu_device="cuda:0",
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_create_reports_clear_error_for_existing_name(shm_name):
    writer = pyshmem.create(
        shm_name, shape=(2, 2), dtype=np.float32, gpu_device="cuda:0"
    )

    with pytest.raises(FileExistsError, match="already exists") as exc_info:
        pyshmem.create(
            shm_name, shape=(2, 2), dtype=np.float32, gpu_device="cuda:0"
        )

    assert f"use pyshmem.open({shm_name!r})" in str(exc_info.value)

    writer.close()


# ---------------------------------------------------------------------------
# GPU handle segment leak fix (#2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_create_removes_gpu_tensor_cache_entry_on_post_creation_failure(
    shm_name, monkeypatch
):
    """If _create raises after _create_gpu_tensor_and_handle, the tensor
    weakref must be removed from _LOCAL_GPU_TENSORS so no stale entry remains.
    """
    import pyshmem._shared as pyshmem_shared

    original_create_gpu = pyshmem_shared._create_gpu_tensor_and_handle

    def patched_create_gpu(*, name, shape, torch_dtype, gpu_device):
        original_create_gpu(
            name=name,
            shape=shape,
            torch_dtype=torch_dtype,
            gpu_device=gpu_device,
        )
        # At this point _cache_gpu_tensor has been called.
        assert name in pyshmem_shared._LOCAL_GPU_TENSORS
        # Raise to simulate a subsequent failure in metadata initialisation.
        raise RuntimeError("injected post-gpu failure")

    monkeypatch.setattr(
        pyshmem_shared, "_create_gpu_tensor_and_handle", patched_create_gpu
    )

    with pytest.raises(RuntimeError, match="injected post-gpu failure"):
        pyshmem.create(
            shm_name, shape=(2,), dtype=np.float32, gpu_device="cuda:0"
        )

    assert shm_name not in pyshmem_shared._LOCAL_GPU_TENSORS


# ---------------------------------------------------------------------------
# Thread-safety in GPU open (#1)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_concurrent_gpu_opens_in_same_process_return_same_tensor(shm_name):
    """Two threads opening the same GPU stream simultaneously must end up
    with the same underlying GPU tensor (cache hit), not two separate tensors
    aliasing the same CUDA IPC memory.
    """
    creator = pyshmem.create(
        shm_name, shape=(4,), dtype=np.float32, gpu_device="cuda:0"
    )
    creator.write(np.arange(4, dtype=np.float32))

    handles: list[pyshmem.SharedMemory] = []
    errors: list[str] = []

    def open_and_collect() -> None:
        try:
            handles.append(pyshmem.open(shm_name, gpu_device="cuda:0"))
        except Exception as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=open_and_collect) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"errors during concurrent open: {errors}"
    assert len(handles) == 8

    # All handles must agree on the data.
    for handle in handles:
        result = handle.read()
        assert torch.equal(result.cpu(), torch.arange(4, dtype=torch.float32))
        handle.close()

    creator.close()


# ---------------------------------------------------------------------------
# write_locked on GPU streams (#8)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_write_locked_writes_payload_when_lock_held(shm_name):
    shm = pyshmem.create(
        shm_name, shape=(3,), dtype=np.float32, gpu_device="cuda:0"
    )

    with shm.locked():
        shm.write_locked(np.array([7.0, 8.0, 9.0], dtype=np.float32))

    received = shm.read()
    assert torch.equal(received.cpu(), torch.tensor([7.0, 8.0, 9.0]))

    shm.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_write_locked_raises_without_active_lock(shm_name):
    shm = pyshmem.create(
        shm_name, shape=(2,), dtype=np.float32, gpu_device="cuda:0"
    )

    with pytest.raises(RuntimeError, match="write_locked"):
        shm.write_locked(np.array([1.0, 2.0], dtype=np.float32))

    shm.close()


# ---------------------------------------------------------------------------
# describe on GPU streams (#13)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_describe_includes_gpu_fields(shm_name):
    shm = pyshmem.create(
        shm_name, shape=(3,), dtype=np.float32, gpu_device="cuda:0"
    )
    shm.write(np.ones(3, dtype=np.float32))

    desc = shm.describe()

    assert "gpu_enabled:  True" in desc
    assert "cuda:0" in desc
    assert "cpu_mirror:   False" in desc

    shm.close()


# ---------------------------------------------------------------------------
# to_config / create_from_config on GPU streams (#16)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_to_config_includes_gpu_device_and_cpu_mirror(shm_name):
    shm = pyshmem.create(
        shm_name, shape=(2,), dtype=np.float32, gpu_device="cuda:0"
    )

    cfg = shm.to_config()

    assert cfg["gpu_device"] == "cuda:0"
    assert cfg["cpu_mirror"] is False

    shm.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_create_from_config_round_trip(shm_name):
    shm = pyshmem.create(
        shm_name, shape=(4,), dtype=np.float32, gpu_device="cuda:0"
    )
    cfg = shm.to_config()
    shm.unlink()

    recreated = pyshmem.SharedMemory.create_from_config(cfg)

    assert recreated.name == shm_name
    assert recreated.shape == (4,)
    assert recreated.dtype == np.dtype(np.float32)
    assert recreated.gpu_device == "cuda:0"

    recreated.close()


# ---------------------------------------------------------------------------
# stream() context manager on GPU (#6)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_stream_context_manager_unlinks_on_exit(shm_name):
    with pyshmem.stream(
        shm_name, shape=(2,), dtype=np.float32, gpu_device="cuda:0"
    ) as shm:
        shm.write(np.ones(2, dtype=np.float32))

    with pytest.raises(FileNotFoundError):
        pyshmem.open(shm_name)
