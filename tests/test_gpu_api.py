from __future__ import annotations

import json
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
import pyshmem._shared as pyshmem_shared


torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu


CUDA_AVAILABLE = pyshmem.gpu_available()
TEST_SRC_PATH = str(Path(__file__).resolve().parents[1] / "src")


def test_numpy_gpu_write_source_stays_on_cpu():
    source = np.arange(4, dtype=np.float32)
    tensor = pyshmem_shared._gpu_write_source(source, torch.float32)
    assert tensor.device.type == "cpu"
    assert tensor.data_ptr() == source.ctypes.data


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
@pytest.mark.parametrize(
    "dtype,values",
    [
        (np.uint16, [1, 2]),
        (np.uint32, [1, 2]),
        (np.uint64, [1, 2]),
        (np.bool_, [True, False]),
        (np.complex64, [1 + 2j, 3 + 4j]),
        (np.complex128, [1 + 2j, 3 + 4j]),
    ],
)
def test_capability_driven_gpu_dtype_round_trip(shm_name, dtype, values):
    if np.dtype(dtype) not in pyshmem.GPU_SUPPORTED_DTYPES:
        pytest.skip("installed torch does not expose this dtype")
    shm = pyshmem.create(
        shm_name, shape=(2,), dtype=dtype, gpu_device="cuda:0"
    )
    payload = np.asarray(values, dtype=dtype)
    try:
        shm.write(payload)
        np.testing.assert_array_equal(shm.read().cpu().numpy(), payload)
    finally:
        shm.unlink()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_pinned_buffer_is_reused_and_writable(shm_name):
    shm = pyshmem.create(
        shm_name,
        shape=(4,),
        dtype=np.float32,
        gpu_device="cuda:0",
    )
    try:
        staging = shm.pinned_buffer()
        assert staging.device.type == "cpu"
        assert staging.is_pinned()
        assert shm.pinned_buffer().data_ptr() == staging.data_ptr()

        staging.numpy()[:] = [1, 2, 3, 4]
        shm.write(staging)
        torch.testing.assert_close(
            shm.read(),
            torch.arange(1, 5, device="cuda:0", dtype=torch.float32),
        )
    finally:
        shm.unlink()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_readonly_gpu_handle_can_snapshot_but_not_mutate(shm_name):
    writer = pyshmem.create(
        shm_name, shape=(2,), dtype=np.float32, gpu_device="cuda:0"
    )
    writer.write(torch.tensor([1, 2], device="cuda:0", dtype=torch.float32))
    reader = pyshmem.open(shm_name, readonly=True)
    try:
        torch.testing.assert_close(
            reader.read(),
            torch.tensor([1, 2], device="cuda:0", dtype=torch.float32),
        )
        with pytest.raises(PermissionError, match="read-only"):
            reader.write(torch.zeros(2, device="cuda:0"))
        with pytest.raises(PermissionError, match="read-only"):
            reader.pinned_buffer()
    finally:
        reader.close()
        writer.unlink()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_dlpack_gpu_roundtrips_on_device(shm_name):
    shm = pyshmem.create(
        shm_name, shape=(3,), dtype=np.float32, gpu_device="cuda:0"
    )
    try:
        shm.write(
            torch.tensor([4, 5, 6], device="cuda:0", dtype=torch.float32)
        )
        device_type, device_id = shm.__dlpack_device__()
        assert int(device_type) == 2  # kDLCUDA
        assert device_id == 0
        exported = torch.from_dlpack(shm)
        assert exported.device.type == "cuda"
        torch.testing.assert_close(
            exported,
            torch.tensor([4, 5, 6], device="cuda:0", dtype=torch.float32),
        )
        # Snapshot semantics: a later write does not mutate the export.
        shm.write(torch.zeros(3, device="cuda:0", dtype=torch.float32))
        torch.testing.assert_close(
            exported,
            torch.tensor([4, 5, 6], device="cuda:0", dtype=torch.float32),
        )
    finally:
        shm.unlink()


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


def _attach_release_worker(src_path, name, ready_q, go_q, done_q, rounds):
    """Persistent consumer: each round, attach to the stream, read, release.

    Reusing one process keeps torch imported/CUDA-initialised once so the
    leak-regression test stays fast.  Each ``close`` releases the consumer's
    CUDA IPC mapping, which is what lets the producer reclaim GPU memory.
    """
    import sys

    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    import pyshmem

    ready_q.put("ready")
    for _ in range(rounds):
        go_q.get()
        shm = pyshmem.open(name, gpu_device="cuda:0")
        _ = shm.read()
        shm.close()
        done_q.put("released")


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
def test_gpu_handle_reconstructs_through_restricted_unpickler(shm_name):
    # A legitimate torch reduction payload must still round-trip: the GPU
    # handle is deserialized via the restricted unpickler, not raw
    # pickle.loads.
    writer = pyshmem.create(
        shm_name, shape=(3,), dtype=np.float32, gpu_device="cuda:0"
    )
    writer.write(torch.tensor([4.0, 5.0, 6.0], device="cuda:0"))
    reader = pyshmem.open(shm_name)
    assert torch.equal(reader.read().cpu(), torch.tensor([4.0, 5.0, 6.0]))
    reader.close()
    writer.close()
    pyshmem.unlink(shm_name)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_open_rejects_tampered_gpu_handle_payload(shm_name):
    import pickle

    writer = pyshmem.create(
        shm_name, shape=(3,), dtype=np.float32, gpu_device="cuda:0"
    )
    writer.write(torch.zeros(3, device="cuda:0"))

    # A same-account attacker overwrites the writable 0600 handle segment with
    # a payload that would execute code under raw pickle.loads.  The restricted
    # unpickler must refuse it rather than run it.  Reconstruction only runs in
    # a non-creator process, so a child performs the open.
    class _Evil:
        def __reduce__(self):
            return (os.system, ("echo tampered",))

    evil = pickle.dumps(_Evil(), protocol=4)
    handle = pyshmem_shared._attach_segment(
        pyshmem_shared._gpu_handle_name(shm_name)
    )
    try:
        handle.buf[: len(evil)] = evil
    finally:
        handle.close()

    child = _run_python_child(f"import pyshmem; pyshmem.open({shm_name!r})")
    assert child.returncode != 0, child.stdout
    assert "disallowed global in GPU handle payload" in child.stderr

    writer.close()
    pyshmem.unlink(shm_name)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_cuda_failure_during_publication_is_recoverable(shm_name, monkeypatch):
    shm = pyshmem.create(
        shm_name, shape=(4,), dtype=np.float32, gpu_device="cuda:0"
    )
    shm.write(torch.zeros(4, device="cuda:0"))

    # A CUDA error surfacing during publication (here at synchronize, after
    # the device copy is enqueued) must leave the stream in the invalid state,
    # not a half-published one, and a later good write must repair it.
    real_sync = pyshmem_shared._synchronize_cuda_operation

    def failing_sync(*args, **kwargs):
        raise RuntimeError("injected cuda failure")

    monkeypatch.setattr(
        pyshmem_shared, "_synchronize_cuda_operation", failing_sync
    )
    with pytest.raises(RuntimeError, match="injected cuda failure"):
        shm.write(torch.ones(4, device="cuda:0"))
    monkeypatch.setattr(
        pyshmem_shared, "_synchronize_cuda_operation", real_sync
    )

    assert shm.write_sequence < 0
    with pytest.raises(pyshmem.InconsistentStreamError):
        shm.read(timeout=1.0)

    replacement = torch.full((4,), 5.0, device="cuda:0")
    shm.write(replacement)
    assert shm.write_sequence > 0 and shm.write_sequence % 2 == 0
    assert torch.equal(shm.read().cpu(), replacement.cpu())

    shm.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_operations_do_not_synchronize_whole_device(shm_name, monkeypatch):
    shm = pyshmem.create(
        shm_name, shape=(4,), dtype=np.float32, gpu_device="cuda:0"
    )

    def forbidden_device_sync(*args, **kwargs):
        raise AssertionError("whole-device synchronization is forbidden")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden_device_sync)
    try:
        shm.write(torch.arange(4, device="cuda:0", dtype=torch.float32))
        torch.testing.assert_close(
            shm.read(), torch.arange(4, device="cuda:0", dtype=torch.float32)
        )
        shm.clear()
    finally:
        shm.unlink()
    pyshmem.unlink(shm_name)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_open_auto_attaches_to_stored_gpu_device(shm_name):
    # open() reconstructs the stream as created: it attaches to the CUDA device
    # recorded in metadata without the caller having to pass gpu_device.
    writer = pyshmem.create(
        shm_name, shape=(2, 2), dtype=np.float32, gpu_device="cuda:0"
    )
    writer.write(np.ones((2, 2), dtype=np.float32))

    reader = pyshmem.open(shm_name)  # no gpu_device passed

    assert reader.gpu_enabled is True
    assert reader.gpu_device == "cuda:0"
    received = reader.read()
    assert isinstance(received, torch.Tensor)
    assert received.device.type == "cuda"
    assert torch.equal(received.cpu(), torch.ones((2, 2)))

    reader.close()
    writer.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_open_auto_attaches_gpu_even_with_cpu_mirror(shm_name):
    # A cpu_mirror stream is still reconstructed with its GPU attachment on a
    # CUDA host (read returns a CUDA tensor, matching how it was created).
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

    assert reader.gpu_device == "cuda:0"
    assert reader.cpu_mirror is True
    received = reader.read()
    assert isinstance(received, torch.Tensor)
    np.testing.assert_array_equal(received.cpu().numpy(), payload)

    reader.close()
    writer.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_open_cpu_only_reads_mirror_even_when_cuda_available(shm_name):
    # gpu_device=False opts out of attaching the producer's CUDA tensor and
    # reads the host mirror as a NumPy array, even on a CUDA-capable host.
    writer = pyshmem.create(
        shm_name,
        shape=(2, 2),
        dtype=np.float32,
        gpu_device="cuda:0",
        cpu_mirror=True,
    )
    payload = np.arange(4, dtype=np.float32).reshape(2, 2)
    writer.write(payload)

    reader = pyshmem.open(shm_name, gpu_device=False)

    assert reader.gpu_device is None  # no GPU attachment
    assert reader.cpu_mirror is True
    received = reader.read()
    assert isinstance(received, np.ndarray)
    np.testing.assert_array_equal(received, payload)

    reader.close()
    writer.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_open_cpu_only_raises_without_cpu_mirror(shm_name):
    # A GPU stream with no mirror cannot be opened CPU-only.
    writer = pyshmem.create(
        shm_name, shape=(2, 2), dtype=np.float32, gpu_device="cuda:0"
    )
    writer.write(np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="without a CPU mirror"):
        pyshmem.open(shm_name, gpu_device=False)

    writer.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_open_falls_back_to_cpu_mirror_when_cuda_unavailable(
    shm_name, monkeypatch
):
    # When CUDA cannot be attached but the stream has a CPU mirror, open()
    # returns a usable CPU handle instead of raising.
    writer = pyshmem.create(
        shm_name,
        shape=(2, 2),
        dtype=np.float32,
        gpu_device="cuda:0",
        cpu_mirror=True,
    )
    payload = np.arange(4, dtype=np.float32).reshape(2, 2)
    writer.write(payload)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reader = pyshmem.open(shm_name)

    assert reader.gpu_device is None  # no GPU attachment
    received = reader.read()
    assert isinstance(received, np.ndarray)
    np.testing.assert_array_equal(received, payload)

    reader.close()
    writer.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_open_raises_when_gpu_unavailable_and_no_cpu_mirror(
    shm_name, monkeypatch
):
    # A GPU-only stream cannot be reconstructed without CUDA and has no mirror
    # to fall back on, so open() raises a clear error.
    writer = pyshmem.create(
        shm_name, shape=(2, 2), dtype=np.float32, gpu_device="cuda:0"
    )
    writer.write(np.ones((2, 2), dtype=np.float32))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        pyshmem.open(shm_name)

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


def _read_gpu_payload_auto(name: str, queue) -> None:
    # Open WITHOUT gpu_device: it must auto-attach to the stored CUDA device.
    shm = pyshmem.open(name)
    payload = shm.read()
    queue.put(
        {
            "device": payload.device.type,
            "gpu_device": shm.gpu_device,
            "values": payload.detach().cpu().tolist(),
        }
    )
    shm.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_stream_auto_attaches_in_another_process(shm_name):
    writer = pyshmem.create(
        shm_name, shape=(2, 2), dtype=np.float32, gpu_device="cuda:0"
    )
    payload = np.arange(4, dtype=np.float32).reshape(2, 2)
    writer.write(payload)

    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_read_gpu_payload_auto, args=(shm_name, queue)
    )
    process.start()
    process.join(timeout=20)

    assert process.exitcode == 0
    message = queue.get(timeout=5)
    assert message["device"] == "cuda"
    assert message["gpu_device"] == "cuda:0"
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


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_write_view_publishes_shared_tensor_and_mirror(shm_name):
    writer = pyshmem.create(
        shm_name,
        shape=(3,),
        dtype=np.float32,
        gpu_device="cuda:0",
        cpu_mirror=True,
    )
    reader = pyshmem.open(shm_name, gpu_device="cuda:0")
    mirror = pyshmem.open(shm_name, gpu_device=False)

    with writer.write_view() as view:
        view.copy_(torch.tensor([2.0, 4.0, 6.0], device="cuda:0"))

    assert torch.equal(reader.read().cpu(), torch.tensor([2.0, 4.0, 6.0]))
    np.testing.assert_array_equal(mirror.read(), [2.0, 4.0, 6.0])

    mirror.close()
    reader.close()
    writer.unlink()


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


# ---------------------------------------------------------------------------
# GPU memory / CUDA IPC cleanup
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_owner_keeps_tensor_on_close_but_drops_on_unlink(shm_name):
    owner = pyshmem.create(
        shm_name, shape=(4,), dtype=np.float32, gpu_device="cuda:0"
    )
    assert owner._gpu_tensor is not None
    owner.close()
    # The owner keeps its tensor after close so the stream stays mappable and
    # can be reopened in-process.
    assert owner._gpu_tensor is not None
    owner.unlink()
    # Destroying the stream releases the producer's tensor so the GPU
    # allocation can be reclaimed.
    assert owner._gpu_tensor is None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_consumer_releases_tensor_on_close(shm_name):
    owner = pyshmem.create(
        shm_name, shape=(4,), dtype=np.float32, gpu_device="cuda:0"
    )
    try:
        consumer = pyshmem.open(shm_name, gpu_device="cuda:0")
        assert consumer._gpu_tensor is not None
        consumer.close()
        # A consumer must drop its mapping on close so torch decrements the
        # producer's IPC ref counter (otherwise the producer can never reclaim
        # the GPU allocation).
        assert consumer._gpu_tensor is None
    finally:
        owner.unlink()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_producer_memory_reclaimed_across_attach_release_cycles(shm_name):
    import gc

    torch.cuda.init()
    rounds = 12
    elements = 512 * 512
    nbytes = elements * 4  # float32

    ctx = mp.get_context("spawn")
    ready_q, go_q, done_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    worker = ctx.Process(
        target=_attach_release_worker,
        args=(TEST_SRC_PATH, shm_name, ready_q, go_q, done_q, rounds),
    )
    worker.start()
    try:
        assert ready_q.get(timeout=60) == "ready"

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        baseline = torch.cuda.memory_allocated()

        for _ in range(rounds):
            shm = pyshmem.create(
                shm_name,
                shape=(512, 512),
                dtype=np.float32,
                gpu_device="cuda:0",
            )
            shm.write(torch.ones(512, 512, device="cuda:0"))
            go_q.put("go")
            assert done_q.get(timeout=60) == "released"
            shm.unlink()
            del shm
            gc.collect()

        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        leaked = torch.cuda.memory_allocated() - baseline
        # With correct CUDA IPC bookkeeping the producer reclaims every
        # allocation after the consumer releases it; allow a small slack for
        # caching-allocator residue but not 12 leaked tensors.
        assert leaked <= 3 * nbytes, (
            f"GPU memory grew by {leaked} bytes over {rounds} "
            f"attach/release/unlink cycles (~{leaked / nbytes:.1f} leaked "
            "tensors); CUDA IPC cleanup is broken"
        )
    finally:
        for _ in range(rounds):
            try:
                go_q.put("go")
            except Exception:
                break
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_purge_preserves_cuda_ipc_files_of_live_stream(shm_name):
    import glob

    owner = pyshmem.create(
        shm_name, shape=(16,), dtype=np.float32, gpu_device="cuda:0"
    )
    try:
        owner.write(torch.ones(16, device="cuda:0"))
        # Only inspect IPC files produced by *this* (live) process; earlier
        # tests' dead subprocesses may have left orphans that the sweep should
        # (correctly) remove.
        my_pid = os.getpid()
        my_files = [
            path
            for path in glob.glob("/dev/shm/cuda.shm.*")
            if pyshmem_shared._cuda_ipc_file_producer_pid(
                os.path.basename(path)
            )
            == my_pid
        ]
        assert my_files, "expected a cuda.shm.* file for the live GPU stream"

        removed = pyshmem_shared._remove_orphaned_cuda_ipc_files()

        # The sweep must not touch IPC files backing this live process.
        for path in my_files:
            assert os.path.exists(path), f"removed live IPC file {path}"
            assert os.path.basename(path) not in removed
        # The tensor is still intact after the sweep.
        assert torch.allclose(owner.read(), torch.ones(16, device="cuda:0"))
    finally:
        owner.unlink()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_safe_read_waits_for_even_write_sequence(shm_name):
    shm = pyshmem.create(
        shm_name, shape=(4,), dtype=np.float32, gpu_device="cuda:0"
    )
    shm.write(torch.zeros(4, device="cuda:0"))
    shm._mark_write_started()

    def finish_write():
        time.sleep(0.05)
        shm._gpu_tensor.copy_(torch.arange(4, device="cuda:0"))
        torch.cuda.synchronize()
        shm._finish_write()

    writer = threading.Thread(target=finish_write)
    writer.start()
    start = time.monotonic()
    result = shm.read(timeout=1.0)
    elapsed = time.monotonic() - start
    writer.join()

    assert elapsed >= 0.04
    assert torch.equal(result.cpu(), torch.arange(4, dtype=torch.float32))
    shm.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_read_rejects_cpu_out_buffer(shm_name):
    shm = pyshmem.create(
        shm_name, shape=(4,), dtype=np.float32, gpu_device="cuda:0"
    )
    with pytest.raises(ValueError, match="safe CPU reads"):
        shm.read(out=np.empty(4, dtype=np.float32))
    shm.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_spawned_process_gpu_benchmark_smoke():
    # Exercises the spawned-process GPU IPC baseline end to end: a producer
    # publishes a CUDA tensor and a separate process maps it over torch IPC and
    # acks, so this validates both the harness and the cross-process GPU path.
    script = Path(__file__).parents[1] / "benchmarks" / "benchmark_ipc.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--payload-bytes",
            "256",
            "--minimum-seconds",
            "0.005",
            "--repeats",
            "1",
            "--gpu",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    result = json.loads(completed.stdout)
    assert "pyshmem_gpu" in result["results"]
    gpu = result["results"]["pyshmem_gpu"]
    assert gpu["samples"] >= 100
    assert gpu["latency_us"]["p99"] > 0
    assert "torch" in result["environment"]
    assert "cuda_device" in result["environment"]
