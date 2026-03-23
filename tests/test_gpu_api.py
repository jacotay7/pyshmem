from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

import pyshare


torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu


CUDA_AVAILABLE = pyshare.gpu_available()


def _read_gpu_payload(name: str, queue) -> None:
    shm = pyshare.open(name, gpu_device="cuda:0")
    payload = shm.read()
    queue.put(
        {
            "device": payload.device.type,
            "shape": tuple(payload.shape),
            "sum": float(payload.detach().cpu().sum().item()),
        }
    )
    shm.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_stream_read_returns_torch_tensor(shm_name):
    shm = pyshare.create(
        shm_name, shape=(2, 2), dtype=np.float32, gpu_device="cuda:0"
    )
    payload = np.arange(4, dtype=np.float32).reshape(2, 2)

    shm.write(payload)
    received = shm.read()

    assert isinstance(received, torch.Tensor)
    assert received.device.type == "cuda"
    assert tuple(received.shape) == (2, 2)
    assert torch.equal(received.cpu(), torch.from_numpy(payload))

    shm.close()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_stream_can_be_opened_in_another_process(shm_name):
    writer = pyshare.create(
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
    assert message["shape"] == (2, 2)
    assert message["sum"] == pytest.approx(float(payload.sum()))

    writer.close()
