from __future__ import annotations

import os
import time

import numpy as np
import pytest

import pyshmem

try:
    import torch
except Exception:
    torch = None


CUDA_AVAILABLE = bool(torch is not None and pyshmem.gpu_available())


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _record_rate(
    *,
    record_property,
    property_name: str,
    label: str,
    iterations: int,
    elapsed: float,
) -> float:
    rate_hz = iterations / elapsed
    record_property(property_name, rate_hz)
    print(
        f"{label}: {rate_hz:.1f} Hz "
        f"({elapsed * 1e3 / iterations:.3f} ms/op)"
    )
    return rate_hz


def _contiguous_vector(index: int, length: int) -> np.ndarray:
    return np.linspace(index, index + 1.0, length, dtype=np.float32)


def _gpu_contiguous_vector(index: int, length: int) -> "torch.Tensor":
    return torch.linspace(
        float(index),
        float(index) + 1.0,
        length,
        dtype=torch.float32,
        device="cuda:0",
    )


@pytest.mark.cpu
@pytest.mark.benchmark
def test_cpu_roundtrip_rate_128_square(shm_name, record_property):
    writer = pyshmem.create(shm_name, shape=(128, 128), dtype=np.float32)
    reader = pyshmem.open(shm_name)
    payload = np.full((128, 128), 1.0, dtype=np.float32)
    iterations = _env_int("pyshmem_CPU_BENCHMARK_ITERATIONS", 2000)
    warmup_iterations = _env_int(
        "pyshmem_CPU_BENCHMARK_WARMUP_ITERATIONS", 200
    )

    for _ in range(warmup_iterations):
        writer.write(payload)
        reader.read()

    start = time.perf_counter()
    for _ in range(iterations):
        writer.write(payload)
        snapshot = reader.read()
    elapsed = time.perf_counter() - start

    target_hz = float(os.environ.get("pyshmem_TARGET_HZ", "50000"))
    enforce_target = os.environ.get("pyshmem_ENFORCE_BENCHMARK", "0") == "1"

    rate_hz = _record_rate(
        record_property=record_property,
        property_name="pyshmem_cpu_roundtrip_hz",
        label="pyshmem 128x128 CPU roundtrip rate",
        iterations=iterations,
        elapsed=elapsed,
    )

    assert snapshot.shape == (128, 128)
    assert snapshot[0, 0] == pytest.approx(1.0)
    if enforce_target:
        assert rate_hz >= target_hz

    writer.close()
    reader.close()


@pytest.mark.gpu
@pytest.mark.benchmark
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_roundtrip_rate_128_square(shm_name, record_property):
    writer = pyshmem.create(
        shm_name, shape=(128, 128), dtype=np.float32, gpu_device="cuda:0"
    )
    reader = pyshmem.open(shm_name, gpu_device="cuda:0")
    payload = np.full((128, 128), 1.0, dtype=np.float32)
    iterations = _env_int("pyshmem_GPU_BENCHMARK_ITERATIONS", 2000)
    warmup_iterations = _env_int(
        "pyshmem_GPU_BENCHMARK_WARMUP_ITERATIONS", 200
    )

    for _ in range(warmup_iterations):
        writer.write(payload)
        reader.read()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        writer.write(payload)
        snapshot = reader.read()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    target_hz = _env_float("pyshmem_GPU_TARGET_HZ", 0.0)
    enforce_target = (
        os.environ.get("pyshmem_ENFORCE_GPU_BENCHMARK", "0") == "1"
    )

    rate_hz = _record_rate(
        record_property=record_property,
        property_name="pyshmem_gpu_roundtrip_hz",
        label="pyshmem 128x128 GPU roundtrip rate",
        iterations=iterations,
        elapsed=elapsed,
    )

    assert tuple(snapshot.shape) == (128, 128)
    assert snapshot.device.type == "cuda"
    assert float(snapshot[0, 0].item()) == pytest.approx(1.0)
    if enforce_target:
        assert rate_hz >= target_hz

    reader.close()
    writer.close()


@pytest.mark.cpu
@pytest.mark.benchmark
def test_cpu_shared_memory_mvm_pipeline(shm_name, record_property):
    matrix_dim = _env_int("pyshmem_MVM_DIM", 1024)
    iterations = _env_int("pyshmem_CPU_MVM_ITERATIONS", 100)
    warmup_iterations = _env_int("pyshmem_CPU_MVM_WARMUP_ITERATIONS", 10)
    matrix_name = f"{shm_name}_matrix"
    vector_name = f"{shm_name}_vector"

    matrix = np.arange(matrix_dim * matrix_dim, dtype=np.float32).reshape(
        matrix_dim, matrix_dim
    )
    matrix /= float(matrix_dim)
    initial_vector = _contiguous_vector(0, matrix_dim)

    matrix_writer = pyshmem.create(
        matrix_name, shape=matrix.shape, dtype=np.float32
    )
    vector_writer = pyshmem.create(
        vector_name, shape=(matrix_dim,), dtype=np.float32
    )
    matrix_reader = pyshmem.open(matrix_name)
    vector_reader = pyshmem.open(vector_name)

    matrix_writer.write(matrix)
    vector_writer.write(initial_vector)

    for index in range(warmup_iterations):
        vector_writer.write(_contiguous_vector(index, matrix_dim))
        result = matrix_reader.read() @ vector_reader.read()

    start = time.perf_counter()
    for index in range(iterations):
        vector_writer.write(
            _contiguous_vector(index + warmup_iterations, matrix_dim)
        )
        result = matrix_reader.read() @ vector_reader.read()
    elapsed = time.perf_counter() - start

    expected = matrix @ _contiguous_vector(
        iterations + warmup_iterations - 1,
        matrix_dim,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    _record_rate(
        record_property=record_property,
        property_name="pyshmem_cpu_mvm_pipeline_hz",
        label=(
            "pyshmem CPU shared-memory MVM pipeline "
            f"({matrix_dim}x{matrix_dim})"
        ),
        iterations=iterations,
        elapsed=elapsed,
    )

    matrix_reader.close()
    vector_reader.close()
    matrix_writer.close()
    vector_writer.close()


@pytest.mark.gpu
@pytest.mark.benchmark
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_shared_memory_mvm_pipeline(shm_name, record_property):
    matrix_dim = _env_int("pyshmem_MVM_DIM", 1024)
    iterations = _env_int("pyshmem_GPU_MVM_ITERATIONS", 200)
    warmup_iterations = _env_int("pyshmem_GPU_MVM_WARMUP_ITERATIONS", 20)
    matrix_name = f"{shm_name}_matrix"
    vector_name = f"{shm_name}_vector"

    matrix = np.arange(matrix_dim * matrix_dim, dtype=np.float32).reshape(
        matrix_dim, matrix_dim
    )
    matrix /= float(matrix_dim)
    initial_vector = _contiguous_vector(0, matrix_dim)

    matrix_writer = pyshmem.create(
        matrix_name,
        shape=matrix.shape,
        dtype=np.float32,
        gpu_device="cuda:0",
    )
    vector_writer = pyshmem.create(
        vector_name,
        shape=(matrix_dim,),
        dtype=np.float32,
        gpu_device="cuda:0",
    )
    matrix_reader = pyshmem.open(matrix_name, gpu_device="cuda:0")
    vector_reader = pyshmem.open(vector_name, gpu_device="cuda:0")

    matrix_writer.write(matrix)
    vector_writer.write(initial_vector)

    for index in range(warmup_iterations):
        vector_writer.write(_contiguous_vector(index, matrix_dim))
        result = matrix_reader.read() @ vector_reader.read()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for index in range(iterations):
        vector_writer.write(
            _contiguous_vector(index + warmup_iterations, matrix_dim)
        )
        result = matrix_reader.read() @ vector_reader.read()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    expected = matrix @ _contiguous_vector(
        iterations + warmup_iterations - 1,
        matrix_dim,
    )
    np.testing.assert_allclose(
        result.detach().cpu().numpy(),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )

    _record_rate(
        record_property=record_property,
        property_name="pyshmem_gpu_mvm_pipeline_hz",
        label=(
            "pyshmem GPU shared-memory MVM pipeline "
            f"({matrix_dim}x{matrix_dim})"
        ),
        iterations=iterations,
        elapsed=elapsed,
    )

    matrix_reader.close()
    vector_reader.close()
    matrix_writer.close()
    vector_writer.close()


@pytest.mark.gpu
@pytest.mark.benchmark
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_device_resident_mvm_pipeline(shm_name, record_property):
    matrix_dim = _env_int(
        "pyshmem_GPU_DEVICE_MVM_DIM",
        _env_int("pyshmem_MVM_DIM", 1024),
    )
    iterations = _env_int("pyshmem_GPU_DEVICE_MVM_ITERATIONS", 200)
    warmup_iterations = _env_int(
        "pyshmem_GPU_DEVICE_MVM_WARMUP_ITERATIONS", 20
    )
    matrix_name = f"{shm_name}_matrix"
    vector_name = f"{shm_name}_vector"

    matrix = np.arange(matrix_dim * matrix_dim, dtype=np.float32).reshape(
        matrix_dim, matrix_dim
    )
    matrix /= float(matrix_dim)
    initial_vector = _gpu_contiguous_vector(0, matrix_dim)

    matrix_writer = pyshmem.create(
        matrix_name,
        shape=matrix.shape,
        dtype=np.float32,
        gpu_device="cuda:0",
    )
    vector_writer = pyshmem.create(
        vector_name,
        shape=(matrix_dim,),
        dtype=np.float32,
        gpu_device="cuda:0",
    )
    matrix_reader = pyshmem.open(matrix_name, gpu_device="cuda:0")
    vector_reader = pyshmem.open(vector_name, gpu_device="cuda:0")

    matrix_writer.write(matrix)
    vector_writer.write(initial_vector)

    for index in range(warmup_iterations):
        vector_writer.write(_gpu_contiguous_vector(index, matrix_dim))
        result = matrix_reader.read() @ vector_reader.read()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for index in range(iterations):
        vector_writer.write(
            _gpu_contiguous_vector(index + warmup_iterations, matrix_dim)
        )
        result = matrix_reader.read() @ vector_reader.read()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    rate_hz = _record_rate(
        record_property=record_property,
        property_name="pyshmem_gpu_device_resident_mvm_pipeline_hz",
        label=(
            "pyshmem GPU device-resident MVM pipeline "
            f"({matrix_dim}x{matrix_dim})"
        ),
        iterations=iterations,
        elapsed=elapsed,
    )

    expected = matrix @ _contiguous_vector(
        iterations + warmup_iterations - 1,
        matrix_dim,
    )
    np.testing.assert_allclose(
        result.detach().cpu().numpy(),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )
    record_property(
        "pyshmem_gpu_device_resident_mvm_pipeline_gflops",
        (2.0 * matrix_dim * matrix_dim * rate_hz) / 1e9,
    )

    matrix_reader.close()
    vector_reader.close()
    matrix_writer.close()
    vector_writer.close()
