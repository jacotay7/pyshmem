from __future__ import annotations

import os
import time

import numpy as np
import pytest

import pyshare


pytestmark = [pytest.mark.cpu, pytest.mark.benchmark]


def test_cpu_roundtrip_rate_128_square(shm_name, record_property):
    writer = pyshare.create(shm_name, shape=(128, 128), dtype=np.float32)
    reader = pyshare.open(shm_name)
    payload = np.full((128, 128), 1.0, dtype=np.float32)
    iterations = 2000
    warmup_iterations = 200

    for _ in range(warmup_iterations):
        writer.write(payload)
        reader.read()

    start = time.perf_counter()
    for _ in range(iterations):
        writer.write(payload)
        snapshot = reader.read()
    elapsed = time.perf_counter() - start

    rate_hz = iterations / elapsed
    target_hz = float(os.environ.get("PYSHARE_TARGET_HZ", "50000"))
    enforce_target = os.environ.get("PYSHARE_ENFORCE_BENCHMARK", "0") == "1"

    record_property("pyshare_cpu_roundtrip_hz", rate_hz)
    print(f"pyshare 128x128 CPU roundtrip rate: {rate_hz:.1f} Hz")

    assert snapshot.shape == (128, 128)
    assert snapshot[0, 0] == pytest.approx(1.0)
    if enforce_target:
        assert rate_hz >= target_hz

    writer.close()
    reader.close()
