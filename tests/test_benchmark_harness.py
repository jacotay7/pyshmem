from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import pyshmem

_BENCHMARK_SCRIPT = (
    Path(__file__).parents[1] / "benchmarks" / "benchmark_ipc.py"
)


def _load_benchmark_ipc():
    # Load the script by path rather than ``from benchmarks import ...``: the
    # repo root is not on sys.path under the ``pytest`` console script (only
    # under ``python -m pytest``), so a package import would fail in CI.
    spec = importlib.util.spec_from_file_location(
        "pyshmem_benchmark_ipc", _BENCHMARK_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark_ipc = _load_benchmark_ipc()

pytestmark = [pytest.mark.cpu, pytest.mark.benchmark]


def test_spawned_process_benchmark_smoke():
    completed = subprocess.run(
        [
            sys.executable,
            str(_BENCHMARK_SCRIPT),
            "--payload-bytes",
            "128",
            "--minimum-seconds",
            "0.005",
            "--repeats",
            "1",
            # Keep the CPU smoke fast and deterministic across CUDA / non-CUDA
            # hosts; the GPU baseline has its own gpu-marked test.
            "--no-gpu",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(completed.stdout)
    assert result["schema_version"] == 1
    assert result["benchmark"] == "spawned_process_ping_pong"
    assert set(result["results"]) == {"raw_shared_memory", "pyshmem"}
    for measurement in result["results"].values():
        assert measurement["samples"] >= 100
        assert measurement["latency_us"]["p99"] > 0


def test_pyshmem_consumer_does_not_miss_a_write_it_did_not_wait_for(shm_name):
    """The consumer must acknowledge a request that was published before it
    started waiting.  Edge-triggered ``read_new`` snapshots its baseline at
    call time and would deadlock here; the level-triggered count wait cannot.
    """
    request = pyshmem.create(f"{shm_name}_req", shape=(8,), dtype=np.uint8)
    ack = pyshmem.create(f"{shm_name}_ack", shape=(1,), dtype=np.int64)
    try:
        # Publish the request *before* the consumer is running, so no amount
        # of waiting inside the consumer can observe a fresh edge.
        request.write(np.arange(8, dtype=np.uint8))
        ready = threading.Event()
        consumer = threading.Thread(
            target=benchmark_ipc._pyshmem_consumer,
            args=(request.name, ack.name, 1, ready),
        )
        consumer.start()
        assert ready.wait(5.0)
        consumer.join(5.0)
        assert not consumer.is_alive(), "consumer deadlocked on a missed write"
        deadline = time.monotonic() + 1.0
        while ack.count < 1 and time.monotonic() < deadline:
            time.sleep(1e-4)
        assert ack.count == 1
        np.testing.assert_array_equal(ack.read(), np.array([1]))
    finally:
        request.unlink()
        ack.unlink()
