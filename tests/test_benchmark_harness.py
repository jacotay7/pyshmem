from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import pyshmem
from benchmarks import benchmark_ipc

pytestmark = [pytest.mark.cpu, pytest.mark.benchmark]


def test_spawned_process_benchmark_smoke():
    script = Path(__file__).parents[1] / "benchmarks" / "benchmark_ipc.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--payload-bytes",
            "128",
            "--minimum-seconds",
            "0.005",
            "--repeats",
            "1",
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
