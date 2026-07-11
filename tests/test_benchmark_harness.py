from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

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
