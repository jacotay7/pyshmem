#!/usr/bin/env python3
"""Reproducible spawned-process pyshmem IPC latency benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import platform
import statistics
import sys
import time
import uuid

import numpy as np

import pyshmem


def _pyshmem_consumer(request_name, ack_name, iterations, ready):
    request = pyshmem.open(request_name)
    ack = pyshmem.open(ack_name)
    ready.set()
    try:
        for generation in range(1, iterations + 1):
            # Level-triggered wait on the publication count.  Unlike
            # edge-triggered read_new (which snapshots a baseline at call
            # time), this cannot miss a request that the producer published
            # in the window between our previous ack and this poll, so the
            # ping-pong never deadlocks under scheduler pressure.
            deadline = time.monotonic() + 10.0
            while request.count < generation:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for request gen {generation}"
                    )
                time.sleep(0)
            request.read()
            ack.write(np.array([generation], dtype=np.int64))
    finally:
        request.close()
        ack.close()


def _raw_consumer(request_name, ack_name, payload_bytes, iterations, ready):
    request = shared_memory.SharedMemory(name=request_name)
    ack = shared_memory.SharedMemory(name=ack_name)
    generation = np.ndarray((1,), dtype=np.uint64, buffer=request.buf[:8])
    payload = np.ndarray(
        (payload_bytes,), dtype=np.uint8, buffer=request.buf[8:]
    )
    ack_value = np.ndarray((1,), dtype=np.uint64, buffer=ack.buf)
    snapshot = np.empty_like(payload)
    ready.set()
    try:
        for expected in range(1, iterations + 1):
            while int(generation[0]) != expected:
                time.sleep(0)
            np.copyto(snapshot, payload)
            ack_value[0] = expected
    finally:
        del generation, payload, ack_value, snapshot
        request.close()
        ack.close()


def _run_pyshmem(iterations: int, payload_bytes: int) -> list[int]:
    suffix = uuid.uuid4().hex
    request = pyshmem.create(
        f"bench_request_{suffix}", shape=(payload_bytes,), dtype=np.uint8
    )
    ack = pyshmem.create(f"bench_ack_{suffix}", shape=(1,), dtype=np.int64)
    payload = np.arange(payload_bytes, dtype=np.uint8)
    ready = mp.get_context("spawn").Event()
    process = mp.get_context("spawn").Process(
        target=_pyshmem_consumer,
        args=(request.name, ack.name, iterations, ready),
    )
    process.start()
    latencies = []
    try:
        if not ready.wait(20.0):
            raise RuntimeError("pyshmem consumer did not become ready")
        for generation in range(1, iterations + 1):
            started = time.perf_counter_ns()
            request.write(payload)
            # Level-triggered wait on the ack count, mirroring the raw
            # baseline.  Polling the published counter (rather than the
            # edge-triggered read_new baseline) cannot miss an ack that the
            # consumer publishes before we start waiting, so the round-trip
            # is deterministic under load.
            deadline = time.monotonic() + 10.0
            while ack.count < generation:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for ack generation {generation}"
                    )
                time.sleep(0)
            latencies.append(time.perf_counter_ns() - started)
        process.join(20.0)
        if process.exitcode != 0:
            raise RuntimeError(
                f"pyshmem consumer exited with code {process.exitcode}"
            )
        return latencies
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        request.unlink()
        ack.unlink()


def _run_raw(iterations: int, payload_bytes: int) -> list[int]:
    request = shared_memory.SharedMemory(create=True, size=8 + payload_bytes)
    ack = shared_memory.SharedMemory(create=True, size=8)
    generation = np.ndarray((1,), dtype=np.uint64, buffer=request.buf[:8])
    payload = np.ndarray(
        (payload_bytes,), dtype=np.uint8, buffer=request.buf[8:]
    )
    ack_value = np.ndarray((1,), dtype=np.uint64, buffer=ack.buf)
    generation[0] = 0
    ack_value[0] = 0
    source = np.arange(payload_bytes, dtype=np.uint8)
    ready = mp.get_context("spawn").Event()
    process = mp.get_context("spawn").Process(
        target=_raw_consumer,
        args=(request.name, ack.name, payload_bytes, iterations, ready),
    )
    process.start()
    latencies = []
    try:
        if not ready.wait(20.0):
            raise RuntimeError(
                "raw shared-memory consumer did not become ready"
            )
        for expected in range(1, iterations + 1):
            started = time.perf_counter_ns()
            np.copyto(payload, source)
            generation[0] = expected
            while int(ack_value[0]) != expected:
                time.sleep(0)
            latencies.append(time.perf_counter_ns() - started)
        process.join(20.0)
        if process.exitcode != 0:
            raise RuntimeError(
                f"raw consumer exited with code {process.exitcode}"
            )
        return latencies
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        del generation, payload, ack_value
        request.close()
        ack.close()
        request.unlink()
        ack.unlink()


def _percentile(values: list[int], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summary(samples: list[int], repeats: int, iterations: int) -> dict:
    seconds = sum(samples) / 1e9
    return {
        "repeats": repeats,
        "iterations_per_repeat": iterations,
        "samples": len(samples),
        "roundtrips_per_second": len(samples) / seconds,
        "latency_us": {
            "mean": statistics.fmean(samples) / 1e3,
            "p50": _percentile(samples, 0.50) / 1e3,
            "p95": _percentile(samples, 0.95) / 1e3,
            "p99": _percentile(samples, 0.99) / 1e3,
        },
    }


def _calibrate(runner, payload_bytes: int, minimum_seconds: float) -> int:
    probe_iterations = 100
    elapsed = sum(runner(probe_iterations, payload_bytes)) / 1e9
    estimate = int(probe_iterations * minimum_seconds / max(elapsed, 1e-9))
    return max(100, min(100_000, estimate))


def run_benchmark(payload_bytes: int, minimum_seconds: float, repeats: int):
    results = {}
    for name, runner in (
        ("raw_shared_memory", _run_raw),
        ("pyshmem", _run_pyshmem),
    ):
        iterations = _calibrate(runner, payload_bytes, minimum_seconds)
        samples = []
        for _ in range(repeats):
            samples.extend(runner(iterations, payload_bytes))
        results[name] = _summary(samples, repeats, iterations)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "spawned_process_ping_pong",
        "payload_bytes": payload_bytes,
        "minimum_repeat_seconds": minimum_seconds,
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyshmem": pyshmem.__version__,
            "pid": os.getpid(),
        },
        "results": results,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-bytes", type=int, default=65_536)
    parser.add_argument("--minimum-seconds", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=str)
    args = parser.parse_args(argv)
    if args.payload_bytes < 1 or args.minimum_seconds <= 0 or args.repeats < 1:
        parser.error("payload bytes, duration, and repeats must be positive")
    result = run_benchmark(
        args.payload_bytes, args.minimum_seconds, args.repeats
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
