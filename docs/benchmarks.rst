Benchmarks
==========

pyshmem includes benchmark-marked tests for both round-trip IO and matrix-vector
multiply pipeline workloads.

Spawned-process IPC benchmark
-----------------------------

``benchmarks/benchmark_ipc.py`` measures an actual producer/consumer process
boundary using the multiprocessing ``spawn`` start method. It performs a
request/acknowledgement ping-pong, calibrates iteration count to a requested
minimum duration, repeats the run, and emits versioned JSON with throughput and
p50/p95/p99 round-trip latency. A raw
``multiprocessing.shared_memory`` polling implementation is included as a
lower-bound copy/poll baseline; unlike pyshmem, it deliberately provides no
locking, consistency snapshots, discovery, or metadata validation.

.. code-block:: bash

   python benchmarks/benchmark_ipc.py \
       --payload-bytes 65536 --minimum-seconds 1 --repeats 5 \
       --output benchmarks/results/my-machine.json

The checked-in ``benchmarks/results/rtx5090-linux-2026-07-10.json`` records a
64 KiB run on the primary Linux development machine. It measured pyshmem at
11,683 round trips/s (p50 77.69 us, p95 139.38 us, p99 140.03 us), versus the
unsafe raw baseline at 18,096 round trips/s (p50 52.67 us, p95 62.86 us, p99
107.37 us). These are one-machine observations, not universal performance
claims; rerun the harness on the intended deployment host.

Running benchmarks locally
--------------------------

CPU benchmark smoke tests:

.. code-block:: bash

   pytest -m "cpu and benchmark" -q -s

GPU benchmark smoke tests:

.. code-block:: bash

   pytest tests/test_benchmark.py -m "gpu and benchmark" -q -s

GPU pipeline shapes
-------------------

Two GPU MVM benchmark shapes are included:

- host-upload pipeline: the vector is produced in NumPy and uploaded each iteration
- device-resident pipeline: the vector is produced directly on GPU each iteration

Environment knobs
-----------------

- ``pyshmem_MVM_DIM``
- ``pyshmem_CPU_MVM_ITERATIONS``
- ``pyshmem_GPU_MVM_ITERATIONS``
- ``pyshmem_GPU_DEVICE_MVM_DIM``
- ``pyshmem_GPU_DEVICE_MVM_ITERATIONS``
- ``pyshmem_GPU_DEVICE_MVM_WARMUP_ITERATIONS``

See the project README for the current measured results captured on the primary
development machine.
