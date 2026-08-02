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

Pass ``--gpu`` to additionally measure a spawned-process GPU baseline
(``pyshmem_gpu``): a separate process maps the producer's CUDA tensor over
torch's IPC handle and reads a consistent device snapshot each round trip. It is
auto-included when CUDA is available and can be forced off with ``--no-gpu``.

.. code-block:: bash

   python benchmarks/benchmark_ipc.py \
       --payload-bytes 65536 --minimum-seconds 1 --repeats 5 --gpu \
       --output benchmarks/results/my-machine.json

The checked-in ``benchmarks/results/rtx5090-linux-2026-07-10.json`` records a
64 KiB run on the primary Linux development machine (Python 3.12, NumPy 2.2.6,
PyTorch 2.10, RTX 5090):

.. list-table::
   :header-rows: 1

   * - Implementation
     - Round trips/s
     - p50
     - p95
     - p99
   * - pyshmem (CPU)
     - 13,988
     - 60.26 us
     - 111.69 us
     - 115.70 us
   * - pyshmem (GPU IPC)
     - 4,872
     - 189.25 us
     - 239.00 us
     - 240.98 us
   * - Raw shared memory polling
     - 16,492
     - 57.58 us
     - 104.45 us
     - 108.95 us

These are one-machine observations, not universal performance claims; rerun the
harness on the intended deployment host.

CUDA completion-event reuse
---------------------------

``benchmarks/results/quadro-p620-cuda-event-reuse-2026-08-01.json`` records a
targeted seven-repetition comparison of newly allocated versus per-handle
reused completion events. On that device the event boundary alone improves
from 7.011 to 5.689 microseconds (18.86%). Including pyshmem locking, payload
copy, consistency, and publication for a 228-by-228 float64 stream, synchronous
writes improve 2.12% and safe reads improve 4.96%. This is an implementation
comparison on one CUDA/PyTorch pair, not a portable threshold.

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
