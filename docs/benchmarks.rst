Benchmarks
==========

pyshmem includes benchmark-marked tests for both round-trip IO and matrix-vector
multiply pipeline workloads.

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