Overview
========

pyshmem gives you a single interface for two related use cases:

- CPU shared-memory streams backed by NumPy arrays
- GPU shared-memory streams backed by CUDA tensors through PyTorch

The design goal is straightforward: move structured numeric payloads between
processes without forcing every application to invent its own lock protocol,
metadata layout, or CPU/GPU branching logic.

Core capabilities
-----------------

- named shared-memory streams with shape and dtype metadata
- cross-process write locking
- safe snapshot reads for CPU streams
- optional CUDA-backed streams for GPU pipelines
- explicit control over CPU mirroring for GPU streams

Public API
----------

The public package surface is intentionally small:

- ``pyshmem.create``
- ``pyshmem.open``
- ``pyshmem.unlink``
- ``pyshmem.gpu_available``
- ``pyshmem.SharedMemory``

If you are starting fresh, the best path is to read :doc:`installation` and
then :doc:`usage`.