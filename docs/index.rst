pyshmem documentation
=====================

pyshmem provides named shared-memory streams for CPU NumPy arrays and optional
CUDA-backed PyTorch pipelines. The library is designed for applications that
need low-latency data exchange between processes while preserving a simple,
uniform API.

Highlights:

- **One primitive** — a named, fixed-shape, typed *stream* in shared memory.
- **CPU and GPU, same API** — NumPy arrays or CUDA tensors behind identical
  ``create`` / ``open`` / ``read`` / ``write`` calls.
- **Attach by name** — consumers open a stream knowing only its name; shape and
  dtype are recovered from metadata.
- **Lock-free consistent reads** plus explicit cross-process locking and
  zero-copy access when you need it.
- **Persistent** — streams survive the creating process on POSIX, with a CLI
  (``pyshmem list`` / ``unlink`` / ``purge``) to manage them.

New here?  Start with the :doc:`quickstart`.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   overview
   installation
   usage
   gpu
   cli
   platforms
   benchmarks
   api
   release

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`