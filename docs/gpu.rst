GPU Streams
===========

GPU streams require a CUDA-capable PyTorch installation.  Install the optional
extra to get it:

.. code-block:: bash

   pip install pyshmem[gpu]

Check that the GPU path is usable before creating streams:

.. code-block:: python

   import pyshmem
   print(pyshmem.gpu_available())   # True / False

Supported dtypes
----------------

Only dtypes with a direct PyTorch equivalent are accepted for GPU streams:

.. code-block:: python

   pyshmem.GPU_SUPPORTED_DTYPES
   # frozenset({float16, float32, float64, int8, int16, int32, int64, uint8})

``uint16``, ``uint32``, and ``uint64`` have no PyTorch counterpart.  Passing
them to :func:`pyshmem.create` with ``gpu_device`` set raises
:class:`ValueError` at construction time, before any shared-memory segments are
allocated.

.. code-block:: python

   import numpy as np

   np.dtype("float32") in pyshmem.GPU_SUPPORTED_DTYPES   # True
   np.dtype("uint32")  in pyshmem.GPU_SUPPORTED_DTYPES   # False

Creating a GPU stream
---------------------

.. code-block:: python

   import numpy as np
   import pyshmem

   shm = pyshmem.create(
       "activations",
       shape=(4096, 4096),
       dtype=np.float32,
       gpu_device="cuda:0",
   )

CPU mirroring modes
-------------------

GPU streams support two deliberately different operating modes, selected at
creation time by the ``cpu_mirror`` parameter.

Performance mode (default)
^^^^^^^^^^^^^^^^^^^^^^^^^^

``cpu_mirror=False`` is the default for GPU streams.

.. code-block:: python

   shm = pyshmem.create(
       "activations",
       shape=(4096,),
       dtype=np.float32,
       gpu_device="cuda:0",
       # cpu_mirror=False is implied
   )

Behaviour:

- No CPU mirror is maintained.  Every write goes straight to the CUDA tensor.
- Fastest path — avoids the GPU→CPU copy on every write.
- CPU-only handles (opened without ``gpu_device=``) can still inspect metadata
  and take locks, but ``read()`` and ``write()`` raise :class:`RuntimeError`.
- Intended for GPU-heavy pipelines where throughput matters most.

Compatibility mode
^^^^^^^^^^^^^^^^^^

``cpu_mirror=True`` keeps a CPU copy in sync with the GPU tensor on every
write.

.. code-block:: python

   shm = pyshmem.create(
       "activations",
       shape=(4096,),
       dtype=np.float32,
       gpu_device="cuda:0",
       cpu_mirror=True,
   )

Behaviour:

- A NumPy array backed by the CPU data segment is updated on every write.
- CPU-only readers (opened without ``gpu_device=``) can call ``read()`` and
  get a consistent CPU snapshot.
- Safe-snapshot semantics (the odd/even write-sequence consistency check) also
  apply to the CPU mirror, so ``read(safe=True)`` works correctly under
  concurrent writes.
- Trades throughput for compatibility.

Opening GPU streams in another process
---------------------------------------

Always pass ``gpu_device`` when you want a CUDA tensor view:

.. code-block:: python

   reader = pyshmem.open("activations", gpu_device="cuda:0")
   tensor = reader.read()   # torch.Tensor on cuda:0

Omitting ``gpu_device`` gives a CPU-only handle.  On a stream created with
``cpu_mirror=True``, this still allows ``read()`` and ``write()``.  On a stream
without a CPU mirror, it restricts the handle to metadata and locking only.

Thread safety
-------------

Multiple threads in the same process can open the same GPU stream concurrently.
pyshmem serialises the CUDA IPC handle reconstruction with a per-stream lock,
so all threads that race to open the same stream will share one reconstructed
tensor rather than producing aliased copies.

GPU handle reconstruction happens at most once per (stream, process) pair.
Subsequent opens in the same process reuse the cached
:class:`torch.Tensor` from an internal weakref cache.

Platform note
-------------

GPU IPC via ``torch.UntypedStorage._share_cuda_()`` has been tested on Linux.
macOS does not support CUDA.  Windows is not tested for GPU streams.
