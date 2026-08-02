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

GPU dtypes are capability-driven: pyshmem accepts every dtype in its persistent
format table that the installed PyTorch exposes directly:

.. code-block:: python

   pyshmem.GPU_SUPPORTED_DTYPES
   # exact contents depend on the installed torch version

Current PyTorch releases expose unsigned 16/32/64-bit integers, booleans, and
complex types in addition to the original signed integer and floating types.
Older supported PyTorch releases may expose a smaller set. Passing a dtype not
listed by :data:`pyshmem.GPU_SUPPORTED_DTYPES` raises :class:`ValueError` at
construction time, before shared-memory segments are allocated.

.. code-block:: python

   import numpy as np

   np.dtype("float32") in pyshmem.GPU_SUPPORTED_DTYPES   # True
   np.dtype("uint32") in pyshmem.GPU_SUPPORTED_DTYPES

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
- NumPy arrays and CPU tensors copy directly into shared CUDA storage without a
  temporary CUDA tensor and second device-to-device copy. Writes remain
  synchronous; pageable inputs are not automatically staged through pinned
  memory.
- A process without access to the stream's CUDA device cannot read it: there is
  no mirror to fall back on, so :func:`~pyshmem.open` raises a clear error.
- Intended for GPU-heavy pipelines where throughput matters most.

Pinned staging
^^^^^^^^^^^^^^

For repeated host-to-GPU writes, :meth:`~pyshmem.SharedMemory.pinned_buffer`
returns one reusable page-locked CPU tensor with the stream's shape and dtype.
Fill it directly, or fill its zero-copy NumPy view, then publish it:

.. code-block:: python

   staging = shm.pinned_buffer()
   staging.numpy()[:] = next_numpy_frame
   shm.write(staging)

The synchronous write is complete before it returns, so the same staging buffer
can be refilled immediately. On the primary development machine, repeated 4 MB
writes from this buffer measured a 149.67 us median versus 171.70 us from a
pageable NumPy array. Measure on the intended hardware; pinning too much host
memory can reduce overall system performance.

CUDA synchronization
^^^^^^^^^^^^^^^^^^^^

Synchronous reads, writes, and clears record and wait for a CUDA event on the
active stream. They do not call whole-device ``torch.cuda.synchronize()``, so
unrelated work in other streams is not drained. The public operation still
returns only after its own copy is complete and the host publication counter is
safe for another process to observe. A future fully asynchronous API requires
an interprocess event/publication protocol; simply returning before updating
metadata would expose incomplete payloads.

Each handle reuses one completion event per process, thread, and active CUDA
stream after the prior wait completes. Concurrent threads and distinct streams
therefore never record the same event, while steady synchronous traffic avoids
per-operation event construction. Closing the handle releases its event cache.

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

:func:`~pyshmem.open` reconstructs the stream as it was created.  For a GPU
stream it **auto-attaches to the CUDA device recorded in metadata** — you do
not need to pass ``gpu_device``:

.. code-block:: python

   reader = pyshmem.open("activations")    # auto-attaches to its cuda:N
   tensor = reader.read()                  # torch.Tensor on that device

An explicit ``gpu_device="cuda:N"`` is validated against the stored device and
must match.

To read the **CPU mirror** of a ``cpu_mirror=True`` stream as a NumPy array —
without attaching the producer's CUDA tensor, even on a CUDA-capable host —
pass ``gpu_device=False``:

.. code-block:: python

   reader = pyshmem.open("activations", gpu_device=False)
   frame = reader.read()                   # numpy.ndarray from the mirror

This is the supported way for CPU-side consumers (viewers, loggers, external
tooling) to read a GPU stream's host mirror from a process that happens to also
have CUDA available.  It raises :class:`ValueError` if the stream has no CPU
mirror.

Fallback behaviour
^^^^^^^^^^^^^^^^^^

When ``gpu_device`` is left at its default and the CUDA device cannot be
attached — because this process has no CUDA, or because the producer has
exited and released its tensor — :func:`~pyshmem.open` falls back to the CPU
mirror if the stream has one (returning a NumPy-backed handle), and otherwise
raises a clear :class:`RuntimeError`.  An explicit ``gpu_device=`` does **not**
fall back: a failed attach is fatal, so misconfiguration surfaces immediately.

Thread safety
-------------

Multiple threads in the same process can open the same GPU stream concurrently.
pyshmem serialises the CUDA IPC handle reconstruction with a per-stream lock,
so all threads that race to open the same stream will share one reconstructed
tensor rather than producing aliased copies.

GPU handle reconstruction happens at most once per (stream, process) pair.
Subsequent opens in the same process reuse the cached
:class:`torch.Tensor` from an internal weakref cache.

How cross-process GPU sharing works
-----------------------------------

GPU streams share tensors across processes using torch's official reduction
path: the producer exports its tensor with
:func:`torch.multiprocessing.reductions.reduce_tensor` and stores the pickled
``(rebuild_fn, args)`` payload in the stream's GPU handle segment; each consumer
reconstructs the tensor with ``rebuild_fn(*args)``.  Using the official path
(rather than calling ``storage._share_cuda_()`` directly) keeps torch's IPC
reference counter correct, which is what lets the producer reclaim GPU memory
once consumers release.

This has consequences for memory lifecycle:

- **Consumers** drop their CUDA tensor on :meth:`~pyshmem.SharedMemory.close`,
  which decrements the producer's IPC reference counter.
- The **owner** keeps its tensor on ``close()`` (so the stream stays mappable
  in-process) and releases it only on :meth:`~pyshmem.SharedMemory.unlink`,
  which also calls :func:`torch.cuda.ipc_collect`.
- If a producer dies without unlinking, it can leave orphaned ``cuda.shm.*``
  reference-count files behind. Ordinary :func:`pyshmem.purge` does not touch
  this process-global PyTorch namespace. Pass ``include_cuda_orphans=True`` (or
  use ``pyshmem purge --include-cuda-orphans``) to explicitly sweep files whose
  producer processes are no longer alive — see :doc:`cli`.

Platform note
-------------

GPU IPC has been tested on Linux.  macOS does not support CUDA.  Windows is not
supported by pyshmem.
