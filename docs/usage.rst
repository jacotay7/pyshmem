Usage
=====

This page walks through every major feature of pyshmem with runnable examples.
If you are new to the library, read the sections in order; each one builds on
the previous.

Creating streams
----------------

Use :func:`pyshmem.create` to allocate a new named stream.  The stream's
shape and dtype are fixed at creation time and visible to any process that
attaches later — the caller does not need to know them in advance.

.. code-block:: python

   import numpy as np
   import pyshmem

   shm = pyshmem.create("my_stream", shape=(480, 640), dtype=np.float32)

Required parameters:

- **name** — a string that uniquely identifies the stream across all processes on
  the machine.
- **shape** — a sequence of positive integers giving the array dimensions.
- **dtype** — any NumPy dtype.  For GPU streams, only dtypes in
  :data:`pyshmem.GPU_SUPPORTED_DTYPES` are accepted.

Optional parameters:

- **gpu_device** — a CUDA device string such as ``"cuda:0"`` to create a
  GPU-backed stream.  Requires PyTorch with CUDA support.
- **cpu_mirror** — controls whether a CPU copy is kept in sync for GPU streams.
  Defaults to ``False`` for GPU streams (fastest path) and ``True`` for
  CPU-only streams.
- **auto_unlink** — when ``True``, the stream is destroyed (not just closed)
  when used as a context manager.

If a stream with the same name already exists, :func:`~pyshmem.create` raises
:class:`FileExistsError`.

Temporary streams with automatic cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`pyshmem.stream` is a context manager that creates a stream and
unconditionally unlinks it when the block exits, even if an exception is raised.
It is equivalent to :func:`~pyshmem.create` with ``auto_unlink=True``.

.. code-block:: python

   with pyshmem.stream("scratch", shape=(256,), dtype=np.float32) as shm:
       shm.write(np.zeros(256, dtype=np.float32))
       result = shm.read()
   # stream is destroyed here

The same effect is achievable via ``create`` if you want to decide at call time:

.. code-block:: python

   shm = pyshmem.create("scratch", shape=(256,), auto_unlink=True)
   with shm:
       shm.write(data)
   # auto-unlinked on exit

Attaching to streams
---------------------

Use :func:`pyshmem.open` in any process to attach to an existing stream.  The
shape and dtype are read from the stream's metadata — no arguments needed beyond
the name.

.. code-block:: python

   reader = pyshmem.open("my_stream")
   print(reader.shape)   # (480, 640)
   print(reader.dtype)   # float32

For GPU-backed streams, :func:`~pyshmem.open` reconstructs the stream exactly
as it was created.  By default it **auto-attaches to the CUDA device recorded
in the stream's metadata** — you do not need to pass ``gpu_device``:

.. code-block:: python

   reader = pyshmem.open("my_gpu_stream")          # attaches to its cuda:N
   tensor = reader.read()                          # torch.Tensor on that device

You may still name the device explicitly; it is validated against the stored
device and must match:

.. code-block:: python

   reader = pyshmem.open("my_gpu_stream", gpu_device="cuda:0")

To read a GPU stream's **CPU mirror** without attaching a CUDA tensor — even on
a host that *has* CUDA, where the default would otherwise attach to the GPU —
pass ``gpu_device=False``.  This requires the stream to have been created with
``cpu_mirror=True`` and makes :meth:`~pyshmem.SharedMemory.read` return a
:class:`numpy.ndarray`:

.. code-block:: python

   reader = pyshmem.open("my_gpu_stream", gpu_device=False)
   frame = reader.read()                           # numpy.ndarray from the mirror

If CUDA cannot be attached at all (no CUDA in this process, or the producer has
exited), :func:`~pyshmem.open` falls back to the CPU mirror automatically when
one exists, and otherwise raises a clear error.  See :doc:`gpu` for the full
GPU model.

Writing data
------------

.. code-block:: python

   import numpy as np

   writer.write(np.ones((480, 640), dtype=np.float32))

:meth:`~pyshmem.SharedMemory.write` acquires the cross-process lock
internally, copies the payload, then releases the lock.  For CPU streams the
value is passed through :func:`numpy.asarray`; for GPU streams it is moved to
the configured device via :func:`torch.as_tensor`.

The payload must match the stream's shape exactly.  A :class:`ValueError` is
raised otherwise.

:meth:`~pyshmem.SharedMemory.clear` zeros the payload and records a write:

.. code-block:: python

   writer.clear()

Reading data
------------

Basic read
~~~~~~~~~~

.. code-block:: python

   frame = reader.read()

By default, :meth:`~pyshmem.SharedMemory.read` returns a consistent snapshot
of the latest completed write (``safe=True``).  Internally it polls the
write-sequence counter until it is even (no write in progress), copies the
array, and verifies the sequence did not change mid-copy.

CPU streams return a :class:`numpy.ndarray`; GPU streams return a
:class:`torch.Tensor` on the configured device.

Zero-allocation reads
~~~~~~~~~~~~~~~~~~~~~

Pass a pre-allocated buffer to avoid a heap allocation on every call.  This is
useful in tight real-time loops:

.. code-block:: python

   buf = np.empty(shm.shape, dtype=shm.dtype)

   while True:
       shm.read(out=buf)   # writes into buf; no new array is allocated
       process(buf)

``out`` is accepted for CPU streams only.  It is silently ignored for GPU
streams.

Waiting for a new write
~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~pyshmem.SharedMemory.read_new` blocks until the stream's write count
advances, then returns the payload:

.. code-block:: python

   next_frame = reader.read_new(timeout=1.0)

If no new write arrives within ``timeout`` seconds, a :class:`TimeoutError` is
raised.  Omit ``timeout`` (or set it to ``None``) to block indefinitely.

``read_new`` correctly skips polling while a write is in progress (odd
``write_sequence``), so it never returns a partial write.

Asyncio-compatible waiting
~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~pyshmem.SharedMemory.read_new_async` is the asyncio-safe counterpart.
It uses :func:`asyncio.sleep` instead of :func:`time.sleep`, so the event loop
is not blocked while waiting:

.. code-block:: python

   import asyncio
   import pyshmem

   async def consumer():
       shm = pyshmem.open("my_stream")
       while True:
           frame = await shm.read_new_async(timeout=5.0)
           await process(frame)

   asyncio.run(consumer())

Locking
-------

:meth:`~pyshmem.SharedMemory.write` acquires the lock automatically.  For
more advanced scenarios — reading without a copy, or writing multiple streams
atomically — take the lock explicitly.

The lock is cross-process (backed by a ``portalocker`` file lock) and
re-entrant within the current thread.

Context manager (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   with shm.locked():
       raw = shm.read(safe=False)          # zero-copy view into backing storage
       transformed = raw * 2.0
       shm.write_locked(transformed)       # write back without re-locking

Explicit acquire / release
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   shm.acquire(timeout=0.5)
   try:
       data = shm.read(safe=False)
       shm.write_locked(data + 1)
   finally:
       shm.release()

``safe=False`` reads
~~~~~~~~~~~~~~~~~~~~

:meth:`~pyshmem.SharedMemory.read` with ``safe=False`` returns a direct view
into the backing storage without copying.  This is only valid while the lock is
held:

.. code-block:: python

   with shm.locked():
       view = shm.read(safe=False)
       # view is valid here — do not use it after the block exits

Writing without re-acquiring the lock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~pyshmem.SharedMemory.write_locked` is identical to
:meth:`~pyshmem.SharedMemory.write` but skips the internal lock acquisition.
The caller must already own the lock.  This is the intended pattern for
high-performance consumers that hold the lock for the duration of a pipeline
step:

.. code-block:: python

   with output_stream.locked():
       # perform computation, then write the result in one lock ownership
       output_stream.write_locked(compute(input_data))

Discovering streams
-------------------

:func:`pyshmem.list_streams` returns the sorted **user-visible names** of all
existing pyshmem streams found in ``/dev/shm/``.  It is available on Linux and
returns an empty list on other platforms.

.. code-block:: python

   pyshmem.list_streams()
   # ['my_gpu_stream', 'my_stream']

.. note::

   Segment names are stored internally as SHA-1 hashes (``ps_<hash>``) to stay
   under the POSIX name-length limit, but the original name passed to
   :func:`~pyshmem.create` is recorded in each stream's metadata, so
   ``list_streams()`` reports the friendly name.  Streams created by very old
   pyshmem versions that did not record the name fall back to their hashed
   ``ps_<hash>`` identifier.

The same listing is available from the shell with ``pyshmem list`` — see
:doc:`cli`.

Inspecting streams
------------------

Human-readable summary
~~~~~~~~~~~~~~~~~~~~~~

:meth:`~pyshmem.SharedMemory.describe` returns a formatted multi-line string
with all metadata fields:

.. code-block:: python

   print(shm.describe())

.. code-block:: text

   name:         my_stream
   shape:        (480, 640)
   dtype:        float32
   size:         1228800 bytes
   gpu_enabled:  False
   gpu_device:   None
   cpu_mirror:   True
   count:        42
   write_time:   1748725312.4
   write_seq:    84
   owner:        False

Config export and import
~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~pyshmem.SharedMemory.to_config` returns a plain dictionary that
captures the stream's configuration:

.. code-block:: python

   cfg = shm.to_config()
   # {'name': 'my_stream', 'shape': [480, 640], 'dtype': 'float32',
   #  'gpu_device': None, 'cpu_mirror': True}

:meth:`~pyshmem.SharedMemory.create_from_config` recreates an
identically-configured stream from such a dict:

.. code-block:: python

   shm2 = pyshmem.SharedMemory.create_from_config(cfg)

This is useful for serialising pipeline configurations to YAML or JSON and
reconstructing the streams from them at startup.

Lifecycle
---------

``close()``
~~~~~~~~~~~

:meth:`~pyshmem.SharedMemory.close` releases the local handle (unmaps the
shared-memory segments in the current process).  The underlying stream persists
and can be reattached by other processes.

.. code-block:: python

   shm.close()

If the current thread holds the lock when ``close()`` is called, the lock is
released first.

``unlink()``
~~~~~~~~~~~~

:meth:`~pyshmem.SharedMemory.unlink` destroys the underlying stream
entirely — all three POSIX segments (data, metadata, GPU handle) and the lock
file are removed:

.. code-block:: python

   shm.unlink()

Any other process that still has the stream open will encounter errors on
subsequent operations.  :func:`pyshmem.unlink` provides the same operation by
name, without needing a handle:

.. code-block:: python

   pyshmem.unlink("my_stream")

``delete()``
~~~~~~~~~~~~

:meth:`~pyshmem.SharedMemory.delete` is an alias for
:meth:`~pyshmem.SharedMemory.unlink`.

After ``close()`` or ``unlink()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods that access the stream after the handle is closed — ``read``, ``write``,
``acquire``, ``describe``, metadata properties — raise a :class:`RuntimeError`
with a message that includes the stream name and suggests reopening:

.. code-block:: python

   shm.close()
   shm.read()
   # RuntimeError: cannot read from closed shared memory 'my_stream';
   #               reopen it with pyshmem.open('my_stream')

Environment variables
---------------------

``PYSHMEM_LOCK_DIR``
~~~~~~~~~~~~~~~~~~~~

Lock files are stored in ``/tmp/pyshmem-locks-<uid>/`` by default (where
``<uid>`` is the current user's POSIX UID).  On shared servers where multiple
users run pyshmem, the UID prefix prevents one user's lock files from
interfering with another's.

Set ``PYSHMEM_LOCK_DIR`` to use a different directory:

.. code-block:: bash

   export PYSHMEM_LOCK_DIR=/run/my_pipeline/locks

This is useful in containerised environments where ``/tmp`` is shared but you
want full isolation between pipeline instances.
