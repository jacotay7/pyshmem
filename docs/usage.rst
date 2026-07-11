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

Read-only handles
-----------------

Pass ``readonly=True`` to :func:`~pyshmem.open` for a consumer that must never
mutate the stream.  Reads work normally, but every mutating operation raises
:class:`PermissionError` instead of touching shared state:

.. code-block:: python

   reader = pyshmem.open("my_stream", readonly=True)
   reader.read()                    # ok — snapshots the latest value
   reader.write(value)              # raises PermissionError
   reader.acquire()                 # raises PermissionError
   reader.read(safe=False)          # raises PermissionError (would expose a
                                    # mutable zero-copy view)

The guard rejects :meth:`~pyshmem.SharedMemory.write`,
:meth:`~pyshmem.SharedMemory.write_locked`,
:meth:`~pyshmem.SharedMemory.clear`,
:meth:`~pyshmem.SharedMemory.acquire` (and therefore
:meth:`~pyshmem.SharedMemory.locked`),
:meth:`~pyshmem.SharedMemory.pinned_buffer`, unsafe
(``safe=False``) reads, and handle-level
:meth:`~pyshmem.SharedMemory.unlink`.  It is a per-handle guard, not a
segment-level protection: other writable handles to the same stream continue to
publish normally, and ``readonly`` only reflects how *this* handle was opened
(the owner and any default handles remain writable).

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

``out`` is accepted for safe CPU reads only. GPU and unsafe reads reject it
with :class:`ValueError` rather than silently allocating another result.

Safe reads also accept ``timeout=`` to bound how long they wait for an active
writer. If a payload copy fails, or a writer process exits after beginning a
write, readers raise :class:`pyshmem.InconsistentStreamError` instead of
polling forever. A subsequent successful full write replaces the incomplete
payload and makes the stream readable again.

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

Streams retain only the latest payload. If several writes occur between reads,
intermediate payloads are overwritten rather than queued. After every
successful ``read`` or ``read_new``, inspect ``reader.missed_writes`` for the
number skipped by that read, ``reader.total_missed_writes`` for the handle's
cumulative total, and ``reader.last_read_count`` for the publication captured.
These counters start when the handle attaches; writes before attachment are not
reported as missed.

.. warning::

   ``read_new`` is **edge-triggered relative to the moment you call it**: it
   snapshots the current write count on entry and returns on the first write
   whose count differs.  It answers *"is there anything newer than now?"*, not
   *"has the count reached N?"*.  This is ideal for latest-value streaming
   (grab the freshest frame, skip stale ones), but it is **unsafe for
   synchronous request/response ("ping-pong") exchanges** where the producer
   blocks waiting for your reply.  If the request is published in the small
   window before ``read_new`` snapshots its baseline, that write is folded into
   the baseline and the call waits for the *next* one — which never arrives,
   because the producer is blocked on your response.  Both sides then deadlock
   until their timeouts fire.

   For lock-step exchanges, poll the level instead of the edge: capture
   ``n = shm.count`` before issuing the request, then wait until ``shm.count``
   advances past ``n``.  A level check cannot miss an edge that was already
   published.

   .. code-block:: python

      # Safe request/response consumer (level-triggered):
      expected = request.count + 1
      deadline = time.monotonic() + timeout
      while request.count < expected:
          if time.monotonic() >= deadline:
              raise TimeoutError
          time.sleep(0)
      payload = request.read()

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

Waitable notifications
~~~~~~~~~~~~~~~~~~~~~~~

By default a waiting :meth:`~pyshmem.SharedMemory.read_new` busy-polls the
publication counter.  For low-latency, low-CPU waiting, create the stream with
``notify=True``: writers then wake parked consumers through a Linux futex on the
shared sequence word the instant they publish, so consumers sleep in the kernel
instead of spinning.

.. code-block:: python

   writer = pyshmem.create("frames", shape=(480, 640), notify=True)
   reader = pyshmem.open("frames")     # inherits the stream's notify setting

   frame = reader.read_new(timeout=5.0)   # parks in the kernel, wakes on publish

The setting is a property of the **stream**, so every handle that opens it
participates automatically; ``shm.notify`` reports whether notifications are
active on this handle.  It is opt-in because each write on a notify stream costs
one extra wakeup syscall — default streams are completely unaffected.

Notifications are a **latency/CPU optimization only**; semantics are identical
to a polling ``read_new``.  On platforms without a suitable futex (non-Linux, or
big-endian) the stream still works and simply falls back to polling.  Parked
waits are internally capped so a producer that dies mid-write is still detected
(surfacing :class:`~pyshmem.InconsistentStreamError`) rather than blocking
forever.  ``read_new_async`` also benefits: the kernel wait is offloaded to a
worker thread so the event loop is never blocked.

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
   ``list_streams()`` reports the friendly name. Legacy or unrelated segments
   that do not contain a name which validates against the hash are omitted;
   this prevents discovery and purge from claiming arbitrary ``ps_*`` objects.

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
   readonly:     False
   creator_pid:  10234
   producer:     alive
   age:          0.004 s

Producer liveness and staleness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consumers can tell whether data is fresh and whether the producer is still
running without any producer-side heartbeat thread — every completed write
already stamps the timestamp the checks read.

.. code-block:: python

   shm.age               # seconds since the last completed write
                         #   (math.inf if it was never written)
   shm.is_stale(0.5)     # True if the latest write is older than 0.5 s
   shm.producer_alive()  # is the creating process still running?
   shm.creator_pid       # PID that created the stream

A typical watchdog rejects data that is either stale or orphaned:

.. code-block:: python

   frame = shm.read()
   if shm.is_stale(1.0) or not shm.producer_alive():
       raise RuntimeError("producer stalled or exited")

``producer_alive()`` is a best-effort, single-host POSIX check against the
recorded creator PID: it cannot see producers on other hosts and can be fooled
by PID reuse.  It complements — it does not replace — the seqlock machinery
that detects a writer dying *mid-write* (which surfaces as
:class:`~pyshmem.InconsistentStreamError`); use ``producer_alive()`` to notice a
producer that exited cleanly *between* writes.

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

Framework interop (DLPack)
--------------------------

A :class:`~pyshmem.SharedMemory` handle implements the DLPack protocol
(``__dlpack__`` / ``__dlpack_device__``), so any framework that consumes DLPack
can read a stream directly — no pyshmem-specific code and no framework
lock-in:

.. code-block:: python

   import numpy as np
   import torch

   frame = np.from_dlpack(shm)      # CPU stream  -> numpy.ndarray
   frame = torch.from_dlpack(shm)   # GPU stream  -> torch.Tensor on its device
   # cupy.from_dlpack(shm), jax.dlpack.from_dlpack(shm), ... also work

The export is a **seqlock-consistent snapshot** — exactly what :meth:`read`
returns — not a live view of shared memory.  It is therefore safe on read-only
handles and free of torn reads, and the snapshot's buffer is owned by the
capsule, so it outlives the handle it came from.  ``__dlpack_device__`` reports
``(kDLCPU, 0)`` for CPU streams and the attached CUDA device for GPU streams.

For a genuine zero-copy *live* view of the shared buffer (valid only while you
hold the lock), use ``read(safe=False)`` instead.

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

:meth:`~pyshmem.SharedMemory.unlink` destroys the underlying stream's payload,
metadata, and GPU-handle segments. The small per-name lock file persists so a
later stream with the same name uses the same lifecycle lock:

.. code-block:: python

   shm.unlink()

On POSIX, other processes that already mapped the old generation retain an
isolated mapping until they close it. Their handle cannot unlink a replacement
created under the same name: it raises
:class:`~pyshmem.StaleStreamError`. :func:`pyshmem.unlink` provides an
administrative operation by name and deliberately removes the current
generation:

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
