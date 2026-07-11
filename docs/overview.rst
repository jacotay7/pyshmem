Overview
========

pyshmem gives you a single interface for two related use cases:

- **CPU shared-memory streams** backed by NumPy arrays
- **GPU shared-memory streams** backed by CUDA tensors through PyTorch

The design goal is to move structured numeric payloads between OS processes
without forcing every application to reinvent locking, metadata storage, or
CPU/GPU branching logic.

Core concepts
-------------

A **stream** is the only primitive pyshmem exposes. It is a capacity-one,
latest-value exchange: each write replaces the prior payload; it is not a
queue and does not apply backpressure. It is a named slot in shared memory with
a fixed shape, dtype, and storage backend. Streams persist
across process exits on POSIX systems and can be attached by any process that
knows the name.

Every successful write increments ``count``. After a read, the handle's
``last_read_count``, ``missed_writes``, and ``total_missed_writes`` properties
make skipped intermediate publications observable. Applications that must
consume every item need a queue or ring buffer instead.

Each stream is backed by up to three POSIX shared-memory segments:

- a **data segment** holding the array payload
- a **metadata segment** storing shape, dtype, write count, lock state, and
  other bookkeeping
- a **GPU handle segment** (GPU streams only) holding the serialised CUDA IPC
  handle

Names are hashed with SHA-1 to stay under the POSIX segment name length limit.

Write consistency
-----------------

Writers bracket payloads with an odd/even write-sequence counter.  Readers poll
until the sequence is even (stable), copy the payload, then verify the sequence
did not change mid-copy.  This provides consistent snapshots without requiring
readers to hold the write lock.

For callers that need the raw backing storage without a copy, the lock must be
held explicitly — see :doc:`usage`.

Public API
----------

The public package surface is intentionally small:

- :func:`pyshmem.create` — create a new named stream
- :func:`pyshmem.open` — attach to an existing stream
- :func:`pyshmem.unlink` — destroy a stream by name
- :func:`pyshmem.unlink_quiet` — destroy a stream, ignoring "does not exist"
- :func:`pyshmem.purge` — remove all validated pyshmem segments; global orphaned
  PyTorch CUDA IPC cleanup requires an explicit option
- :func:`pyshmem.stream` — context manager that creates and auto-unlinks
- :func:`pyshmem.list_streams` — list the user-visible names of existing
  streams (Linux)
- :func:`pyshmem.gpu_available` — check whether CUDA streams are usable
- :data:`pyshmem.GPU_SUPPORTED_DTYPES` — dtypes accepted for GPU streams
- :class:`pyshmem.SharedMemory` — the stream handle object

The same lifecycle operations are available from the command line — see
:doc:`cli`.

See :doc:`api` for the full reference.  The best path for new users is to read
:doc:`quickstart`, then :doc:`installation` and :doc:`usage`.
