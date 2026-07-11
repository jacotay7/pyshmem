Platform Notes
==============

Linux
-----

Full support.  Streams persist in ``/dev/shm/`` as POSIX shared-memory
segments.  A segment survives the creator process exiting as long as at least
one other process still has it open (or until :func:`pyshmem.unlink` is called
by any process).

GPU IPC (via torch's ``reduce_tensor`` reduction) is supported on Linux.

CPU architecture
----------------

The lock-free read/write consistency protocol assumes single-copy atomicity of
naturally aligned 64-bit accesses. It is validated under CPython on **x86-64**
and **aarch64**. Other architectures are best-effort: pyshmem inserts no
explicit hardware memory barriers, so torn-read freedom is not guaranteed there.
See :doc:`format` for the full memory model.

macOS
-----

POSIX shared memory is supported; stream persistence works the same as on
Linux.  GPU streams are not supported (no CUDA on macOS).

:func:`pyshmem.list_streams` is not implemented on macOS because macOS does
not use ``/dev/shm/``.

Windows
-------

Windows inherits a hard limitation from :mod:`multiprocessing.shared_memory`:
the operating system destroys a shared-memory block when the **last handle** to
it is closed.

Consequences:

- A segment cannot outlive its creator if no other process still has it open.
- ``close()`` followed by ``pyshmem.open(...)`` fails if that ``close()`` dropped
  the final live handle.
- :func:`pyshmem.list_streams` always returns an empty list on Windows.
- GPU streams are not tested on Windows.

These are operating-system behaviors, not pyshmem policies.

Lock files
----------

pyshmem uses ``portalocker`` file locks for cross-process write serialisation.
Lock files are stored in a per-user directory:

.. code-block:: text

   /tmp/pyshmem-locks-<uid>/

where ``<uid>`` is the current user's POSIX UID.  The per-user directory
prevents lock files from interfering across users on shared servers.

``PYSHMEM_LOCK_DIR`` environment variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``PYSHMEM_LOCK_DIR`` to override the lock directory:

.. code-block:: bash

   export PYSHMEM_LOCK_DIR=/run/my_pipeline/locks

This is useful in containerised environments where ``/tmp`` is shared but full
isolation between pipeline instances is required.  All processes that share
streams must agree on the lock directory — if one process writes to
``/tmp/pyshmem-locks-<uid>/`` and another writes to a different directory, they
will not serialise correctly.

Lock files are small and survive process exits (they are cleaned up by
:func:`pyshmem.unlink`).  ``portalocker`` uses OS-level file locks that are
released automatically when a process crashes, so stale locks do not block
subsequent writers.

Unlink/recreate with live handles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`pyshmem.unlink` removes the lock file, but a process that still holds a
handle keeps the original inode open.  If the stream is then recreated, a new
inode appears at the same pathname.  To keep every process serialising on the
same lock, pyshmem records the lock file's inode and, on each lock acquisition,
rebinds a handle whose pathname now resolves to a different inode.  A recreated
stream therefore converges on one shared lock object rather than splitting into
independent locks per handle generation.

Fork safety
~~~~~~~~~~~

``os.fork()`` duplicates a process's lock state, including the lock file
descriptor and any "held" flag.  pyshmem registers an ``os.register_at_fork``
child handler that, in the child, gives each inherited lock state a fresh
re-entrant lock, clears the held flag, and reopens the lock file to a private
descriptor.  A forked child therefore never believes it inherited the parent's
held lock and cannot release the parent's cross-process lock through a shared
descriptor.  (CUDA state does not survive fork; the child also drops any cached
IPC tensors.)  Forking a multi-threaded process remains generally hazardous —
prefer the ``spawn`` start method — but pyshmem's own lock state is left
consistent.
