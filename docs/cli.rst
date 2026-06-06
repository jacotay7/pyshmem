Command-Line Interface
======================

Installing pyshmem provides a ``pyshmem`` console command for inspecting and
cleaning up streams without writing any Python.  It is the quickest way to find
and remove leftover segments during development.

.. code-block:: bash

   pyshmem --help

All commands operate on the **user-visible names** passed to
:func:`pyshmem.create` (the same names shown by ``pyshmem list``), not the
internal hashed segment identifiers.

``pyshmem list``
----------------

Print the user-visible name of every existing pyshmem stream, one per line:

.. code-block:: bash

   pyshmem list
   # frames
   # my_gpu_stream

If no streams exist, it prints ``no streams found`` to stderr.  This is the CLI
equivalent of :func:`pyshmem.list_streams`.

.. note::

   Listing relies on scanning ``/dev/shm/`` and is therefore Linux-only.  On
   other platforms it reports no streams.

``pyshmem unlink``
------------------

Destroy one or more streams by name.  This removes the data, metadata, and (for
GPU streams) handle segments, plus the lock file:

.. code-block:: bash

   pyshmem unlink frames
   pyshmem unlink stream_a stream_b   # several at once

Unlinking is the CLI equivalent of :func:`pyshmem.unlink`.  Any other process
that still has the stream open will encounter errors on subsequent operations.

``pyshmem purge``
-----------------

Remove **all** pyshmem segments from the machine in one shot.  In addition to
the ``ps_*`` data/metadata/handle segments and their lock files, ``purge`` also
sweeps orphaned torch CUDA IPC reference-count files (``cuda.shm.*``) left
behind by GPU producers that exited without releasing their tensors:

.. code-block:: bash

   pyshmem purge
   # purged 'frames'
   # purged 'my_gpu_stream'
   # removed 2 stream(s)

``purge`` only deletes ``cuda.shm.*`` files whose producer PID is no longer
alive, so it is safe to run while other GPU streams are in use — it will not
corrupt a live process's CUDA tensors.  It is the CLI equivalent of
:func:`pyshmem.purge`.

.. warning::

   ``purge`` removes every pyshmem stream on the machine for the current user,
   not just your own pipeline's.  Use ``pyshmem unlink`` when you want to remove
   specific streams.

This is the right tool for clearing a development machine that has accumulated
stale streams, or for cleaning up after a crashed run.  It is not reversible.
