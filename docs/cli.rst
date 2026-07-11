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

Remove every **validated** pyshmem stream in one shot. A candidate is removed
only when its stored user-visible name hashes back to its exact ``ps_*`` segment
identifier, so unrelated objects that happen to use the prefix are preserved:

.. code-block:: bash

   pyshmem purge
   # purged 'frames'
   # purged 'my_gpu_stream'
   # removed 2 stream(s)

Ordinary purge does not touch PyTorch's process-global ``cuda.shm.*`` namespace.
To also remove files whose encoded producer PID is no longer alive, opt in:

.. code-block:: bash

   pyshmem purge --include-cuda-orphans

This broader cleanup can remove orphaned files created by non-pyshmem PyTorch
applications running under the same OS account. It is the CLI equivalent of
``pyshmem.purge(include_cuda_orphans=True)``.

.. warning::

   ``purge`` removes every pyshmem stream on the machine for the current user,
   not just your own pipeline's.  Use ``pyshmem unlink`` when you want to remove
   specific streams.

This is the right tool for clearing a development machine that has accumulated
stale streams, or for cleaning up after a crashed run.  It is not reversible.
