Quick Start
===========

This page gets you from ``pip install`` to data flowing between two processes in
a few minutes.  For the full feature tour read :doc:`usage`; for the GPU model
read :doc:`gpu`.

Install
-------

.. code-block:: bash

   pip install pyshmem          # CPU streams (NumPy)
   pip install pyshmem[gpu]     # adds CUDA-backed GPU streams (PyTorch)

Single process round trip
--------------------------

Create a named stream, write an array, read it back:

.. code-block:: python

   import numpy as np
   import pyshmem

   shm = pyshmem.create("hello", shape=(4,), dtype=np.float32)
   shm.write(np.array([1, 2, 3, 4], dtype=np.float32))
   print(shm.read())            # [1. 2. 3. 4.]
   shm.unlink()                 # destroy the stream when done

The stream's shape and dtype are fixed at creation and stored in shared memory,
so any other process can attach to it by name alone.

Two processes: producer and consumer
-------------------------------------

The point of pyshmem is sharing data *between* processes.  The producer creates
the stream and writes to it; the consumer attaches by name and blocks for new
writes.  Run these as two separate programs (in either order — the consumer
retries until the stream exists).

**producer.py**

.. code-block:: python

   import time
   import numpy as np
   import pyshmem

   shm = pyshmem.create("frames", shape=(480, 640), dtype=np.float32)
   try:
       i = 0
       while True:
           shm.write(np.full((480, 640), i, dtype=np.float32))
           i += 1
           time.sleep(0.1)
   finally:
       shm.unlink()             # clean up shared memory on exit

**consumer.py**

.. code-block:: python

   import pyshmem

   shm = pyshmem.open("frames")          # attach by name; shape/dtype recovered
   try:
       while True:
           frame = shm.read_new(timeout=5.0)   # blocks until the next write
           print("got frame", frame[0, 0])
   finally:
       shm.close()              # detach this handle; stream itself persists

Key points:

- :func:`pyshmem.create` allocates the stream; :func:`pyshmem.open` attaches to
  an existing one without needing to know its shape or dtype.
- :meth:`~pyshmem.SharedMemory.read_new` blocks until a *new* write arrives and
  never returns a half-written payload.
- :meth:`~pyshmem.SharedMemory.close` detaches one handle; the stream lives on
  in ``/dev/shm`` until someone calls :meth:`~pyshmem.SharedMemory.unlink` (or
  ``pyshmem unlink`` / ``pyshmem purge`` on the CLI).

GPU in one step
---------------

Pass ``gpu_device`` to back the stream with a CUDA tensor.  Reads and writes use
:class:`torch.Tensor`; a consumer on the same machine attaches with no extra
arguments:

.. code-block:: python

   import numpy as np
   import pyshmem

   shm = pyshmem.create(
       "activations", shape=(4096,), dtype=np.float32, gpu_device="cuda:0"
   )

   reader = pyshmem.open("activations")  # auto-attaches to cuda:0
   tensor = reader.read()                # torch.Tensor on cuda:0

Add ``cpu_mirror=True`` if you also need CPU-side consumers (viewers, loggers);
they read the host mirror with ``pyshmem.open(name, gpu_device=False)``.  See
:doc:`gpu` for the trade-offs.

Cleaning up
-----------

Streams persist after a process exits (that is the point), so leftover segments
can accumulate during development.  List and remove them from the shell:

.. code-block:: bash

   pyshmem list                 # show user-visible stream names
   pyshmem unlink frames        # remove one stream
   pyshmem purge                # remove ALL pyshmem segments

See :doc:`cli` for the full command-line reference.

Where to go next
----------------

- :doc:`installation` — extras, development install, verifying the build
- :doc:`usage` — every method, with locking, zero-copy reads, and async waits
- :doc:`gpu` — GPU streams, CPU mirroring modes, cross-process sharing
- :doc:`api` — the complete API reference
