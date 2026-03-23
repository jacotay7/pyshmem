Usage
=====

CPU stream quickstart
---------------------

.. code-block:: python

   import numpy as np
   import pyshare

   writer = pyshare.create("frames", shape=(480, 640), dtype=np.float32)
   reader = pyshare.open("frames")

   writer.write(np.ones((480, 640), dtype=np.float32))
   frame = reader.read()

GPU stream quickstart
---------------------

.. code-block:: python

   import numpy as np
   import pyshare

   writer = pyshare.create(
       "activations",
       shape=(1024, 1024),
       dtype=np.float32,
       gpu_device="cuda:0",
   )
   reader = pyshare.open("activations", gpu_device="cuda:0")

   writer.write(np.ones((1024, 1024), dtype=np.float32))
   activation = reader.read()

Reading modes
-------------

``read(safe=True)`` returns a consistent snapshot of the latest completed write.

``read(safe=False)`` exposes the live backing storage and therefore requires the
caller to hold the stream lock first.

.. code-block:: python

   with reader.locked():
       raw = reader.read(safe=False)

Waiting for the next write
--------------------------

Use ``read_new`` when you want to block until a new payload arrives.

.. code-block:: python

   next_frame = reader.read_new(timeout=1.0)

Lifecycle
---------

- ``close()`` releases only the current handle
- ``unlink()`` destroys the shared-memory stream
- ``delete()`` is an alias for ``unlink()``

Be conservative with ``unlink()`` in multi-process systems: any attached handle
can destroy the stream.