GPU Streams
===========

GPU streams require a CUDA-capable PyTorch installation.

Creating a GPU stream
---------------------

.. code-block:: python

   shm = pyshare.create(
       "weights",
       shape=(4096, 4096),
       dtype=np.float32,
       gpu_device="cuda:0",
   )

CPU mirroring modes
-------------------

GPU streams support two intentionally different modes.

Performance mode
^^^^^^^^^^^^^^^^

The default GPU configuration uses ``cpu_mirror=False``.

- fastest path for GPU-heavy workloads
- avoids updating a CPU mirror on every write
- CPU-only handles can still inspect metadata and take locks
- payload reads require a GPU attachment

Compatibility mode
^^^^^^^^^^^^^^^^^^

Set ``cpu_mirror=True`` when you need CPU-side reopen or CPU-side payload reads.

.. code-block:: python

   shm = pyshare.create(
       "weights",
       shape=(4096, 4096),
       dtype=np.float32,
       gpu_device="cuda:0",
       cpu_mirror=True,
   )

This mode trades throughput for compatibility and stronger safe-read semantics
under concurrent writes.

Opening GPU streams
-------------------

Always pass ``gpu_device`` when you want a CUDA tensor view:

.. code-block:: python

   reader = pyshare.open("weights", gpu_device="cuda:0")

If you omit ``gpu_device``, the handle remains CPU-only.