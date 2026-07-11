API Reference
=============

This page documents the complete public surface of the ``pyshmem`` package.

Top-level functions and constants
----------------------------------

.. autofunction:: pyshmem.create

.. autofunction:: pyshmem.open

.. autofunction:: pyshmem.unlink

.. autofunction:: pyshmem.unlink_quiet

.. autoexception:: pyshmem.StaleStreamError

.. autofunction:: pyshmem.purge

.. autofunction:: pyshmem.stream

.. autofunction:: pyshmem.list_streams

.. autofunction:: pyshmem.gpu_available

.. autoexception:: pyshmem.InconsistentStreamError

.. autodata:: pyshmem.GPU_SUPPORTED_DTYPES

   A :class:`frozenset` of :class:`numpy.dtype` objects that can be used with
   the ``gpu_device=`` parameter of :func:`create`.

   .. code-block:: python

      pyshmem.GPU_SUPPORTED_DTYPES
      # contents reflect the installed torch's dtype capabilities

   The set is derived from dtype attributes exposed by the installed PyTorch.
   Passing an unsupported dtype to :func:`create` with ``gpu_device`` set raises
   :class:`ValueError` at construction time.

SharedMemory
------------

.. autoclass:: pyshmem.SharedMemory
   :members:
   :member-order: bysource
