"""Public package surface for pyshmem."""

from pyshmem._shared import SharedMemory, create, gpu_available, open, unlink

__version__ = "1.0.1"

__all__ = [
    "SharedMemory",
    "create",
    "open",
    "unlink",
    "gpu_available",
    "__version__",
]
