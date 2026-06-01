"""Public package surface for pyshmem."""

from pyshmem._shared import (
    GPU_SUPPORTED_DTYPES,
    SharedMemory,
    create,
    gpu_available,
    list_streams,
    open,
    stream,
    unlink,
)

__version__ = "1.0.2"

__all__ = [
    "GPU_SUPPORTED_DTYPES",
    "SharedMemory",
    "create",
    "gpu_available",
    "list_streams",
    "open",
    "stream",
    "unlink",
    "__version__",
]
