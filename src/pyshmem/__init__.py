"""Public package surface for pyshmem."""

from importlib.metadata import PackageNotFoundError, version

from pyshmem._shared import (
    GPU_SUPPORTED_DTYPES,
    InconsistentStreamError,
    Publication,
    StaleStreamError,
    SharedMemory,
    create,
    gpu_available,
    list_streams,
    locked_many,
    open,
    purge,
    stat,
    stream,
    unlink,
    unlink_quiet,
)

try:
    __version__ = version("pyshmem")
except PackageNotFoundError:  # source tree imported without installation
    __version__ = "0+unknown"

__all__ = [
    "GPU_SUPPORTED_DTYPES",
    "InconsistentStreamError",
    "Publication",
    "StaleStreamError",
    "SharedMemory",
    "create",
    "gpu_available",
    "list_streams",
    "locked_many",
    "open",
    "purge",
    "stat",
    "stream",
    "unlink",
    "unlink_quiet",
    "__version__",
]
