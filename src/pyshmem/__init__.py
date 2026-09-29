"""Public package surface for pyshmem."""

from importlib.metadata import PackageNotFoundError, version

from pyshmem import _shared
from pyshmem._shared import (
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


def __getattr__(name: str):
    # GPU_SUPPORTED_DTYPES needs torch, which is imported only on first use.
    if name == "GPU_SUPPORTED_DTYPES":
        return _shared.GPU_SUPPORTED_DTYPES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
