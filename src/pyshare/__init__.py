"""Public package surface for pyshare."""

from pyshare._shared import SharedMemory, create, gpu_available, open, unlink

__version__ = "1.0.0"

__all__ = [
	"SharedMemory",
	"create",
	"open",
	"unlink",
	"gpu_available",
	"__version__",
]
