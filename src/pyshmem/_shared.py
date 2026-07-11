"""Shared-memory primitives for CPU NumPy arrays and optional CUDA tensors.

The public API exposed through :mod:`pyshmem` is intentionally small:

- :func:`create` creates a named shared-memory stream
- :func:`open` attaches to an existing stream
- :func:`unlink` destroys a stream by name
- :func:`gpu_available` reports whether CUDA-backed streams are available

The :class:`SharedMemory` object presents one interface for CPU-only
streams and GPU-backed streams, with GPU CPU-mirroring controlled
explicitly through the ``cpu_mirror`` argument passed to :func:`create`.
"""

from __future__ import annotations

import asyncio
import hashlib
import builtins
from contextlib import contextmanager
import glob
import math
import os
import pickle
import tempfile
import threading
import time
import weakref
from multiprocessing import resource_tracker, shared_memory
from typing import Any, Sequence

import numpy as np
import portalocker

try:
    import torch
except Exception:
    torch = None


DTYPE_TABLE = (
    np.dtype(np.int8),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.int64),
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.uint32),
    np.dtype(np.uint64),
    np.dtype(np.float16),
    np.dtype(np.float32),
    np.dtype(np.float64),
)
DTYPE_TO_CODE = {dtype: index for index, dtype in enumerate(DTYPE_TABLE)}
if torch is not None:
    TORCH_DTYPE_MAP = {
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.int16): torch.int16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.uint8): torch.uint8,
    }
else:
    TORCH_DTYPE_MAP = {}

GPU_SUPPORTED_DTYPES: frozenset = frozenset(
    {
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.int32),
        np.dtype(np.int64),
        np.dtype(np.uint8),
    }
)

METADATA_VERSION = 3
LEGACY_METADATA_VERSION = 2
METADATA_INDEX_VERSION = 0
METADATA_INDEX_COUNT = 1
METADATA_INDEX_DTYPE = 2
METADATA_INDEX_NDIM = 3
METADATA_INDEX_SIZE = 4
METADATA_INDEX_GPU_ENABLED = 5
METADATA_INDEX_DEVICE_INDEX = 6
METADATA_INDEX_CREATOR_PID = 7
METADATA_INDEX_WRITE_TIME = 8
METADATA_INDEX_WRITE_SEQUENCE = 9
METADATA_INDEX_LOCK_OWNER_PID = 10
METADATA_INDEX_LOCK_DEPTH = 11
METADATA_INDEX_CPU_MIRROR_ENABLED = 12
METADATA_INDEX_SHAPE_START = 13
METADATA_SIZE = 32
METADATA_BYTES = METADATA_SIZE * np.dtype(np.float64).itemsize
# Version 3 replaces the float64 metadata block with an explicitly laid-out,
# little-endian 256-byte header. Frequently updated counters are fixed-width
# integers and naturally 8-byte aligned. Atomic ordering is a separate concern:
# NumPy access to these fields is not claimed to provide interprocess atomics.
METADATA_MAGIC = b"PYSHMEM\x00"
METADATA_FLAG_GPU_ENABLED = 1 << 0
METADATA_FLAG_CPU_MIRROR_ENABLED = 1 << 1
METADATA_V3_DTYPE = np.dtype(
    {
        "names": [
            "magic",
            "version",
            "header_size",
            "flags",
            "dtype_code",
            "ndim",
            "device_index",
            "creator_pid",
            "size",
            "count",
            "write_sequence",
            "write_time",
            "lock_owner_pid",
            "lock_depth",
            "reserved",
            "shape",
        ],
        "formats": [
            "S8",
            "<u2",
            "<u2",
            "<u4",
            "<u2",
            "<u2",
            "<i4",
            "<i8",
            "<u8",
            "<u8",
            "<i8",
            "<f8",
            "<i8",
            "<u4",
            "V28",
            ("<u8", METADATA_SIZE - METADATA_INDEX_SHAPE_START),
        ],
        "offsets": [
            0,
            8,
            10,
            12,
            16,
            18,
            20,
            24,
            32,
            40,
            48,
            56,
            64,
            72,
            76,
            104,
        ],
        "itemsize": METADATA_BYTES,
    }
)
# A fixed byte region appended after the float64 metadata block stores the
# original, user-visible stream name (UTF-8, null-padded).  Segment ids are a
# one-way SHA-1 hash of the name, so this is the only way list() and the CLI
# can report the friendly name instead of the hashed ``ps_*`` id.
METADATA_NAME_MAX = 256
METADATA_TOTAL_BYTES = METADATA_BYTES + METADATA_NAME_MAX

_THREAD_LOCK_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, "_SharedLockState"] = {}
_LOCAL_GPU_TENSORS: dict[str, weakref.ReferenceType[Any]] = {}
# Per-name locks for serialising GPU handle reconstruction across threads.
_GPU_OPEN_LOCKS_GUARD = threading.Lock()
_GPU_OPEN_LOCKS: dict[str, threading.Lock] = {}


class _SharedLockState:
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.thread_lock = threading.RLock()
        self.file_handle = builtins.open(path, "a+b")
        self.owner_thread_id: int | None = None
        self.depth = 0
        self.reference_count = 0


class InconsistentStreamError(RuntimeError):
    """Raised when a writer failed before publishing a complete payload."""


class _MetadataView:
    """Index-compatible view over v2 and v3 metadata layouts."""

    def __init__(self, buffer, *, initialize: bool = False) -> None:
        if len(buffer) < METADATA_BYTES:
            raise ValueError(
                f"metadata segment is too small: expected at least "
                f"{METADATA_BYTES} bytes, got {len(buffer)}"
            )
        if initialize:
            buffer[:METADATA_BYTES] = b"\x00" * METADATA_BYTES
            self.layout_version = METADATA_VERSION
            self._v2 = None
            self._v3 = np.ndarray((), dtype=METADATA_V3_DTYPE, buffer=buffer)
            self._v3["magic"] = METADATA_MAGIC
            self._v3["version"] = METADATA_VERSION
            self._v3["header_size"] = METADATA_BYTES
            return

        if bytes(buffer[: len(METADATA_MAGIC)]) == METADATA_MAGIC:
            self._v2 = None
            self._v3 = np.ndarray((), dtype=METADATA_V3_DTYPE, buffer=buffer)
            self.layout_version = int(self._v3["version"])
            if self.layout_version != METADATA_VERSION:
                raise ValueError(
                    f"unsupported pyshmem metadata version: "
                    f"{self.layout_version}"
                )
            if int(self._v3["header_size"]) != METADATA_BYTES:
                raise ValueError("invalid pyshmem metadata header size")
            return

        self._v3 = None
        self._v2 = np.ndarray(
            (METADATA_SIZE,), dtype=np.float64, buffer=buffer
        )
        self.layout_version = int(self._v2[METADATA_INDEX_VERSION])
        if self.layout_version != LEGACY_METADATA_VERSION:
            raise ValueError(
                f"unsupported pyshmem metadata version: {self.layout_version}"
            )

    def _flags(self) -> int:
        return int(self._v3["flags"])

    def __getitem__(self, index: int):
        if self._v2 is not None:
            return self._v2[index]
        fields = {
            METADATA_INDEX_VERSION: "version",
            METADATA_INDEX_COUNT: "count",
            METADATA_INDEX_DTYPE: "dtype_code",
            METADATA_INDEX_NDIM: "ndim",
            METADATA_INDEX_SIZE: "size",
            METADATA_INDEX_DEVICE_INDEX: "device_index",
            METADATA_INDEX_CREATOR_PID: "creator_pid",
            METADATA_INDEX_WRITE_TIME: "write_time",
            METADATA_INDEX_WRITE_SEQUENCE: "write_sequence",
            METADATA_INDEX_LOCK_OWNER_PID: "lock_owner_pid",
            METADATA_INDEX_LOCK_DEPTH: "lock_depth",
        }
        if index == METADATA_INDEX_GPU_ENABLED:
            return bool(self._flags() & METADATA_FLAG_GPU_ENABLED)
        if index == METADATA_INDEX_CPU_MIRROR_ENABLED:
            return bool(self._flags() & METADATA_FLAG_CPU_MIRROR_ENABLED)
        if index >= METADATA_INDEX_SHAPE_START:
            return self._v3["shape"][index - METADATA_INDEX_SHAPE_START]
        try:
            return self._v3[fields[index]]
        except KeyError as exc:
            raise IndexError(index) from exc

    def __setitem__(self, index: int, value) -> None:
        if self._v2 is not None:
            self._v2[index] = value
            return
        fields = {
            METADATA_INDEX_VERSION: "version",
            METADATA_INDEX_COUNT: "count",
            METADATA_INDEX_DTYPE: "dtype_code",
            METADATA_INDEX_NDIM: "ndim",
            METADATA_INDEX_SIZE: "size",
            METADATA_INDEX_DEVICE_INDEX: "device_index",
            METADATA_INDEX_CREATOR_PID: "creator_pid",
            METADATA_INDEX_WRITE_TIME: "write_time",
            METADATA_INDEX_WRITE_SEQUENCE: "write_sequence",
            METADATA_INDEX_LOCK_OWNER_PID: "lock_owner_pid",
            METADATA_INDEX_LOCK_DEPTH: "lock_depth",
        }
        if index in (
            METADATA_INDEX_GPU_ENABLED,
            METADATA_INDEX_CPU_MIRROR_ENABLED,
        ):
            bit = (
                METADATA_FLAG_GPU_ENABLED
                if index == METADATA_INDEX_GPU_ENABLED
                else METADATA_FLAG_CPU_MIRROR_ENABLED
            )
            flags = self._flags()
            self._v3["flags"] = flags | bit if bool(value) else flags & ~bit
            return
        if index >= METADATA_INDEX_SHAPE_START:
            self._v3["shape"][index - METADATA_INDEX_SHAPE_START] = value
            return
        try:
            self._v3[fields[index]] = value
        except KeyError as exc:
            raise IndexError(index) from exc


def gpu_available() -> bool:
    """Return ``True`` when CUDA-backed PyTorch streams are available."""
    return bool(torch is not None and torch.cuda.is_available())


def _segment_base_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:14]
    return f"ps_{digest}"


def _data_name(name: str) -> str:
    return _segment_base_name(name)


def _metadata_name(name: str) -> str:
    return f"{_segment_base_name(name)}_meta"


def _gpu_handle_name(name: str) -> str:
    return f"{_segment_base_name(name)}_gpu"


def _encode_stream_name(name: str) -> bytes:
    encoded = name.encode("utf-8")
    if len(encoded) > METADATA_NAME_MAX:
        raise ValueError(
            "stream name is too long to store in metadata "
            f"(max {METADATA_NAME_MAX} UTF-8 bytes, got {len(encoded)})"
        )
    return encoded


def _write_stream_name(metadata_shm: shared_memory.SharedMemory, name: str):
    """Store the user-visible name in the name region of a metadata segment."""
    buf = metadata_shm.buf
    if len(buf) < METADATA_TOTAL_BYTES:
        return
    encoded = _encode_stream_name(name)
    buf[METADATA_BYTES:METADATA_TOTAL_BYTES] = b"\x00" * METADATA_NAME_MAX
    buf[METADATA_BYTES : METADATA_BYTES + len(encoded)] = encoded


def _read_stream_name(
    metadata_shm: shared_memory.SharedMemory,
) -> str | None:
    """Recover the user-visible name from a metadata segment, if present.

    Returns ``None`` for segments written by older pyshmem versions (which did
    not reserve the name region) or for any unreadable bytes.
    """
    buf = metadata_shm.buf
    if len(buf) < METADATA_TOTAL_BYTES:
        return None
    raw = bytes(buf[METADATA_BYTES:METADATA_TOTAL_BYTES]).split(b"\x00", 1)[0]
    if not raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _lock_path(name: str) -> str:
    env_dir = os.environ.get("PYSHMEM_LOCK_DIR")
    if env_dir:
        directory = env_dir
    else:
        uid = getattr(os, "getuid", lambda: 0)()
        directory = os.path.join(tempfile.gettempdir(), f"pyshmem-locks-{uid}")
    return os.path.join(directory, f"{_segment_base_name(name)}.lock")


def _lock_state(name: str) -> _SharedLockState:
    path = _lock_path(name)
    with _THREAD_LOCK_GUARD:
        state = _THREAD_LOCKS.get(path)
        if state is None:
            state = _SharedLockState(path)
            _THREAD_LOCKS[path] = state
        state.reference_count += 1
    return state


def _release_lock_state(state: _SharedLockState) -> None:
    """Drop one handle reference and close unused per-name lock state."""
    with _THREAD_LOCK_GUARD:
        state.reference_count -= 1
        if state.reference_count > 0:
            return
        if state.reference_count < 0:
            raise RuntimeError("shared lock state reference count underflow")
        if state.owner_thread_id is not None:
            raise RuntimeError(
                "cannot discard a shared lock while it is owned"
            )
        current = _THREAD_LOCKS.get(state.path)
        if current is state:
            _THREAD_LOCKS.pop(state.path, None)
        state.file_handle.close()


def _cache_gpu_tensor(name: str, gpu_tensor: Any) -> None:
    _LOCAL_GPU_TENSORS[name] = weakref.ref(gpu_tensor)


def _get_cached_gpu_tensor(name: str) -> Any | None:
    reference = _LOCAL_GPU_TENSORS.get(name)
    if reference is None:
        return None
    gpu_tensor = reference()
    if gpu_tensor is None:
        _LOCAL_GPU_TENSORS.pop(name, None)
    return gpu_tensor


def _acquire_file_lock(
    file_handle, *, timeout: float | None, poll_interval: float
) -> None:
    if timeout is None:
        portalocker.lock(file_handle, portalocker.LOCK_EX)
        return

    deadline = time.monotonic() + float(timeout)
    while True:
        try:
            portalocker.lock(
                file_handle,
                portalocker.LOCK_EX | portalocker.LOCK_NB,
            )
            return
        except portalocker.exceptions.LockException:
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for shared memory lock")
            time.sleep(poll_interval)


def _release_file_lock(file_handle) -> None:
    portalocker.unlock(file_handle)


def _unregister(shm: shared_memory.SharedMemory) -> None:
    if os.name == "nt":
        return
    name = getattr(shm, "_name", None)
    if not name:
        return
    try:
        resource_tracker.unregister(name, "shared_memory")
    except Exception:
        pass


def _can_directly_unlink_posix_segments() -> bool:
    return bool(
        os.name != "nt"
        and hasattr(shared_memory, "_posixshmem")
        and hasattr(shared_memory._posixshmem, "shm_unlink")
    )


def _normalize_segment_name(name: str) -> str:
    return name if name.startswith("/") else f"/{name}"


def _safe_posix_shm_unlink(name: str) -> None:
    try:
        shared_memory._posixshmem.shm_unlink(_normalize_segment_name(name))
    except FileNotFoundError:
        pass


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _collect_cuda_ipc() -> None:
    """Force torch to release this process's freed CUDA IPC ref-count files.

    Torch shares GPU tensors across processes by ref-counting them through
    ``cuda.shm.*`` segments in ``/dev/shm``.  Calling ``ipc_collect`` lets
    torch reclaim the ones whose tensors have been dropped.  No-op sans CUDA.
    """
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.path.isdir(f"/proc/{pid}"):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except OSError:
        return True  # be conservative: assume alive
    return True


def _cuda_ipc_file_producer_pid(base: str) -> int | None:
    """Parse the producer PID from a torch CUDA IPC filename.

    Torch names its ref-count segments ``cuda.shm.<id>.<pid_hex>.<seq>`` where
    ``<pid_hex>`` is the hex PID of the producer process.  Returns the PID, or
    ``None`` if the name does not match that layout.
    """
    parts = base.split(".")
    if len(parts) < 4 or parts[0] != "cuda" or parts[1] != "shm":
        return None
    try:
        return int(parts[-2], 16)
    except ValueError:
        return None


def _remove_orphaned_cuda_ipc_files() -> list[str]:
    """Unlink *orphaned* torch CUDA IPC ref-count files from ``/dev/shm``.

    When a producer process exits before releasing its shared CUDA tensors
    the ``cuda.shm.*`` ref-count segments it created are orphaned.  ``purge``
    sweeps these as part of clearing all pyshmem state.  A file is removed only
    when its producer PID (encoded in the filename) belongs to a process that
    is no longer alive — removing a *live* producer's ref-count file would
    corrupt that process's CUDA tensors.  Files whose name we cannot parse are
    left alone.  Returns the basenames removed.
    """
    if os.name == "nt":
        return []
    shm_dir = "/dev/shm"
    if not os.path.isdir(shm_dir):
        return []
    removed = []
    for path in glob.glob(os.path.join(shm_dir, "cuda.shm.*")):
        base = os.path.basename(path)
        producer_pid = _cuda_ipc_file_producer_pid(base)
        if producer_pid is None or _pid_is_alive(producer_pid):
            continue
        _safe_remove(path)
        removed.append(base)
    return sorted(removed)


def _normalize_shape(shape: Sequence[int]) -> tuple[int, ...]:
    if not shape:
        raise ValueError("shape must contain at least one dimension")
    normalized = tuple(int(axis) for axis in shape)
    if any(axis <= 0 for axis in normalized):
        raise ValueError("shape dimensions must be positive")
    if len(normalized) > METADATA_SIZE - METADATA_INDEX_SHAPE_START:
        raise ValueError("shape has too many dimensions for metadata storage")
    return normalized


def _normalize_dtype(dtype: Any) -> np.dtype:
    normalized = np.dtype(dtype)
    if normalized not in DTYPE_TO_CODE:
        raise ValueError(f"unsupported dtype: {normalized}")
    return normalized


def _normalize_cpu_mirror(
    resolved_gpu: Any | None, cpu_mirror: bool | None
) -> bool:
    if resolved_gpu is None:
        return True
    if cpu_mirror is None:
        return False
    return bool(cpu_mirror)


def _normalize_size(
    shape: tuple[int, ...], dtype: np.dtype, size: int | None
) -> int:
    expected = math.prod(shape) * dtype.itemsize
    if size is None:
        return expected
    if int(size) != expected:
        message = (
            "size does not match shape and dtype: "
            f"expected {expected}, got {size}"
        )
        raise ValueError(message)
    return expected


def _dtype_to_code(dtype: np.dtype) -> int:
    return DTYPE_TO_CODE[np.dtype(dtype)]


def _code_to_dtype(code: float) -> np.dtype:
    if not np.isfinite(code) or float(code) != int(code):
        raise ValueError(f"invalid dtype code in metadata: {code}")
    index = int(code)
    if index < 0 or index >= len(DTYPE_TABLE):
        raise ValueError(f"invalid dtype code in metadata: {code}")
    return DTYPE_TABLE[index]


def _metadata_integer(value, field: str) -> int:
    """Decode an integer metadata field without accepting truncation."""
    try:
        numeric = float(value)
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {field} in metadata: {value!r}") from exc
    if not np.isfinite(numeric) or numeric != result:
        raise ValueError(f"invalid {field} in metadata: {value!r}")
    return result


def _decode_metadata_header(
    metadata: _MetadataView,
    metadata_shm: shared_memory.SharedMemory,
    *,
    expected_name: str,
) -> dict[str, Any]:
    """Validate metadata fields and return a normalized stream description."""
    if metadata.layout_version == METADATA_VERSION:
        if len(metadata_shm.buf) != METADATA_TOTAL_BYTES:
            raise ValueError(
                f"invalid metadata segment size: expected "
                f"{METADATA_TOTAL_BYTES}, got {len(metadata_shm.buf)}"
            )
        flags = int(metadata._v3["flags"])
        known_flags = (
            METADATA_FLAG_GPU_ENABLED | METADATA_FLAG_CPU_MIRROR_ENABLED
        )
        if flags & ~known_flags:
            raise ValueError(f"unsupported metadata flags: 0x{flags:x}")
        if any(bytes(metadata._v3["reserved"])):
            raise ValueError("reserved metadata bytes must be zero")

    stored_name = _read_stream_name(metadata_shm)
    if metadata.layout_version == METADATA_VERSION and stored_name is None:
        raise ValueError("version 3 metadata does not contain a stream name")
    if stored_name is not None and stored_name != expected_name:
        raise ValueError(
            f"metadata name mismatch: expected {expected_name!r}, "
            f"found {stored_name!r}"
        )

    dtype = _code_to_dtype(metadata[METADATA_INDEX_DTYPE])
    ndim = _metadata_integer(metadata[METADATA_INDEX_NDIM], "ndim")
    max_ndim = METADATA_SIZE - METADATA_INDEX_SHAPE_START
    if ndim <= 0 or ndim > max_ndim:
        raise ValueError(f"invalid ndim in metadata: {ndim}")
    shape = tuple(
        _metadata_integer(
            metadata[METADATA_INDEX_SHAPE_START + index],
            f"shape[{index}]",
        )
        for index in range(ndim)
    )
    if any(axis <= 0 for axis in shape):
        raise ValueError(f"invalid shape in metadata: {shape}")
    if metadata.layout_version == METADATA_VERSION:
        unused_shape = metadata._v3["shape"][ndim:]
        if np.any(unused_shape != 0):
            raise ValueError("unused shape entries in metadata must be zero")

    size = _metadata_integer(metadata[METADATA_INDEX_SIZE], "size")
    expected_size = math.prod(shape) * dtype.itemsize
    if size != expected_size:
        raise ValueError(
            f"metadata size does not match shape and dtype: "
            f"expected {expected_size}, got {size}"
        )

    gpu_raw = _metadata_integer(
        metadata[METADATA_INDEX_GPU_ENABLED], "gpu_enabled"
    )
    mirror_raw = _metadata_integer(
        metadata[METADATA_INDEX_CPU_MIRROR_ENABLED], "cpu_mirror"
    )
    if gpu_raw not in (0, 1) or mirror_raw not in (0, 1):
        raise ValueError("metadata boolean fields must be zero or one")
    gpu_enabled = bool(gpu_raw)
    cpu_mirror = bool(mirror_raw)
    device_index = _metadata_integer(
        metadata[METADATA_INDEX_DEVICE_INDEX], "device_index"
    )
    if gpu_enabled and device_index < 0:
        raise ValueError("GPU metadata requires a non-negative device index")
    if not gpu_enabled and device_index != -1:
        raise ValueError("CPU metadata requires device_index=-1")
    if not gpu_enabled and not cpu_mirror:
        raise ValueError("CPU metadata requires its shared payload")

    creator_pid = _metadata_integer(
        metadata[METADATA_INDEX_CREATOR_PID], "creator_pid"
    )
    if creator_pid <= 0:
        raise ValueError(f"invalid creator_pid in metadata: {creator_pid}")
    count = _metadata_integer(metadata[METADATA_INDEX_COUNT], "count")
    if count < 0:
        raise ValueError(f"invalid count in metadata: {count}")
    _metadata_integer(
        metadata[METADATA_INDEX_WRITE_SEQUENCE], "write_sequence"
    )
    write_time = float(metadata[METADATA_INDEX_WRITE_TIME])
    if not np.isfinite(write_time) or write_time < 0:
        raise ValueError(f"invalid write_time in metadata: {write_time}")
    lock_owner = _metadata_integer(
        metadata[METADATA_INDEX_LOCK_OWNER_PID], "lock_owner_pid"
    )
    lock_depth = _metadata_integer(
        metadata[METADATA_INDEX_LOCK_DEPTH], "lock_depth"
    )
    if lock_owner < 0 or lock_depth < 0:
        raise ValueError("lock metadata cannot be negative")
    if (lock_owner == 0) != (lock_depth == 0):
        raise ValueError("lock owner and depth metadata are inconsistent")

    return {
        "dtype": dtype,
        "shape": shape,
        "size": size,
        "gpu_enabled": gpu_enabled,
        "device_index": device_index,
        "creator_pid": creator_pid,
        "cpu_mirror": cpu_mirror,
    }


def _normalize_gpu_device(gpu_device: str | int | None) -> Any | None:
    if gpu_device is None:
        return None
    if torch is None:
        raise RuntimeError("PyTorch is required for GPU shared memory")
    device = torch.device(gpu_device)
    if device.type != "cuda":
        raise ValueError("only CUDA devices are currently supported")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current PyTorch installation"
        )
    index = 0 if device.index is None else int(device.index)
    return torch.device(f"cuda:{index}")


def _torch_dtype_for(dtype: np.dtype):
    torch_dtype = TORCH_DTYPE_MAP.get(np.dtype(dtype))
    if torch_dtype is None:
        _supported = ", ".join(str(d) for d in TORCH_DTYPE_MAP)
        raise ValueError(
            f"dtype {dtype} is not supported for GPU shared memory; "
            f"supported: {_supported}"
        )
    return torch_dtype


def _resolve_open_target_device(
    name: str,
    device_index: int,
    gpu_device: str | int | None,
    cpu_mirror: bool,
):
    """Decide which CUDA device ``open()`` should attach a GPU stream to.

    ``open()`` reconstructs the stream as it was created, so by default
    (``gpu_device is None``) it attaches to the device recorded in metadata.

    Returns ``(device, strict)`` where ``device`` is a ``torch.device`` to
    attach to, or ``None`` to open without a GPU attachment — only viable when
    the stream carries a CPU mirror.  ``strict`` is ``True`` when the caller
    named a device explicitly, in which case a failed attach must propagate
    rather than fall back to the mirror.  Raises when the stream needs a GPU
    that this host cannot provide and there is no CPU mirror to fall back on.
    """
    if gpu_device is not None:
        resolved = _normalize_gpu_device(gpu_device)
        if device_index < 0:
            raise ValueError(
                f"{name!r} does not advertise a valid CUDA device"
            )
        if resolved.index != device_index:
            raise ValueError(
                f"requested GPU device {resolved} does not match stored "
                f"device cuda:{device_index}"
            )
        return resolved, True

    if device_index < 0:
        raise ValueError(
            f"{name!r} is GPU-backed but advertises no CUDA device"
        )
    if torch is None or not torch.cuda.is_available():
        if cpu_mirror:
            return None, False
        raise RuntimeError(
            f"{name!r} is a GPU stream on cuda:{device_index} but CUDA is not "
            "available in this process; install a CUDA-enabled torch, or "
            "recreate the stream with cpu_mirror=True for CPU access"
        )
    if device_index >= torch.cuda.device_count():
        if cpu_mirror:
            return None, False
        raise RuntimeError(
            f"{name!r} is on cuda:{device_index} but this host only has "
            f"{torch.cuda.device_count()} CUDA device(s)"
        )
    return torch.device(f"cuda:{device_index}"), False


def _open_existing_segment(name: str) -> shared_memory.SharedMemory | None:
    try:
        shm = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return None
    _unregister(shm)
    return shm


def _missing_name_error(name: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"shared memory {name!r} does not exist; "
        f"create it with pyshmem.create({name!r}, ...) first"
    )


def _duplicate_name_error(name: str) -> FileExistsError:
    return FileExistsError(
        f"shared memory {name!r} already exists; "
        f"use pyshmem.open({name!r}) to attach to it"
    )


def purge(*, include_cuda_orphans: bool = False) -> list[str]:
    """Remove all pyshmem segments from shared memory.

    Scans ``/dev/shm`` for data segments whose metadata name hashes back to the
    same pyshmem segment id.  Only validated streams and their exact metadata,
    GPU-handle, and lock files are removed; unrelated ``ps_*`` objects are
    preserved.  Returns the user-visible names of the streams removed.

    By default this function does not touch PyTorch's process-global
    ``cuda.shm.*`` files.  Set ``include_cuda_orphans=True`` to explicitly
    sweep files whose encoded producer PID is no longer alive.  That operation
    is broader than pyshmem and may remove orphaned files from other PyTorch
    applications running under the same OS account.

    This is the correct tool for cleaning up after a test run or clearing a
    machine that has accumulated stale streams.  It is *not* reversible.

    Use ``purge`` when you want to remove every validated pyshmem stream
    without enumerating individual names.
    """
    if os.name == "nt":
        return []
    shm_dir = "/dev/shm"
    if not os.path.isdir(shm_dir):
        return []
    validated = []
    for path in glob.glob(os.path.join(shm_dir, "ps_*")):
        base = os.path.basename(path)
        if base.endswith("_meta") or base.endswith("_gpu"):
            continue
        stream_name = _validated_stream_name_for_base(base)
        if stream_name is not None:
            validated.append((base, stream_name))

    segment_names = set()
    for base, _ in validated:
        for candidate in (base, f"{base}_meta", f"{base}_gpu"):
            if os.path.exists(os.path.join(shm_dir, candidate)):
                segment_names.add(candidate)
    if _can_directly_unlink_posix_segments():
        for segment_name in sorted(segment_names):
            _safe_posix_shm_unlink(segment_name)
    else:
        for segment_name in sorted(segment_names):
            shm = _open_existing_segment(segment_name)
            if shm is not None:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
                finally:
                    try:
                        shm.close()
                    except Exception:
                        pass
    uid = getattr(os, "getuid", lambda: 0)()
    lock_dir = os.environ.get("PYSHMEM_LOCK_DIR") or os.path.join(
        tempfile.gettempdir(), f"pyshmem-locks-{uid}"
    )
    for base, stream_name in validated:
        _safe_remove(os.path.join(lock_dir, f"{base}.lock"))
        _LOCAL_GPU_TENSORS.pop(stream_name, None)
    _collect_cuda_ipc()
    if include_cuda_orphans:
        _remove_orphaned_cuda_ipc_files()
    return sorted(stream_name for _, stream_name in validated)


def unlink_quiet(name: str) -> None:
    """Destroy a stream by name, silently succeeding when it does not exist.

    Identical to :func:`unlink` but explicitly documented as safe to call
    without first checking whether the stream exists.  Useful for shutdown
    and cleanup code that must be resilient to partially-created streams or
    double-unlink calls.
    """
    unlink(name)


def unlink(name: str) -> None:
    _LOCAL_GPU_TENSORS.pop(name, None)
    if _can_directly_unlink_posix_segments():
        for segment_name in (
            _data_name(name),
            _metadata_name(name),
            _gpu_handle_name(name),
        ):
            _safe_posix_shm_unlink(segment_name)
        _safe_remove(_lock_path(name))
        return
    for segment_name in (
        _data_name(name),
        _metadata_name(name),
        _gpu_handle_name(name),
    ):
        shm = _open_existing_segment(segment_name)
        if shm is None:
            continue
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        finally:
            try:
                shm.close()
            except Exception:
                pass
    _safe_remove(_lock_path(name))


def _stream_name_for_base(base: str) -> str | None:
    """Recover the user-visible name stored in a stream's metadata segment."""
    try:
        meta_shm = shared_memory.SharedMemory(name=f"{base}_meta")
    except FileNotFoundError:
        return None
    _unregister(meta_shm)
    try:
        return _read_stream_name(meta_shm)
    finally:
        try:
            meta_shm.close()
        except Exception:
            pass


def _validated_stream_name_for_base(base: str) -> str | None:
    """Return a name only when metadata proves ``base`` is ours."""
    if (
        len(base) != 17
        or not base.startswith("ps_")
        or any(char not in "0123456789abcdef" for char in base[3:])
    ):
        return None
    try:
        meta_shm = shared_memory.SharedMemory(name=f"{base}_meta")
    except FileNotFoundError:
        return None
    _unregister(meta_shm)
    try:
        if len(meta_shm.buf) < METADATA_TOTAL_BYTES:
            return None
        stream_name = _read_stream_name(meta_shm)
        if stream_name is None or _segment_base_name(stream_name) != base:
            return None
        metadata = _MetadataView(meta_shm.buf)
        _decode_metadata_header(metadata, meta_shm, expected_name=stream_name)
        return stream_name
    except (TypeError, ValueError, BufferError):
        return None
    finally:
        try:
            meta_shm.close()
        except Exception:
            pass


def list_streams() -> list[str]:
    """Return the user-visible names of all existing pyshmem streams.

    On Linux, scans ``/dev/shm/`` for ``ps_*`` data-segment files and recovers
    the original name passed to :func:`create` from each stream's metadata.
    A candidate is listed only when its stored name hashes back to the exact
    segment id; legacy or unrelated ``ps_*`` objects without validated metadata
    are ignored. Returns an empty list on platforms where scanning is not
    supported.
    """
    if os.name == "nt":
        return []
    shm_dir = "/dev/shm"
    if not os.path.isdir(shm_dir):
        return []
    result = []
    for path in glob.glob(os.path.join(shm_dir, "ps_*")):
        base = os.path.basename(path)
        if base.endswith("_meta") or base.endswith("_gpu"):
            continue
        stream_name = _validated_stream_name_for_base(base)
        if stream_name is not None:
            result.append(stream_name)
    return sorted(result)


class SharedMemory:
    """A named shared-memory stream.

    Instances are created via :func:`create` or attached to via
    :func:`open`. The object exposes shape and dtype metadata, lock
    management, read and write operations, and lifecycle helpers such as
    :meth:`close` and :meth:`unlink`.

    For GPU-backed streams, ``gpu_device`` identifies the attached CUDA device.
    When ``cpu_mirror`` is ``False``, CPU-only handles may still inspect
    metadata and take locks, but they cannot read the payload without
    reopening with a CUDA attachment.
    """

    def __init__(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        size: int,
        gpu_device: str | None,
        gpu_enabled: bool,
        cpu_mirror: bool,
        data_shm: shared_memory.SharedMemory,
        metadata_shm: shared_memory.SharedMemory,
        owner: bool,
        gpu_handle_shm: shared_memory.SharedMemory | None = None,
        gpu_tensor=None,
        torch_dtype=None,
    ) -> None:
        self.name = name
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.size = int(size)
        self.gpu_device = gpu_device
        self.gpu_enabled = bool(gpu_enabled)
        self.cpu_mirror = bool(cpu_mirror)
        self.owner = bool(owner)
        self._data_shm = data_shm
        self._metadata_shm = metadata_shm
        self._gpu_handle_shm = gpu_handle_shm
        self._array = np.ndarray(
            self.shape, dtype=self.dtype, buffer=self._data_shm.buf
        )
        self._metadata = _MetadataView(self._metadata_shm.buf)
        self._gpu_tensor = gpu_tensor
        self._torch_dtype = torch_dtype
        self._last_seen_count = int(self._metadata[METADATA_INDEX_COUNT])
        self._lock_state = _lock_state(name)
        self._lock_state_released = False
        self._closed = False
        self._auto_unlink = False

    def __repr__(self) -> str:
        dtype_name = str(self.dtype)
        return (
            f"SharedMemory(name={self.name!r}, "
            f"shape={self.shape!r}, "
            f"dtype={dtype_name!r}, "
            f"gpu_device={self.gpu_device!r})"
        )

    @property
    def count(self) -> int:
        """Return the number of completed writes recorded on the stream."""
        self._ensure_open("read count from")
        return int(self._metadata[METADATA_INDEX_COUNT])

    @property
    def write_time(self) -> float:
        """Return the UNIX timestamp of the most recent completed write."""
        self._ensure_open("read write_time from")
        return float(self._metadata[METADATA_INDEX_WRITE_TIME])

    @property
    def write_sequence(self) -> int:
        """Return the internal write sequence counter for the stream."""
        self._ensure_open("read write_sequence from")
        return int(self._metadata[METADATA_INDEX_WRITE_SEQUENCE])

    def _ensure_open(self, operation: str) -> None:
        if self._closed:
            raise RuntimeError(
                f"cannot {operation} closed shared memory {self.name!r}; "
                f"reopen it with pyshmem.open({self.name!r})"
            )

    def _lock_owned_by_current_thread(self) -> bool:
        return self._lock_state.owner_thread_id == threading.get_ident()

    def _invalidate_abandoned_write(self, observed_sequence: int) -> bool:
        """Mark an odd generation invalid after its writer process died."""
        owner_pid = int(self._metadata[METADATA_INDEX_LOCK_OWNER_PID])
        if owner_pid <= 0 or _pid_is_alive(owner_pid):
            return False
        try:
            self.acquire(timeout=0.0)
        except TimeoutError:
            return False
        try:
            current = self.write_sequence
            if current == observed_sequence and current > 0 and current % 2:
                self._metadata[METADATA_INDEX_WRITE_SEQUENCE] = -current
                return True
            return current < 0
        finally:
            self.release()

    def _wait_for_stable_writer(
        self, poll_interval: float, timeout: float | None = None
    ) -> int:
        deadline = (
            None if timeout is None else time.monotonic() + float(timeout)
        )
        while True:
            sequence = self.write_sequence
            if sequence < 0:
                raise InconsistentStreamError(
                    f"stream {self.name!r} contains an incomplete write; "
                    "a successful write is required before it can be read"
                )
            if sequence % 2 == 0:
                return sequence
            if self._invalidate_abandoned_write(sequence):
                raise InconsistentStreamError(
                    f"writer process for stream {self.name!r} exited during "
                    "a write; a successful write is required before it can "
                    "be read"
                )
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for a stable write on {self.name!r}"
                )
            time.sleep(poll_interval)

    def _finish_write(self) -> None:
        self._metadata[METADATA_INDEX_COUNT] += 1
        self._metadata[METADATA_INDEX_WRITE_TIME] = time.time()
        self._metadata[METADATA_INDEX_WRITE_SEQUENCE] += 1

    def _mark_write_started(self) -> None:
        sequence = int(self._metadata[METADATA_INDEX_WRITE_SEQUENCE])
        if sequence < 0:
            # A previous copy failed or its writer died.  The new write fully
            # replaces the payload, so start a fresh odd generation beyond it.
            sequence = abs(sequence)
            self._metadata[METADATA_INDEX_WRITE_SEQUENCE] = (
                sequence + 2 if sequence % 2 else sequence + 1
            )
            return
        self._metadata[METADATA_INDEX_WRITE_SEQUENCE] = sequence + 1

    def _abort_write(self) -> None:
        """Publish an invalid generation without claiming partial data."""
        sequence = int(self._metadata[METADATA_INDEX_WRITE_SEQUENCE])
        if sequence >= 0:
            self._metadata[METADATA_INDEX_WRITE_SEQUENCE] = -max(sequence, 1)

    def _lock_metadata_on_acquire(self) -> None:
        self._metadata[METADATA_INDEX_LOCK_OWNER_PID] = os.getpid()
        self._metadata[METADATA_INDEX_LOCK_DEPTH] = self._lock_state.depth

    def _lock_metadata_on_release(self) -> None:
        if self._lock_state.depth == 0:
            self._metadata[METADATA_INDEX_LOCK_OWNER_PID] = 0
            self._metadata[METADATA_INDEX_LOCK_DEPTH] = 0
            return
        self._metadata[METADATA_INDEX_LOCK_DEPTH] = self._lock_state.depth

    def _read_consistent_cpu(
        self, poll_interval: float, out=None, timeout: float | None = None
    ):
        deadline = (
            None if timeout is None else time.monotonic() + float(timeout)
        )
        while True:
            remaining = (
                None
                if deadline is None
                else max(0.0, deadline - time.monotonic())
            )
            start_sequence = self._wait_for_stable_writer(
                poll_interval, timeout=remaining
            )
            if out is not None:
                np.copyto(out, self._array)
                result = out
            else:
                result = np.copy(self._array)
            end_sequence = self.write_sequence
            if start_sequence == end_sequence:
                self._last_seen_count = self.count
                return result

    def _read_consistent_gpu(
        self, poll_interval: float, timeout: float | None = None
    ):
        deadline = (
            None if timeout is None else time.monotonic() + float(timeout)
        )
        if self.cpu_mirror:
            while True:
                remaining = (
                    None
                    if deadline is None
                    else max(0.0, deadline - time.monotonic())
                )
                start_sequence = self._wait_for_stable_writer(
                    poll_interval, timeout=remaining
                )
                cpu_snapshot = np.copy(self._array)
                end_sequence = self.write_sequence
                if start_sequence == end_sequence:
                    self._last_seen_count = self.count
                    result = torch.as_tensor(
                        cpu_snapshot,
                        dtype=self._torch_dtype,
                        device=self.gpu_device,
                    )
                    torch.cuda.synchronize(device=self.gpu_device)
                    return result

        while True:
            remaining = (
                None
                if deadline is None
                else max(0.0, deadline - time.monotonic())
            )
            start_sequence = self._wait_for_stable_writer(
                poll_interval, timeout=remaining
            )
            result = self._gpu_tensor.clone()
            torch.cuda.synchronize(device=self.gpu_device)
            end_sequence = self.write_sequence
            if start_sequence == end_sequence:
                self._last_seen_count = self.count
                return result

    def acquire(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float = 1e-3,
    ) -> None:
        """Acquire the cross-process write lock for the stream.

        The lock is re-entrant within the current thread. When ``timeout`` is
        provided, a :class:`TimeoutError` is raised if the lock cannot be
        acquired before the deadline.
        """
        self._ensure_open("acquire")
        deadline = (
            None if timeout is None else time.monotonic() + float(timeout)
        )
        if deadline is None:
            acquired_thread_lock = self._lock_state.thread_lock.acquire()
        else:
            remaining = max(0.0, deadline - time.monotonic())
            acquired_thread_lock = self._lock_state.thread_lock.acquire(
                timeout=remaining
            )
        if not acquired_thread_lock:
            raise TimeoutError("timed out waiting for shared memory lock")
        thread_id = threading.get_ident()
        if self._lock_state.owner_thread_id == thread_id:
            self._lock_state.depth += 1
            self._lock_metadata_on_acquire()
            return

        try:
            _acquire_file_lock(
                self._lock_state.file_handle,
                timeout=(
                    None
                    if deadline is None
                    else max(0.0, deadline - time.monotonic())
                ),
                poll_interval=poll_interval,
            )
        except Exception:
            self._lock_state.thread_lock.release()
            raise

        self._lock_state.owner_thread_id = thread_id
        self._lock_state.depth = 1
        self._lock_metadata_on_acquire()

    def release(self) -> None:
        """Release one level of the current thread's re-entrant lock state."""
        self._ensure_open("release")
        thread_id = threading.get_ident()
        if self._lock_state.owner_thread_id is None:
            raise RuntimeError("cannot release an unlocked shared memory lock")
        if self._lock_state.owner_thread_id != thread_id:
            raise RuntimeError("cannot release a lock owned by another thread")

        self._lock_state.depth -= 1
        self._lock_metadata_on_release()
        if self._lock_state.depth == 0:
            self._lock_state.owner_thread_id = None
            _release_file_lock(self._lock_state.file_handle)
        self._lock_state.thread_lock.release()

    @contextmanager
    def locked(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float = 1e-3,
    ):
        """Return a context manager for the stream lock."""
        self.acquire(timeout=timeout, poll_interval=poll_interval)
        try:
            yield self
        finally:
            self.release()

    @classmethod
    def _create(
        cls,
        name: str,
        *,
        shape: Sequence[int],
        dtype: Any = np.float32,
        size: int | None = None,
        gpu_device: str | int | None = None,
        cpu_mirror: bool | None = None,
    ) -> "SharedMemory":
        _encode_stream_name(name)  # validate length before allocating segments
        normalized_shape = _normalize_shape(shape)
        normalized_dtype = _normalize_dtype(dtype)
        normalized_size = _normalize_size(
            normalized_shape, normalized_dtype, size
        )
        resolved_gpu = _normalize_gpu_device(gpu_device)
        cpu_mirror_enabled = _normalize_cpu_mirror(resolved_gpu, cpu_mirror)
        torch_dtype = (
            _torch_dtype_for(normalized_dtype)
            if resolved_gpu is not None
            else None
        )

        try:
            data_shm = shared_memory.SharedMemory(
                name=_data_name(name), create=True, size=normalized_size
            )
        except FileExistsError as exc:
            raise _duplicate_name_error(name) from exc

        try:
            metadata_shm = shared_memory.SharedMemory(
                name=_metadata_name(name),
                create=True,
                size=METADATA_TOTAL_BYTES,
            )
        except FileExistsError as exc:
            try:
                data_shm.close()
                data_shm.unlink()
            except Exception:
                pass
            raise _duplicate_name_error(name) from exc
        _unregister(data_shm)
        _unregister(metadata_shm)

        gpu_handle_shm = None
        gpu_tensor = None
        try:
            array = np.ndarray(
                normalized_shape, dtype=normalized_dtype, buffer=data_shm.buf
            )
            array.fill(0)
            metadata = _MetadataView(metadata_shm.buf, initialize=True)

            if resolved_gpu is not None:
                gpu_tensor, gpu_handle_shm = _create_gpu_tensor_and_handle(
                    name=name,
                    shape=normalized_shape,
                    torch_dtype=torch_dtype,
                    gpu_device=resolved_gpu,
                )

            metadata[METADATA_INDEX_VERSION] = METADATA_VERSION
            metadata[METADATA_INDEX_COUNT] = 0
            metadata[METADATA_INDEX_DTYPE] = _dtype_to_code(normalized_dtype)
            metadata[METADATA_INDEX_NDIM] = len(normalized_shape)
            metadata[METADATA_INDEX_SIZE] = normalized_size
            metadata[METADATA_INDEX_GPU_ENABLED] = (
                1 if resolved_gpu is not None else 0
            )
            metadata[METADATA_INDEX_DEVICE_INDEX] = (
                -1 if resolved_gpu is None else resolved_gpu.index
            )
            metadata[METADATA_INDEX_CREATOR_PID] = os.getpid()
            metadata[METADATA_INDEX_WRITE_TIME] = 0.0
            metadata[METADATA_INDEX_WRITE_SEQUENCE] = 0
            metadata[METADATA_INDEX_LOCK_OWNER_PID] = 0
            metadata[METADATA_INDEX_LOCK_DEPTH] = 0
            metadata[METADATA_INDEX_CPU_MIRROR_ENABLED] = (
                1 if cpu_mirror_enabled else 0
            )
            for index, axis in enumerate(normalized_shape):
                metadata[METADATA_INDEX_SHAPE_START + index] = axis
            _write_stream_name(metadata_shm, name)
        except Exception:
            try:
                data_shm.close()
                data_shm.unlink()
            except Exception:
                pass
            try:
                metadata_shm.close()
                metadata_shm.unlink()
            except Exception:
                pass
            if gpu_handle_shm is not None:
                try:
                    gpu_handle_shm.close()
                    gpu_handle_shm.unlink()
                except Exception:
                    pass
            # Remove any stale weakref entry inserted by
            # _create_gpu_tensor_and_handle.
            _LOCAL_GPU_TENSORS.pop(name, None)
            raise

        return cls(
            name=name,
            shape=normalized_shape,
            dtype=normalized_dtype,
            size=normalized_size,
            gpu_device=None if resolved_gpu is None else str(resolved_gpu),
            gpu_enabled=resolved_gpu is not None,
            cpu_mirror=cpu_mirror_enabled,
            data_shm=data_shm,
            metadata_shm=metadata_shm,
            owner=True,
            gpu_handle_shm=gpu_handle_shm,
            gpu_tensor=gpu_tensor,
            torch_dtype=torch_dtype,
        )

    @classmethod
    def _open(
        cls,
        name: str,
        *,
        gpu_device: str | int | bool | None = None,
    ) -> "SharedMemory":
        try:
            metadata_shm = shared_memory.SharedMemory(
                name=_metadata_name(name)
            )
        except FileNotFoundError as exc:
            raise _missing_name_error(name) from exc
        _unregister(metadata_shm)
        try:
            metadata = _MetadataView(metadata_shm.buf)
        except ValueError as exc:
            metadata_shm.close()
            raise ValueError(
                f"{name!r} does not contain a supported pyshmem metadata block"
            ) from exc

        try:
            decoded = _decode_metadata_header(
                metadata, metadata_shm, expected_name=name
            )
        except ValueError:
            metadata_shm.close()
            raise
        dtype = decoded["dtype"]
        shape = decoded["shape"]
        size = decoded["size"]
        gpu_enabled = decoded["gpu_enabled"]
        device_index = decoded["device_index"]
        creator_pid = decoded["creator_pid"]
        cpu_mirror_enabled = decoded["cpu_mirror"]

        try:
            data_shm = shared_memory.SharedMemory(name=_data_name(name))
        except FileNotFoundError as exc:
            metadata_shm.close()
            raise _missing_name_error(name) from exc
        _unregister(data_shm)
        if data_shm.size != size:
            metadata_shm.close()
            data_shm.close()
            raise ValueError(
                f"data segment size for {name!r} does not match metadata: "
                f"expected {size}, got {data_shm.size}"
            )

        resolved_gpu = None
        gpu_tensor = None
        gpu_handle_shm = None
        torch_dtype = None
        if gpu_enabled and gpu_device is False:
            # Explicit CPU-only attach: read the host mirror without mapping
            # the producer's CUDA tensor.  Only viable when a mirror exists.
            if not cpu_mirror_enabled:
                metadata_shm.close()
                data_shm.close()
                raise ValueError(
                    f"{name!r} is a GPU stream without a CPU mirror; cannot "
                    "open it CPU-only (recreate it with cpu_mirror=True)"
                )
        elif gpu_enabled:
            try:
                target_device, strict = _resolve_open_target_device(
                    name, device_index, gpu_device, cpu_mirror_enabled
                )
                if target_device is not None:
                    torch_dtype = _torch_dtype_for(dtype)
                    try:
                        gpu_tensor, gpu_handle_shm = (
                            _open_gpu_tensor_from_handle(
                                name=name,
                                shape=shape,
                                torch_dtype=torch_dtype,
                                creator_pid=creator_pid,
                            )
                        )
                        resolved_gpu = target_device
                    except Exception:
                        # The IPC handle could not be mapped (e.g. the producer
                        # process exited).  Fall back to the CPU mirror when
                        # the stream has one and the device was not requested
                        # explicitly; otherwise the failure is fatal.
                        if strict or not cpu_mirror_enabled:
                            raise
                        resolved_gpu = None
                        gpu_tensor = None
                        gpu_handle_shm = None
                        torch_dtype = None
            except Exception:
                metadata_shm.close()
                data_shm.close()
                raise

        return cls(
            name=name,
            shape=shape,
            dtype=dtype,
            size=size,
            gpu_device=None if resolved_gpu is None else str(resolved_gpu),
            gpu_enabled=gpu_enabled,
            cpu_mirror=cpu_mirror_enabled,
            data_shm=data_shm,
            metadata_shm=metadata_shm,
            owner=False,
            gpu_handle_shm=gpu_handle_shm,
            gpu_tensor=gpu_tensor,
            torch_dtype=torch_dtype,
        )

    def close(self) -> None:
        """Close this local handle without destroying the underlying stream."""
        if self._closed:
            return
        if self._lock_state.owner_thread_id is not None:
            if not self._lock_owned_by_current_thread():
                raise RuntimeError(
                    "cannot close shared memory while another thread "
                    "owns its lock"
                )
            while self._lock_state.owner_thread_id is not None:
                self.release()
        for segment in (
            self._gpu_handle_shm,
            self._metadata_shm,
            self._data_shm,
        ):
            if segment is None:
                continue
            try:
                segment.close()
            except Exception:
                pass
        self._gpu_handle_shm = None
        self._metadata_shm = None
        self._data_shm = None
        # Release a *consumer's* mapped CUDA tensor here so torch decrements
        # the producer's cross-process IPC ref counter; without this the
        # producer can never reclaim the GPU allocation (the counter stays
        # non-zero) and GPU memory grows with every attach.  The owner keeps
        # its reference on close so the stream remains mappable and reopenable
        # in-process; the owner only releases it in :meth:`unlink`, when the
        # stream is destroyed.
        if self._gpu_tensor is not None and not self.owner:
            self._gpu_tensor = None
        self._closed = True
        if not self._lock_state_released:
            _release_lock_state(self._lock_state)
            self._lock_state_released = True

    def unlink(self) -> None:
        """Destroy the underlying named shared-memory stream."""
        had_gpu_tensor = self._gpu_tensor is not None
        self.close()
        # The stream is being destroyed: drop the owner's CUDA tensor too
        # (close keeps it for the owner) so the producer memory can be freed,
        # then prompt torch to reclaim any IPC blocks whose consumers have
        # released them.
        self._gpu_tensor = None
        unlink(self.name)
        if had_gpu_tensor:
            _collect_cuda_ipc()

    def delete(self) -> None:
        """Alias for :meth:`unlink`."""
        self.unlink()

    def clear(self) -> None:
        """Reset the current payload to zeros and record a new write."""
        self._ensure_open("clear")
        if (
            self.gpu_enabled
            and self._gpu_tensor is None
            and not self.cpu_mirror
        ):
            raise RuntimeError(
                "cannot clear GPU shared memory without a GPU attachment; "
                "reopen it with "
                f"pyshmem.open({self.name!r}, gpu_device='cuda:N')"
            )
        with self.locked():
            self._mark_write_started()
            try:
                if self._gpu_tensor is not None:
                    self._gpu_tensor.zero_()
                if self.cpu_mirror:
                    self._array.fill(0)
                if self._gpu_tensor is not None:
                    torch.cuda.synchronize(device=self.gpu_device)
            except BaseException:
                self._abort_write()
                raise
            else:
                self._finish_write()

    def write(self, value: Any) -> None:
        """Write a full payload into the stream.

        ``value`` must match the configured shape. CPU-backed streams accept
        values understood by :func:`numpy.asarray`; GPU-backed streams also
        accept CUDA tensors on the configured device.
        """
        self._ensure_open("write to")
        tensor = None
        array = None
        if self._gpu_tensor is not None:
            tensor = torch.as_tensor(
                value, dtype=self._torch_dtype, device=self.gpu_device
            )
            if tuple(tensor.shape) != self.shape:
                message = (
                    f"expected shape {self.shape}, got {tuple(tensor.shape)}"
                )
                raise ValueError(message)
        elif self.gpu_enabled and not self.cpu_mirror:
            raise RuntimeError(
                "cannot write to GPU shared memory without a GPU attachment; "
                "reopen it with "
                f"pyshmem.open({self.name!r}, gpu_device='cuda:N')"
            )
        else:
            array = np.asarray(value, dtype=self.dtype)
            if tuple(array.shape) != self.shape:
                message = (
                    f"expected shape {self.shape}, got {tuple(array.shape)}"
                )
                raise ValueError(message)

        with self.locked():
            self._mark_write_started()
            try:
                if tensor is not None:
                    self._gpu_tensor.copy_(tensor)
                    if self.cpu_mirror:
                        np.copyto(self._array, tensor.detach().cpu().numpy())
                    torch.cuda.synchronize(device=self.gpu_device)
                else:
                    np.copyto(self._array, array)
            except BaseException:
                self._abort_write()
                raise
            else:
                self._finish_write()

    def read(
        self,
        *,
        safe: bool = True,
        poll_interval: float = 1e-6,
        out=None,
        timeout: float | None = None,
    ):
        """Read the current payload from the stream.

        When ``safe`` is ``True``, the method returns a consistent snapshot of
        the latest completed write. When ``safe`` is ``False``, the caller must
        already own the stream lock via :meth:`locked` or :meth:`acquire`.

        ``out`` may be a pre-allocated NumPy array with the correct shape and
        dtype; when supplied for CPU streams, the data is written into it
        directly without allocating a result.  ``out`` is not supported for
        GPU streams or in ``safe=False`` mode.

        ``timeout`` bounds how long a safe read waits for an in-progress write
        to finish.  A writer that exits mid-write raises
        :class:`InconsistentStreamError` immediately; a successful replacement
        write makes the stream readable again.
        """
        self._ensure_open("read from")
        if out is not None and (self._gpu_tensor is not None or not safe):
            raise ValueError(
                "out is supported only for safe CPU reads; pass a NumPy "
                "destination buffer to a CPU-backed handle"
            )
        if not safe:
            if not self._lock_owned_by_current_thread():
                raise RuntimeError(
                    "safe=False requires an active 'with shm.locked()' block"
                )
            self._last_seen_count = self.count
            if self._gpu_tensor is not None:
                return self._gpu_tensor
            if self.gpu_enabled and not self.cpu_mirror:
                raise RuntimeError(
                    f"GPU stream {self.name!r} is not attached on this handle "
                    "and has no CPU mirror; open it from a process with "
                    "access to its CUDA device"
                )
            return self._array

        if self._gpu_tensor is not None:
            return self._read_consistent_gpu(poll_interval, timeout=timeout)
        if self.gpu_enabled and not self.cpu_mirror:
            raise RuntimeError(
                "GPU shared memory was created without cpu_mirror=True; "
                "reopen it with "
                f"pyshmem.open({self.name!r}, gpu_device='cuda:N')"
            )
        return self._read_consistent_cpu(
            poll_interval, out=out, timeout=timeout
        )

    def read_new(
        self,
        *,
        timeout: float | None = None,
        safe: bool = True,
        poll_interval: float = 1e-5,
        out=None,
    ):
        """Block until a new write arrives, then return its payload.

        ``out`` is forwarded to :meth:`read`: a pre-allocated NumPy array
        receives the payload directly (zero-alloc) for CPU streams.
        """
        self._ensure_open("read from")
        baseline = self.count
        start = time.monotonic()
        while True:
            # Skip polling count while a write is in progress (odd sequence).
            sequence = self.write_sequence
            if sequence < 0:
                raise InconsistentStreamError(
                    f"stream {self.name!r} contains an incomplete write; "
                    "a successful write is required before it can be read"
                )
            if sequence % 2 == 1 and self._invalidate_abandoned_write(
                sequence
            ):
                raise InconsistentStreamError(
                    f"writer process for stream {self.name!r} exited during "
                    "a write; a successful write is required before it can "
                    "be read"
                )
            if sequence % 2 == 0 and self.count != baseline:
                break
            if timeout is not None and (time.monotonic() - start) >= float(
                timeout
            ):
                raise TimeoutError(
                    f"timed out waiting for a new write on {self.name!r}"
                )
            time.sleep(poll_interval)
        remaining = (
            None
            if timeout is None
            else max(0.0, float(timeout) - (time.monotonic() - start))
        )
        return self.read(safe=safe, out=out, timeout=remaining)

    def write_locked(self, value: Any) -> None:
        """Write a payload without acquiring the lock.

        Identical to :meth:`write` but skips the internal ``with
        self.locked()`` acquisition.  The caller must already own the lock via
        :meth:`locked` or :meth:`acquire`.  This is the intended public
        replacement for the private ``_mark_write_started`` / ``_finish_write``
        pattern used in high-performance consumers such as shmpipeline.
        """
        self._ensure_open("write to")
        if not self._lock_owned_by_current_thread():
            raise RuntimeError(
                "write_locked() requires an active 'with shm.locked()' block"
            )
        if self._gpu_tensor is not None:
            tensor = torch.as_tensor(
                value, dtype=self._torch_dtype, device=self.gpu_device
            )
            if tuple(tensor.shape) != self.shape:
                raise ValueError(
                    f"expected shape {self.shape}, got {tuple(tensor.shape)}"
                )
            self._mark_write_started()
            try:
                self._gpu_tensor.copy_(tensor)
                if self.cpu_mirror:
                    np.copyto(self._array, tensor.detach().cpu().numpy())
                torch.cuda.synchronize(device=self.gpu_device)
            except BaseException:
                self._abort_write()
                raise
            else:
                self._finish_write()
        elif self.gpu_enabled and not self.cpu_mirror:
            raise RuntimeError(
                "cannot write to GPU shared memory without a GPU attachment; "
                "reopen it with "
                f"pyshmem.open({self.name!r}, gpu_device='cuda:N')"
            )
        else:
            array = np.asarray(value, dtype=self.dtype)
            if tuple(array.shape) != self.shape:
                raise ValueError(
                    f"expected shape {self.shape}, got {tuple(array.shape)}"
                )
            self._mark_write_started()
            try:
                np.copyto(self._array, array)
            except BaseException:
                self._abort_write()
                raise
            else:
                self._finish_write()

    async def read_new_async(
        self,
        *,
        timeout: float | None = None,
        safe: bool = True,
        poll_interval: float = 1e-5,
        out=None,
    ):
        """Async variant of :meth:`read_new` that yields to the event loop.

        Uses :func:`asyncio.sleep` instead of :func:`time.sleep` so the
        caller's event loop is not blocked while waiting for a new write.
        """
        self._ensure_open("read from")
        baseline = self.count
        start = time.monotonic()
        while True:
            sequence = self.write_sequence
            if sequence < 0:
                raise InconsistentStreamError(
                    f"stream {self.name!r} contains an incomplete write; "
                    "a successful write is required before it can be read"
                )
            if sequence % 2 == 1 and self._invalidate_abandoned_write(
                sequence
            ):
                raise InconsistentStreamError(
                    f"writer process for stream {self.name!r} exited during "
                    "a write; a successful write is required before it can "
                    "be read"
                )
            if sequence % 2 == 0 and self.count != baseline:
                break
            if timeout is not None and (time.monotonic() - start) >= float(
                timeout
            ):
                raise TimeoutError(
                    f"timed out waiting for a new write on {self.name!r}"
                )
            await asyncio.sleep(poll_interval)
        remaining = (
            None
            if timeout is None
            else max(0.0, float(timeout) - (time.monotonic() - start))
        )
        return self.read(safe=safe, out=out, timeout=remaining)

    def describe(self) -> str:
        """Return a human-readable summary of the stream's metadata."""
        self._ensure_open("describe")
        lines = [
            f"name:         {self.name}",
            f"shape:        {self.shape}",
            f"dtype:        {self.dtype}",
            f"size:         {self.size} bytes",
            f"gpu_enabled:  {self.gpu_enabled}",
            f"gpu_device:   {self.gpu_device}",
            f"cpu_mirror:   {self.cpu_mirror}",
            f"count:        {self.count}",
            f"write_time:   {self.write_time}",
            f"write_seq:    {self.write_sequence}",
            f"owner:        {self.owner}",
        ]
        return "\n".join(lines)

    def to_config(self) -> dict:
        """Export stream configuration as a plain dictionary.

        The returned dict can be passed to :meth:`create_from_config` to
        recreate an identically-configured stream.
        """
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "gpu_device": self.gpu_device,
            "cpu_mirror": self.cpu_mirror,
        }

    @classmethod
    def create_from_config(cls, config: dict) -> "SharedMemory":
        """Create a new stream from a configuration dictionary.

        Accepts dicts produced by :meth:`to_config` or hand-written configs
        with ``name``, ``shape``, and optionally ``dtype``, ``gpu_device``,
        and ``cpu_mirror`` keys.
        """
        return create(
            config["name"],
            shape=config["shape"],
            dtype=config.get("dtype", "float32"),
            gpu_device=config.get("gpu_device"),
            cpu_mirror=config.get("cpu_mirror"),
        )

    def __enter__(self) -> "SharedMemory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._auto_unlink:
            self.unlink()
        else:
            self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _create_gpu_tensor_and_handle(
    *, name: str, shape: tuple[int, ...], torch_dtype, gpu_device: Any
):
    from torch.multiprocessing.reductions import reduce_tensor

    gpu_tensor = torch.empty(shape, dtype=torch_dtype, device=gpu_device)
    # Export the CUDA IPC handle through torch's official tensor reduction
    # rather than calling ``storage._share_cuda_()`` directly.  reduce_tensor
    # registers the storage in torch's ``shared_cache`` and wires up the
    # cross-process ref counter, which is what lets the producer's GPU memory
    # actually be reclaimed by ``torch.cuda.ipc_collect()`` once consumers
    # release the tensor and the stream is unlinked.  The direct
    # ``_share_cuda_()`` path bypasses that bookkeeping and leaks the
    # allocation for the lifetime of the producer process.
    rebuild_fn, rebuild_args = reduce_tensor(gpu_tensor)
    handle_payload = pickle.dumps((rebuild_fn, rebuild_args), protocol=4)
    handle_shm = shared_memory.SharedMemory(
        name=_gpu_handle_name(name), create=True, size=len(handle_payload)
    )
    _unregister(handle_shm)

    handle_shm.buf[: len(handle_payload)] = handle_payload
    _cache_gpu_tensor(name, gpu_tensor)

    return gpu_tensor, handle_shm


def _open_gpu_tensor_from_handle(
    *, name: str, shape: tuple[int, ...], torch_dtype, creator_pid: int
):
    # Acquire a per-name lock to serialise handle reconstruction across
    # threads; without it two threads could both find an empty cache and
    # independently reconstruct tensors from the same IPC handle, producing
    # two GPU tensors aliasing the same CUDA memory.
    with _GPU_OPEN_LOCKS_GUARD:
        if name not in _GPU_OPEN_LOCKS:
            _GPU_OPEN_LOCKS[name] = threading.Lock()
        name_lock = _GPU_OPEN_LOCKS[name]

    with name_lock:
        if creator_pid == os.getpid():
            gpu_tensor = _get_cached_gpu_tensor(name)
            if gpu_tensor is None:
                raise RuntimeError(
                    "cannot reopen GPU shared memory in the creator process "
                    "after all local GPU handles have been released"
                )
            return gpu_tensor, None

        # Non-creator path: check the cache first (another thread may have
        # already reconstructed the tensor while we waited for the lock).
        gpu_tensor = _get_cached_gpu_tensor(name)
        if gpu_tensor is not None:
            return gpu_tensor, None

        handle_shm = shared_memory.SharedMemory(name=_gpu_handle_name(name))
        _unregister(handle_shm)
        rebuild_fn, rebuild_args = pickle.loads(bytes(handle_shm.buf))

        torch.cuda._lazy_init()
        # ``rebuild_fn`` is torch's ``rebuild_cuda_tensor``; it maps the IPC
        # handle and installs the consumer-side ref-counted storage whose
        # finalizer decrements the producer's counter on release.  That
        # decrement is what eventually lets the producer reclaim the memory.
        tensor = rebuild_fn(*rebuild_args)
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != torch_dtype:
            raise ValueError(
                f"reconstructed GPU tensor for {name!r} does not match the "
                f"stream geometry (expected shape {tuple(shape)} dtype "
                f"{torch_dtype}, got shape {tuple(tensor.shape)} dtype "
                f"{tensor.dtype})"
            )
        _cache_gpu_tensor(name, tensor)
        return tensor, handle_shm


def create(
    name: str,
    *,
    shape: Sequence[int],
    dtype: Any = np.float32,
    size: int | None = None,
    gpu_device: str | int | None = None,
    cpu_mirror: bool | None = None,
    auto_unlink: bool = False,
) -> SharedMemory:
    """Create a new named shared-memory stream.

    Parameters
    ----------
    name:
        User-visible stream name.
    shape:
        Payload shape.
    dtype:
        NumPy dtype stored in the stream.
    size:
        Optional explicit size check. When provided, it must exactly match the
        size implied by ``shape`` and ``dtype``.
    gpu_device:
        Optional CUDA device identifier such as ``"cuda:0"``.
    cpu_mirror:
        Controls whether GPU-backed streams also maintain a CPU mirror.
        Defaults to ``True`` for CPU streams and ``False`` for GPU streams.
    auto_unlink:
        When ``True`` the stream is destroyed (not just closed) when used as
        a context manager.  Equivalent to calling :meth:`~SharedMemory.unlink`
        instead of :meth:`~SharedMemory.close` on ``__exit__``.  See also the
        :func:`stream` helper which sets this flag automatically.
    """
    shm = SharedMemory._create(
        name,
        shape=shape,
        dtype=dtype,
        size=size,
        gpu_device=gpu_device,
        cpu_mirror=cpu_mirror,
    )
    shm._auto_unlink = auto_unlink
    return shm


def open(
    name: str, *, gpu_device: str | int | bool | None = None
) -> SharedMemory:
    """Attach to an existing named shared-memory stream.

    For a GPU stream, ``gpu_device=None`` (the default) auto-attaches to the
    CUDA device recorded in metadata; an explicit ``gpu_device='cuda:N'`` must
    match the stored device.  Pass ``gpu_device=False`` to open a GPU stream
    *CPU-only* — the producer's CUDA tensor is not mapped and :meth:`read`
    returns the host mirror as a NumPy array.  This requires the stream to have
    been created with ``cpu_mirror=True`` and is the way to consume a GPU
    stream's mirror from a process that also has CUDA available (where the
    default would otherwise attach to the GPU).
    """
    return SharedMemory._open(name, gpu_device=gpu_device)


@contextmanager
def stream(
    name: str,
    *,
    shape: Sequence[int],
    dtype: Any = np.float32,
    size: int | None = None,
    gpu_device: str | int | None = None,
    cpu_mirror: bool | None = None,
):
    """Context manager that creates a stream and unlinks it on exit.

    Equivalent to ``pyshmem.create(..., auto_unlink=True)`` used as a ``with``
    block; intended for temporary streams in tests and one-shot pipelines.

    Example::

        with pyshmem.stream("my_stream", shape=(100,)) as shm:
            shm.write(data)
            result = shm.read()
        # stream is destroyed here
    """
    shm = create(
        name,
        shape=shape,
        dtype=dtype,
        size=size,
        gpu_device=gpu_device,
        cpu_mirror=cpu_mirror,
        auto_unlink=True,
    )
    try:
        yield shm
    finally:
        shm.unlink()
