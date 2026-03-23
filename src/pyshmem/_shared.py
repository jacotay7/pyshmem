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

import hashlib
import builtins
from contextlib import contextmanager
import os
import pickle
import sys
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
TORCH_DTYPE_MAP = {
    np.dtype(np.float16): getattr(torch, "float16", None)
    if torch is not None
    else None,
    np.dtype(np.float32): getattr(torch, "float32", None)
    if torch is not None
    else None,
    np.dtype(np.float64): getattr(torch, "float64", None)
    if torch is not None
    else None,
    np.dtype(np.int8): getattr(torch, "int8", None)
    if torch is not None
    else None,
    np.dtype(np.int16): getattr(torch, "int16", None)
    if torch is not None
    else None,
    np.dtype(np.int32): getattr(torch, "int32", None)
    if torch is not None
    else None,
    np.dtype(np.int64): getattr(torch, "int64", None)
    if torch is not None
    else None,
    np.dtype(np.uint8): getattr(torch, "uint8", None)
    if torch is not None
    else None,
}

METADATA_VERSION = 2
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

_THREAD_LOCK_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, "_SharedLockState"] = {}
_LOCAL_GPU_TENSORS: dict[str, weakref.ReferenceType[Any]] = {}


class _SharedLockState:
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.thread_lock = threading.RLock()
        self.file_handle = builtins.open(path, "a+b")
        self.owner_thread_id: int | None = None
        self.depth = 0


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


def _lock_path(name: str) -> str:
    directory = os.path.join(tempfile.gettempdir(), "pyshmem-locks")
    return os.path.join(directory, f"{_segment_base_name(name)}.lock")


def _lock_state(name: str) -> _SharedLockState:
    path = _lock_path(name)
    with _THREAD_LOCK_GUARD:
        state = _THREAD_LOCKS.get(path)
        if state is None:
            state = _SharedLockState(path)
            _THREAD_LOCKS[path] = state
    return state


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
    if os.name == "nt" or sys.platform == "darwin":
        return
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass


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


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = [1] * len(shape)
    running = 1
    for index in range(len(shape) - 1, -1, -1):
        stride[index] = running
        running *= shape[index]
    return tuple(stride)


def _normalize_size(
    shape: tuple[int, ...], dtype: np.dtype, size: int | None
) -> int:
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if size is None:
        return expected
    if int(size) != expected:
        message = (
            "size does not match shape and dtype: "
            f"expected {expected}, got {size}"
        )
        raise ValueError(
            message
        )
    return expected


def _dtype_to_code(dtype: np.dtype) -> int:
    return DTYPE_TO_CODE[np.dtype(dtype)]


def _code_to_dtype(code: float) -> np.dtype:
    index = int(code)
    if index < 0 or index >= len(DTYPE_TABLE):
        raise ValueError(f"invalid dtype code in metadata: {code}")
    return DTYPE_TABLE[index]


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
        raise ValueError(
            f"dtype {dtype} is not supported for GPU shared memory"
        )
    return torch_dtype


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


def unlink(name: str) -> None:
    _LOCAL_GPU_TENSORS.pop(name, None)
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
    try:
        os.remove(_lock_path(name))
    except FileNotFoundError:
        pass
    except OSError:
        pass


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
        self._metadata = np.ndarray(
            (METADATA_SIZE,), dtype=np.float64, buffer=self._metadata_shm.buf
        )
        self._gpu_tensor = gpu_tensor
        self._torch_dtype = torch_dtype
        self._last_seen_count = int(self._metadata[METADATA_INDEX_COUNT])
        self._lock_state = _lock_state(name)
        self._closed = False

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

    def _wait_for_stable_writer(self, poll_interval: float) -> int:
        while True:
            sequence = self.write_sequence
            if sequence % 2 == 0:
                return sequence
            time.sleep(poll_interval)

    def _finish_write(self) -> None:
        self._metadata[METADATA_INDEX_COUNT] = self.count + 1
        self._metadata[METADATA_INDEX_WRITE_TIME] = time.time()
        self._metadata[METADATA_INDEX_WRITE_SEQUENCE] += 1

    def _mark_write_started(self) -> None:
        self._metadata[METADATA_INDEX_WRITE_SEQUENCE] += 1

    def _lock_metadata_on_acquire(self) -> None:
        self._metadata[METADATA_INDEX_LOCK_OWNER_PID] = os.getpid()
        self._metadata[METADATA_INDEX_LOCK_DEPTH] = self._lock_state.depth

    def _lock_metadata_on_release(self) -> None:
        if self._lock_state.depth == 0:
            self._metadata[METADATA_INDEX_LOCK_OWNER_PID] = 0
            self._metadata[METADATA_INDEX_LOCK_DEPTH] = 0
            return
        self._metadata[METADATA_INDEX_LOCK_DEPTH] = self._lock_state.depth

    def _read_consistent_cpu(self, poll_interval: float):
        while True:
            start_sequence = self._wait_for_stable_writer(poll_interval)
            result = np.copy(self._array)
            end_sequence = self.write_sequence
            if start_sequence == end_sequence:
                self._last_seen_count = self.count
                return result

    def _read_consistent_gpu(self, poll_interval: float):
        if self.cpu_mirror:
            while True:
                start_sequence = self._wait_for_stable_writer(poll_interval)
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

        result = self._gpu_tensor.clone()
        torch.cuda.synchronize(device=self.gpu_device)
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
        self._lock_state.thread_lock.acquire()
        thread_id = threading.get_ident()
        if self._lock_state.owner_thread_id == thread_id:
            self._lock_state.depth += 1
            self._lock_metadata_on_acquire()
            return

        try:
            _acquire_file_lock(
                self._lock_state.file_handle,
                timeout=timeout,
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
                size=METADATA_SIZE * np.dtype(np.float64).itemsize,
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
            metadata = np.ndarray(
                (METADATA_SIZE,), dtype=np.float64, buffer=metadata_shm.buf
            )
            metadata.fill(0)

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
        gpu_device: str | int | None = None,
    ) -> "SharedMemory":
        try:
            metadata_shm = shared_memory.SharedMemory(
                name=_metadata_name(name)
            )
        except FileNotFoundError as exc:
            raise _missing_name_error(name) from exc
        _unregister(metadata_shm)
        metadata = np.ndarray(
            (METADATA_SIZE,), dtype=np.float64, buffer=metadata_shm.buf
        )
        if int(metadata[METADATA_INDEX_VERSION]) != METADATA_VERSION:
            metadata_shm.close()
            raise ValueError(
                f"{name!r} does not contain a supported pyshmem metadata block"
            )

        dtype = _code_to_dtype(metadata[METADATA_INDEX_DTYPE])
        ndim = int(metadata[METADATA_INDEX_NDIM])
        shape = tuple(
            int(metadata[METADATA_INDEX_SHAPE_START + index])
            for index in range(ndim)
        )
        size = int(metadata[METADATA_INDEX_SIZE])
        gpu_enabled = bool(int(metadata[METADATA_INDEX_GPU_ENABLED]))
        device_index = int(metadata[METADATA_INDEX_DEVICE_INDEX])
        creator_pid = int(metadata[METADATA_INDEX_CREATOR_PID])
        cpu_mirror_enabled = bool(
            int(metadata[METADATA_INDEX_CPU_MIRROR_ENABLED])
        )

        try:
            data_shm = shared_memory.SharedMemory(name=_data_name(name))
        except FileNotFoundError as exc:
            metadata_shm.close()
            raise _missing_name_error(name) from exc
        _unregister(data_shm)

        resolved_gpu = None
        gpu_tensor = None
        gpu_handle_shm = None
        torch_dtype = None
        if gpu_enabled and gpu_device is not None:
            resolved_gpu = _normalize_gpu_device(gpu_device)
            if device_index < 0:
                metadata_shm.close()
                data_shm.close()
                raise ValueError(
                    f"{name!r} does not advertise a valid CUDA device"
                )
            if resolved_gpu.index != device_index:
                metadata_shm.close()
                data_shm.close()
                message = (
                    f"requested GPU device {resolved_gpu} "
                    "does not match stored device "
                    f"cuda:{device_index}"
                )
                raise ValueError(message)
            torch_dtype = _torch_dtype_for(dtype)
            gpu_tensor, gpu_handle_shm = _open_gpu_tensor_from_handle(
                name=name,
                shape=shape,
                torch_dtype=torch_dtype,
                creator_pid=creator_pid,
            )

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
        self._closed = True

    def unlink(self) -> None:
        """Destroy the underlying named shared-memory stream."""
        self.close()
        unlink(self.name)

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
            if self._gpu_tensor is not None:
                self._gpu_tensor.zero_()
            if self.cpu_mirror:
                self._array.fill(0)
            if self._gpu_tensor is not None:
                torch.cuda.synchronize(device=self.gpu_device)
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
                    f"expected shape {self.shape}, "
                    f"got {tuple(tensor.shape)}"
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
                    f"expected shape {self.shape}, "
                    f"got {tuple(array.shape)}"
                )
                raise ValueError(message)

        with self.locked():
            self._mark_write_started()
            if tensor is not None:
                self._gpu_tensor.copy_(tensor)
                if self.cpu_mirror:
                    np.copyto(self._array, tensor.detach().cpu().numpy())
                torch.cuda.synchronize(device=self.gpu_device)
            else:
                np.copyto(self._array, array)
            self._finish_write()

    def read(self, *, safe: bool = True, poll_interval: float = 1e-6):
        """Read the current payload from the stream.

        When ``safe`` is ``True``, the method returns a consistent snapshot of
        the latest completed write. When ``safe`` is ``False``, the caller must
        already own the stream lock via :meth:`locked` or :meth:`acquire`.
        """
        self._ensure_open("read from")
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
                    "GPU shared memory was created without cpu_mirror=True; "
                    "reopen it with "
                    f"pyshmem.open({self.name!r}, gpu_device='cuda:N')"
                )
            return self._array

        if self._gpu_tensor is not None:
            return self._read_consistent_gpu(poll_interval)
        if self.gpu_enabled and not self.cpu_mirror:
            raise RuntimeError(
                "GPU shared memory was created without cpu_mirror=True; "
                "reopen it with "
                f"pyshmem.open({self.name!r}, gpu_device='cuda:N')"
            )
        return self._read_consistent_cpu(poll_interval)

    def read_new(
        self,
        *,
        timeout: float | None = None,
        safe: bool = True,
        poll_interval: float = 1e-5,
    ):
        """Block until a new write arrives, then return its payload."""
        self._ensure_open("read from")
        baseline = self.count
        start = time.monotonic()
        while self.count == baseline:
            if timeout is not None and (time.monotonic() - start) >= float(
                timeout
            ):
                raise TimeoutError(
                    f"timed out waiting for a new write on {self.name!r}"
                )
            time.sleep(poll_interval)
        return self.read(safe=safe)

    def __enter__(self) -> "SharedMemory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _create_gpu_tensor_and_handle(
    *, name: str, shape: tuple[int, ...], torch_dtype, gpu_device: Any
):
    gpu_tensor = torch.empty(shape, dtype=torch_dtype, device=gpu_device)
    storage = gpu_tensor._typed_storage()
    handle_payload = pickle.dumps(storage._share_cuda_(), protocol=4)
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
    if creator_pid == os.getpid():
        gpu_tensor = _get_cached_gpu_tensor(name)
        if gpu_tensor is None:
            raise RuntimeError(
                "cannot reopen GPU shared memory in the creator process "
                "after all local GPU handles have been released"
            )
        return gpu_tensor, None

    handle_shm = shared_memory.SharedMemory(name=_gpu_handle_name(name))
    _unregister(handle_shm)
    (
        device_index,
        handle_bytes,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = pickle.loads(bytes(handle_shm.buf))

    torch.cuda._lazy_init()
    storage = torch.UntypedStorage._new_shared_cuda(
        device_index,
        handle_bytes,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    )
    typed_storage = torch.storage.TypedStorage(
        wrap_storage=storage,
        dtype=torch_dtype,
        _internal=True,
    )
    tensor = torch._utils._rebuild_tensor(
        typed_storage,
        0,
        shape,
        _contiguous_stride(shape),
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
    """
    return SharedMemory._create(
        name,
        shape=shape,
        dtype=dtype,
        size=size,
        gpu_device=gpu_device,
        cpu_mirror=cpu_mirror,
    )


def open(name: str, *, gpu_device: str | int | None = None) -> SharedMemory:
    """Attach to an existing named shared-memory stream."""
    return SharedMemory._open(name, gpu_device=gpu_device)
