# pyshmem — Developer Reference

## Purpose

pyshmem is a unified CPU/GPU shared-memory library for Python. It wraps `multiprocessing.shared_memory` with a clean API for named, typed, array-shaped streams that work identically on CPU (NumPy) and GPU (CUDA/PyTorch). The library is intentionally minimal — the entire implementation lives in `src/pyshmem/_shared.py`.

## Core Concepts

### Streams
A **stream** is a named slot in shared memory with a fixed shape, dtype, and storage backend. It is the only primitive this library exposes. Streams persist across process exits (on POSIX) and can be attached by any process that knows the name.

### Internal segments
Each logical stream `name` maps to up to three POSIX shared-memory segments:
- **data segment** — the array payload (`ps_<sha1hash>`)
- **metadata segment** — a `float64[32]` array with shape, dtype, counts, lock state, etc. (`ps_<sha1hash>_meta`)
- **GPU handle segment** — serialized `_share_cuda_()` handle for cross-process GPU tensor reconstruction (`ps_<sha1hash>_gpu`)

Names are hashed (SHA-1, first 14 chars) to stay under the POSIX segment name limit while remaining collision-resistant.

### Metadata layout (METADATA_INDEX_* constants)
The metadata array stores (in order): version, write count, dtype code, ndim, size, gpu_enabled flag, device index, creator PID, write timestamp, write sequence number, lock owner PID, lock depth, cpu_mirror flag, then shape dimensions starting at index 13. `METADATA_SIZE = 32` (total slots).

### Locking model
- Cross-process: `portalocker` file locks in `/tmp/pyshmem-locks-<uid>/` (or `$PYSHMEM_LOCK_DIR`)
- Per-thread: `threading.RLock` (re-entrant within a thread)
- The `_lock_state(name)` function returns/creates a `_SharedLockState` that is shared across all `SharedMemory` handles opened for the same name in the same process.
- Crash recovery: `portalocker` uses OS-level file locks that are released automatically on process exit.

### Write sequence protocol
Writers bracket payloads with odd/even sequence numbers:
1. Increment `WRITE_SEQUENCE` (now odd → write in progress)
2. Copy payload
3. Increment `WRITE_SEQUENCE` again (now even → write complete)

Readers poll until `WRITE_SEQUENCE` is even (stable), snapshot the data, then verify the sequence didn't change. This lock-free consistency mechanism is in `_read_consistent_cpu()` and `_read_consistent_gpu()`.

### GPU IPC
GPU streams use `torch.UntypedStorage._share_cuda_()` / `_new_shared_cuda()` for cross-process tensor sharing. The serialized CUDA IPC handle is stored in the GPU handle segment. A per-process weakref cache (`_LOCAL_GPU_TENSORS`) avoids re-importing handles within the creator process.

## Public API

```python
import pyshmem

# Create
shm = pyshmem.create("my_stream", shape=(100,), dtype="float32")
shm = pyshmem.create("my_gpu_stream", shape=(100,), dtype="float32",
                      gpu_device="cuda:0", cpu_mirror=False)
# auto-unlink on context exit
shm = pyshmem.create("tmp", shape=(10,), auto_unlink=True)
with pyshmem.stream("tmp2", shape=(10,)) as shm:   # always auto-unlinks
    ...

# Attach
shm = pyshmem.open("my_stream")
shm = pyshmem.open("my_gpu_stream", gpu_device="cuda:0")

# Discover
pyshmem.list_streams()    # returns sorted list of ps_* segment base names

# Use
shm.write(array)          # CPU: numpy array; GPU: numpy or CUDA tensor
data = shm.read()         # returns np.ndarray (CPU) or torch.Tensor (GPU)
data = shm.read(out=buf)  # zero-alloc: writes into pre-allocated buffer
data = shm.read_new(timeout=1.0)         # blocks until a new write arrives
data = await shm.read_new_async(timeout=1.0)  # asyncio-safe variant

# Locking (explicit)
shm.acquire(timeout=0.5)
shm.read(safe=False)           # zero-copy view — only valid inside lock
shm.write_locked(value)        # write without re-acquiring lock (shmpipeline fast path)
shm.release()

# Context manager
with shm.locked():
    shm.read(safe=False)
    shm.write_locked(new_value)

# Metadata
shm.describe()             # human-readable summary string
cfg = shm.to_config()      # dict: name/shape/dtype/gpu_device/cpu_mirror
shm2 = pyshmem.SharedMemory.create_from_config(cfg)

# Lifecycle
shm.close()               # detach this handle; stream persists
shm.unlink()              # destroy the stream entirely
pyshmem.unlink("my_stream")
```

## Constants

| Name | Description |
|------|-------------|
| `GPU_SUPPORTED_DTYPES` | `frozenset` of NumPy dtypes that can be used with `gpu_device=` |

## CLI

```bash
pyshmem list                     # list all existing pyshmem stream segments
pyshmem unlink my_stream         # destroy a stream by user-visible name
pyshmem unlink stream_a stream_b # destroy multiple streams
```

## Key Implementation Details

- **`create()`** calls `SharedMemory._create()` which creates segments atomically (with cleanup on failure).
- **`open()`** calls `SharedMemory._open()` which reads metadata to reconstruct shape/dtype without needing the caller to know them.
- **GPU streams without `cpu_mirror`**: only processes that open with `gpu_device=` can `read()` or `write()` to the tensor. Metadata and locking still work without a GPU attachment.
- **Resource tracker suppression**: `_unregister()` removes segments from Python's `resource_tracker` so child process exits don't spuriously warn about leaked shared memory.
- **Platform**: Full Linux support. macOS: GPU IPC is not tested. Windows: uses named shared memory (no POSIX shm_unlink).

## Supported dtypes

CPU streams: `int8 int16 int32 int64 uint8 uint16 uint32 uint64 float16 float32 float64`

GPU streams (torch-mapped): `int8 int16 int32 int64 uint8 float16 float32 float64`
Note: `uint16 uint32 uint64` are **not** supported for GPU (no PyTorch equivalent); `create()` raises `ValueError` at construction time.

## Project Structure

```
src/pyshmem/
  __init__.py     # public surface (create, open, unlink, stream, list_streams, gpu_available, GPU_SUPPORTED_DTYPES)
  _shared.py      # entire implementation
  _cli.py         # CLI entry point (pyshmem list / unlink)
tests/
  conftest.py     # shm_name fixture (auto-cleanup via uuid)
  test_cpu_api.py # CPU stream tests (pytest.mark.cpu)
  test_gpu_api.py # GPU stream tests (pytest.mark.gpu, skipif no CUDA)
  test_benchmark.py
```

## Running Tests

```bash
# All tests (CPU + GPU if CUDA available)
python -m pytest tests/ -q

# CPU only
python -m pytest tests/ -m cpu -q

# GPU only (requires CUDA)
python -m pytest tests/ -m gpu -q
```

## Lint

After any significant code change, run lint and auto-format before committing:

```bash
ruff check .           # must be clean (zero errors)
ruff format .          # auto-fixes formatting; re-run check after
```

The CI lint job enforces both `ruff check` (E/W/F rules, line-length 79) and `ruff format --check`. Fix all reported issues before pushing.

## Test Coverage Policy

**Every code change must be accompanied by tests.** This is non-negotiable.

- **New feature** → add at least one happy-path test and one error/edge-case test.
- **Bug fix** → add a regression test that fails on the unfixed code and passes after.
- **Correctness fix** (race condition, leak, etc.) → add a targeted test using
  `monkeypatch` or `threading` to reproduce the scenario.
- **Public API addition** → test all new parameters and return values, including
  `closed` state checks if the method calls `_ensure_open`.

Test placement:
- CPU-only behaviour → `tests/test_cpu_api.py` (marked `pytest.mark.cpu`)
- GPU-specific behaviour → `tests/test_gpu_api.py` (marked `pytest.mark.gpu`,
  `@pytest.mark.skipif(not CUDA_AVAILABLE, ...)`)
- Use the `shm_name` fixture for all stream names (guarantees cleanup on teardown).

The CPU suite must pass at all times:
```bash
python -m pytest tests/test_cpu_api.py -q   # must be green before any merge
```

## Package Info

- Package name on PyPI: `pyshmem` (v1.0.3)
- License: GPL-3.0-only
- Required deps: `numpy>=1.26,<3`, `portalocker>=3.1`
- Optional deps: `torch>=2.2` (GPU support)
- Python: 3.9–3.13
- GitHub: `https://github.com/jacotay7/pyshmem`

## Coupling with shmpipeline

`shmpipeline` intentionally accesses several private `pyshmem` attributes for performance:
- `_mark_write_started()`, `_finish_write()` — bracket writes without acquiring locks (used when the worker already holds the lock)
- `_array` — zero-copy view into the CPU data segment
- `_LOCAL_GPU_TENSORS`, `_data_name`, `_metadata_name`, `_gpu_handle_name`, `_lock_path` — used by `shmpipeline.shm_cleanup` for direct POSIX unlink without going through the resource tracker

Any changes to these private names must also update `shmpipeline/src/shmpipeline/runtime.py` and `shmpipeline/src/shmpipeline/shm_cleanup.py`.
