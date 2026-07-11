# pyshmem

[PyPI](https://pypi.org/project/pyshmem/) | [Documentation](https://pyshmem.readthedocs.io/) | [Source](https://github.com/jacotay7/pyshmem) | [Issues](https://github.com/jacotay7/pyshmem/issues)

pyshmem provides named shared-memory streams for NumPy arrays and optional
CUDA-backed PyTorch pipelines. It is designed for applications that need
low-latency data exchange between OS processes — sensor pipelines, adaptive
optics control systems, and similar real-time workloads — without reinventing
locking, metadata, and CPU/GPU lifecycle management each time.

## Why pyshmem

- **One API for CPU and GPU.** NumPy arrays and CUDA tensors use the same
  `create`/`open`/`write`/`read` interface.
- **Cross-process write locking.** `portalocker` file locks plus a
  thread-reentrant `threading.RLock` give safe concurrent writes from multiple
  processes and threads.
- **Consistent snapshot reads.** An odd/even write-sequence counter lets readers
  take coherent snapshots without holding the write lock.
- **Explicit GPU modes.** Choose between a fast no-mirror GPU path and a
  CPU-mirrored compatibility path.
- **Persistent streams on POSIX.** Streams survive process exits and can be
  reattached by any process that knows the name.
- **Zero-allocation reading.** Pass a pre-allocated buffer to `read(out=...)` to
  avoid per-read heap allocations in tight loops.
- **Asyncio-compatible.** `read_new_async` yields to the event loop instead of
  blocking while waiting for a new write.

## Installation

Install from PyPI:

```bash
pip install pyshmem
```

For GPU streams add the `gpu` extra (requires a CUDA-capable PyTorch build):

```bash
pip install pyshmem[gpu]
```

For local development from a checkout:

```bash
pip install -e .[test]
```

## Quick Start

### CPU stream

```python
import numpy as np
import pyshmem

# Create in one process
writer = pyshmem.create("demo_frame", shape=(480, 640), dtype=np.float32)

# Attach in another process (or the same one)
reader = pyshmem.open("demo_frame")

writer.write(np.ones((480, 640), dtype=np.float32))

frame = reader.read()                   # latest completed write
next_frame = reader.read_new(timeout=1.0)  # block until a new write arrives
```

### GPU stream

```python
import numpy as np
import pyshmem

writer = pyshmem.create(
    "demo_cuda",
    shape=(1024, 1024),
    dtype=np.float32,
    gpu_device="cuda:0",
)
writer.write(np.ones((1024, 1024), dtype=np.float32))

reader = pyshmem.open("demo_cuda", gpu_device="cuda:0")
tensor = reader.read()   # returns a torch.Tensor on cuda:0
```

### Temporary stream with automatic cleanup

```python
import numpy as np
import pyshmem

with pyshmem.stream("scratch", shape=(256,)) as shm:
    shm.write(np.zeros(256, dtype=np.float32))
    result = shm.read()
# stream is destroyed when the block exits
```

## API Reference

### Top-level functions

| Function / Constant | Description |
|---|---|
| `pyshmem.create(name, *, shape, dtype, gpu_device, cpu_mirror, auto_unlink)` | Create a new named stream |
| `pyshmem.open(name, *, gpu_device)` | Attach to an existing stream |
| `pyshmem.unlink(name)` | Destroy a stream by name |
| `pyshmem.stream(name, *, shape, dtype, gpu_device, cpu_mirror)` | Context manager — creates and auto-unlinks |
| `pyshmem.list_streams()` | List all existing stream segment identifiers (Linux) |
| `pyshmem.gpu_available()` | `True` when CUDA-backed streams are usable |
| `pyshmem.GPU_SUPPORTED_DTYPES` | `frozenset` of NumPy dtypes accepted by `gpu_device=` |

### `SharedMemory` attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | User-visible stream name |
| `shape` | `tuple[int, ...]` | Payload shape |
| `dtype` | `np.dtype` | Payload dtype |
| `size` | `int` | Payload size in bytes |
| `gpu_device` | `str \| None` | Attached CUDA device string, or `None` |
| `gpu_enabled` | `bool` | Whether the stream was created with a GPU device |
| `cpu_mirror` | `bool` | Whether the CPU mirror is enabled |
| `owner` | `bool` | `True` if this handle created the stream |
| `count` | `int` | Number of completed writes |
| `write_time` | `float` | UNIX timestamp of the most recent write |
| `write_sequence` | `int` | Internal write sequence: positive odd = writing, non-negative even = stable, negative = incomplete write |

### `SharedMemory` methods

**Reading**

| Method | Description |
|---|---|
| `read(*, safe=True, poll_interval=1e-6, out=None, timeout=None)` | Return the current payload. `out` accepts a pre-allocated NumPy array (CPU only); `timeout` bounds an in-progress write. |
| `read_new(*, timeout=None, safe=True, poll_interval=1e-5)` | Block until a new write, then return the payload. |
| `await read_new_async(*, timeout=None, safe=True, poll_interval=1e-5)` | Asyncio-safe variant of `read_new`; yields to the event loop. |

**Writing**

| Method | Description |
|---|---|
| `write(value)` | Write a full payload (acquires the lock internally). |
| `write_locked(value)` | Write without re-acquiring the lock. Requires an active `with shm.locked()` block. |
| `clear()` | Zero the payload and record a write. |

**Locking**

| Method | Description |
|---|---|
| `acquire(*, timeout=None, poll_interval=1e-3)` | Acquire the cross-process write lock. Re-entrant within the current thread. |
| `release()` | Release one level of the re-entrant lock. |
| `locked(*, timeout=None, poll_interval=1e-3)` | Context manager around `acquire`/`release`. |

**Inspection**

| Method | Description |
|---|---|
| `describe()` | Return a human-readable multi-line summary of all metadata. |
| `to_config()` | Return a plain dict with `name`, `shape`, `dtype`, `gpu_device`, `cpu_mirror`. |
| `SharedMemory.create_from_config(config)` | Class method — create a new stream from a config dict. |

**Lifecycle**

| Method | Description |
|---|---|
| `close()` | Detach this local handle. The stream persists. |
| `unlink()` | Destroy the underlying stream entirely. |
| `delete()` | Alias for `unlink()`. |

## Feature Examples

### Zero-allocation reads

Pass a pre-allocated buffer to avoid heap allocations in hot loops:

```python
import numpy as np
import pyshmem

shm = pyshmem.open("stream")
buf = np.empty(shm.shape, dtype=shm.dtype)

while True:
    shm.read(out=buf)   # writes into buf directly; no new array allocated
    process(buf)
```

### Explicit locking and `write_locked`

Acquire the lock once to perform multiple operations atomically:

```python
with shm.locked():
    raw = shm.read(safe=False)    # zero-copy view into backing storage
    shm.write_locked(transform(raw))  # write back without re-locking
```

### Asyncio integration

```python
import asyncio
import pyshmem

async def consumer(name: str):
    shm = pyshmem.open(name)
    while True:
        frame = await shm.read_new_async(timeout=5.0)
        await process(frame)
```

### Stream introspection

```python
shm = pyshmem.open("my_stream")
print(shm.describe())
# name:         my_stream
# shape:        (480, 640)
# dtype:        float32
# size:         1228800 bytes
# gpu_enabled:  False
# gpu_device:   None
# cpu_mirror:   True
# count:        42
# write_time:   1748725312.4
# write_seq:    84
# owner:        False
```

### Config round-trip

```python
# Export the stream configuration
cfg = shm.to_config()
# {'name': 'my_stream', 'shape': [480, 640], 'dtype': 'float32',
#  'gpu_device': None, 'cpu_mirror': True}

# Recreate an identically-configured stream
shm2 = pyshmem.SharedMemory.create_from_config(cfg)
```

### Checking GPU dtype support

```python
import numpy as np
import pyshmem

np.dtype("float32") in pyshmem.GPU_SUPPORTED_DTYPES   # True
np.dtype("uint32") in pyshmem.GPU_SUPPORTED_DTYPES    # False (no PyTorch equivalent)
```

## Command-line Interface

pyshmem ships a `pyshmem` CLI for stream management on POSIX systems.

```bash
# List all live pyshmem stream segment identifiers
pyshmem list

# Destroy one or more streams by their user-visible names
pyshmem unlink my_stream
pyshmem unlink stream_a stream_b stream_c
```

## Reading Modes

`read(safe=True)` — default. Returns a consistent snapshot of the latest
completed write. Internally polls the write-sequence counter until it is even
(stable), copies the payload, then verifies the sequence did not change mid-copy.

`read(safe=False)` — zero-copy view into the live backing storage. Requires the
caller to hold the stream lock first:

```python
with reader.locked():
    raw = reader.read(safe=False)
```

`read(out=buf)` — writes the snapshot into a pre-allocated NumPy array instead
of allocating a new one. CPU streams only; GPU handles reject `out`.

## Behavior Notes

- After `close()`, methods such as `read`, `write`, `acquire`, `describe`, and
  metadata access raise a `RuntimeError` that includes the stream name and
  suggests calling `pyshmem.open(...)`.
- After `unlink()`, the underlying segments are destroyed. Any other process
  still attached will see errors on subsequent operations.
- `read_new` and `read_new_async` skip count checks while a write is in
  progress (odd `write_sequence`), preventing them from returning a partial
  write.
- If a copy fails or its writer process exits mid-write, reads raise
  `pyshmem.InconsistentStreamError` instead of polling forever. A subsequent
  successful full write repairs the stream.
- Locks are cross-process (`portalocker` file locks) and thread-reentrant
  (`threading.RLock`). Lock files live in `/tmp/pyshmem-locks-<uid>/` by
  default; set `PYSHMEM_LOCK_DIR` to override.

## GPU Streams

GPU streams require a CUDA-capable PyTorch installation (`pip install pyshmem[gpu]`).

### Performance mode (default)

```python
shm = pyshmem.create("activations", shape=(4096,), dtype="float32",
                      gpu_device="cuda:0")
```

- `cpu_mirror=False` by default — no CPU copy on every write
- Fastest path for GPU-heavy workloads
- CPU-only handles can still inspect metadata and take locks, but `read()`
  raises unless the handle was opened with `gpu_device=`

### Compatibility mode

```python
shm = pyshmem.create("activations", shape=(4096,), dtype="float32",
                      gpu_device="cuda:0", cpu_mirror=True)
```

- Maintains a CPU mirror on every write
- Allows CPU-only readers and safe-snapshot semantics under concurrent writes
- Trades throughput for compatibility

### Supported GPU dtypes

Only dtypes with a direct PyTorch equivalent are accepted:

```python
pyshmem.GPU_SUPPORTED_DTYPES
# frozenset({float16, float32, float64, int8, int16, int32, int64, uint8})
```

`uint16`, `uint32`, and `uint64` are not supported for GPU streams; `create()`
raises `ValueError` at construction time.

### Opening a GPU stream in another process

By default, `open()` auto-attaches to the CUDA device stored in the stream:

```python
reader = pyshmem.open("activations")
tensor = reader.read()   # torch.Tensor on cuda:0
```

Passing `gpu_device="cuda:0"` explicitly is also supported and validates that
the requested device matches. Pass `gpu_device=False` to opt into a CPU-only
view of a stream created with `cpu_mirror=True`.

## Platform Notes

### Linux and macOS

POSIX platforms support persistent streams: a segment survives the creator
process exiting as long as at least one other process holds it open (or until
`unlink()` is called). GPU IPC has been tested on Linux.

The lock-free read/write consistency protocol is validated under CPython on
**x86-64** and **aarch64**; other CPU architectures are best-effort because
pyshmem inserts no explicit hardware memory barriers. See the
[shared-memory format docs](https://pyshmem.readthedocs.io/en/latest/format.html)
for the full memory model.

### Windows

Windows inherits a hard limitation from `multiprocessing.shared_memory`: the
operating system destroys the shared-memory segment when the last handle closes.

The following behaviors are unsupported on Windows:

- a segment outliving its creator when no other process still has it open
- `close()` followed by `pyshmem.open(...)` when that `close()` dropped the
  last live handle

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PYSHMEM_LOCK_DIR` | `/tmp/pyshmem-locks-<uid>/` | Directory for lock files. Override to isolate locks on shared servers. |

## Testing

```bash
pip install -e .[test]

# Run the CPU suite
pytest -m cpu

# Run the GPU suite (requires CUDA)
pip install -e .[test,gpu]
pytest -m gpu

# Benchmark smoke tests
pytest -m "cpu and benchmark" -q -s
pytest tests/test_benchmark.py -m "gpu and benchmark" -q -s
```

## Performance

The benchmark suite measures round-trip IO and matrix-vector multiply pipelines
that keep the matrix in shared memory.

### Measured Results

- OS: Linux 6.17.0-14-generic x86_64
- Python: 3.12.0
- NumPy: 2.2.6
- PyTorch: 2.10.0+cu128
- GPU: NVIDIA GeForce RTX 5090

All benchmarks use `float32` payloads with warmup iterations before timing.
IO throughput counts both `write` and `read` bytes per iteration. MVM GFLOP/s
uses $2n^2$ floating-point operations per matrix-vector multiply.

GPU results below reflect the default no-mirror (`cpu_mirror=False`) path.

#### IO vs Image Size

| Image size | Payload (MiB) | CPU roundtrip Hz | CPU IO (GB/s) | GPU roundtrip Hz | GPU IO (GB/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100×100 | 0.038 | 180 311 | 14.42 | 36 214 | 2.90 |
| 1000×1000 | 3.815 | 9 922 | 79.38 | 5 027 | 40.22 |
| 10000×10000 | 381.470 | 20 | 16.29 | 50 | 39.97 |

#### Shared-Memory MVM Pipeline

Host-upload GPU pipeline:

| Matrix size | Matrix (MiB) | CPU Hz | CPU GFLOP/s | GPU Hz | GPU GFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100×100 | 0.038 | 109 844 | 2.20 | 26 466 | 0.53 |
| 1000×1000 | 3.815 | 11 125 | 22.25 | 22 485 | 44.97 |
| 10000×10000 | 381.470 | 26 | 5.24 | 1 299 | 259.86 |

Fully device-resident GPU pipeline:

| Matrix size | Matrix (MiB) | GPU Hz | GPU GFLOP/s |
| --- | ---: | ---: | ---: |
| 100×100 | 0.038 | 30 241 | 0.60 |
| 1000×1000 | 3.815 | 26 734 | 53.47 |
| 10000×10000 | 381.470 | 1 322 | 264.33 |

**Interpretation:**

- Small matrices (`100×100`) are dominated by launch and synchronization
  overhead, where CPU outperforms GPU.
- Once the workload is large enough, the no-mirror GPU path pulls ahead
  decisively — the `1000×1000` and `10000×10000` cases outperform CPU
  by a wide margin.
- Keeping vector generation on GPU further reduces the overhead for the
  fully device-resident pipeline.
