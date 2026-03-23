# pyshmem

[PyPI](https://pypi.org/project/pyshare/) | [Documentation](https://pyshare.readthedocs.io/) | [Source](https://github.com/jacotay7/pyshare) | [Issues](https://github.com/jacotay7/pyshare/issues)

pyshare provides named shared-memory streams for NumPy arrays and optional
CUDA-backed PyTorch pipelines.

It is designed for applications that need a small, predictable API for moving
numeric payloads between processes without rebuilding the same locking,
metadata, and lifecycle rules around raw shared memory.

## Why pyshare

- one API for CPU NumPy buffers and CUDA-backed tensors
- cross-process write locking with explicit lock ownership
- safe snapshot reads for CPU streams
- explicit GPU performance mode or CPU-mirrored compatibility mode
- tested lifecycle and recovery behavior across supported platforms

## Installation

Install from PyPI:

```bash
pip install pyshare
```

Optional extras:

```bash
pip install pyshare[test]
pip install pyshare[gpu]
pip install pyshare[docs]
```

For local development from a checkout:

```bash
pip install -e .[test]
```

## Quick Start

### CPU stream

```python
import numpy as np
import pyshare

writer = pyshare.create("demo_frame", shape=(4, 4), dtype=np.float32)
reader = pyshare.open("demo_frame")

writer.write(np.ones((4, 4), dtype=np.float32))
frame = reader.read()
next_frame = reader.read_new(timeout=1.0)
```

### GPU stream

```python
import numpy as np
import pyshmem

writer = pyshare.create(
    "demo_cuda",
    shape=(4, 4),
    dtype=np.float32,
    gpu_device="cuda:0",
)
writer.write(np.ones((4, 4), dtype=np.float32))

reader = pyshare.open("demo_cuda", gpu_device="cuda:0")
frame = reader.read()
```

## Public API

- `pyshare.SharedMemory`
- `pyshare.create(name, *, shape, dtype=np.float32, size=None, gpu_device=None, cpu_mirror=None)`
- `pyshare.open(name, *, gpu_device=None)`
- `pyshare.unlink(name)`
- `pyshare.gpu_available()`

`SharedMemory` instances expose metadata, locking, lifecycle, and IO methods:

- `name`, `shape`, `dtype`, `size`, `gpu_device`, `cpu_mirror`, `owner`
- `count`, `write_time`, `write_sequence`
- `acquire(timeout=None, poll_interval=1e-3)`
- `release()`
- `locked(timeout=None, poll_interval=1e-3)`
- `write(value)`
- `read(safe=True, poll_interval=1e-6)`
- `read_new(timeout=None, safe=True, poll_interval=1e-5)`
- `clear()`
- `close()`
- `unlink()`
- `delete()`

## Behavior Notes

Writes are serialized with a cross-platform file lock backend.

- `read(safe=True)` returns a consistent snapshot of the most recent completed write
- `read(safe=False)` exposes the live backing storage and therefore requires `with shm.locked():`
- `close()` releases only the local handle
- `unlink()` destroys the underlying shared-memory stream

Closed handles are guarded explicitly. After `close()`, methods such as
`read`, `write`, `acquire`, `clear`, and metadata access raise a `RuntimeError`
that instructs the caller to reopen the stream.

Missing segments raise `FileNotFoundError` with a pyshare-specific message that
points the caller toward `pyshare.create(...)`.

## GPU Modes

GPU-backed streams have two deliberately different operating modes.

Performance mode:

- `pyshare.create(..., gpu_device="cuda:N")` defaults to `cpu_mirror=False`
- avoids CPU mirror maintenance on every write
- optimized for GPU-heavy pipelines where throughput matters most

Compatibility mode:

- `pyshare.create(..., gpu_device="cuda:N", cpu_mirror=True)` keeps the CPU mirror updated
- allows CPU-side payload reads and stronger safe-snapshot semantics under concurrent writes

Important attachment rule:

- pass `gpu_device="cuda:N"` to `pyshare.open(...)` whenever the caller needs a CUDA `torch.Tensor` view
- opening a GPU stream without `gpu_device` still allows metadata inspection and lock management, but payload reads require either a GPU attachment or `cpu_mirror=True`

## Platform Notes

### Windows limitation

Windows inherits a hard limitation from `multiprocessing.shared_memory`: the
operating system deletes a shared-memory block as soon as the last handle to it
is closed.

That means the following behaviors are unsupported on Windows:

- a segment outliving its creator when no other process still has it open
- `close()` followed by `pyshare.open(...)` when that `close()` dropped the final live handle

Those behaviors remain supported on POSIX platforms.

## Testing

Install test dependencies and run the CPU suite:

```bash
pip install -e .[test]
pytest -m cpu
```

Run the CUDA suite on a GPU machine:

```bash
pip install -e .[test,gpu]
pytest -m gpu
```

The repository also includes benchmark-marked tests:

```bash
pytest -m "cpu and benchmark" -q -s
pytest tests/test_benchmark.py -m "gpu and benchmark" -q -s
```

GitHub-hosted runners do not provide CUDA by default, so the CUDA workflow is
manual and targets either a self-hosted GPU runner or a larger GitHub runner
with CUDA support.

## Performance

The benchmark suite measures both raw shared-memory IO and matrix-vector
multiply pipelines that keep the matrix in shared memory.

Two GPU MVM shapes are covered:

- host-upload pipeline: the vector payload is created in NumPy and uploaded each iteration
- device-resident pipeline: the vector payload is produced directly on GPU each iteration

The CPU benchmark target remains 50 kHz for a `128x128` round trip. Hard
enforcement is opt-in because hosted CI is not a reliable performance lab.

### Measured Results

The following numbers were measured on this machine:

- OS: Linux 6.17.0-14-generic x86_64
- Python: 3.12.0
- NumPy: 2.2.6
- PyTorch: 2.10.0+cu128
- GPU: NVIDIA GeForce RTX 5090

Methodology:

- `float32` payloads throughout
- each benchmark case used warmup iterations before timing
- each timed case ran for at least 1.5 seconds to reduce one-off noise
- IO throughput is computed from `write` plus `read` bytes per iteration
- MVM throughput is reported both as pipeline rate and estimated GFLOP/s using $2n^2$ floating-point operations per matrix-vector multiply

Important interpretation note:

- GPU-backed segments now default to `cpu_mirror=False`
- the fast GPU path avoids CPU mirror maintenance unless the creator explicitly asks for it with `cpu_mirror=True`
- the stronger concurrent-read consistency contract is provided by the mirrored mode; the default no-mirror mode is optimized for throughput first
- the GPU numbers below therefore reflect the optimized no-mirror path, which is the intended performance configuration

#### IO vs Image Size

| Image size | Payload (MiB) | CPU roundtrip Hz | CPU IO (GB/s) | GPU roundtrip Hz | GPU IO (GB/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100x100 | 0.038 | 180311.2 | 14.42 | 36214.1 | 2.90 |
| 1000x1000 | 3.815 | 9922.1 | 79.38 | 5027.4 | 40.22 |
| 10000x10000 | 381.470 | 20.36 | 16.29 | 49.96 | 39.97 |

#### Shared-Memory MVM Pipeline

Host-upload GPU pipeline:

| Matrix size | Matrix payload (MiB) | CPU pipeline Hz | CPU GFLOP/s | GPU pipeline Hz | GPU GFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100x100 | 0.038 | 109844.4 | 2.20 | 26465.8 | 0.53 |
| 1000x1000 | 3.815 | 11124.9 | 22.25 | 22485.3 | 44.97 |
| 10000x10000 | 381.470 | 26.21 | 5.24 | 1299.3 | 259.86 |

Fully device-resident GPU pipeline:

| Matrix size | Matrix payload (MiB) | GPU pipeline Hz | GPU GFLOP/s |
| --- | ---: | ---: | ---: |
| 100x100 | 0.038 | 30240.6 | 0.60 |
| 1000x1000 | 3.815 | 26733.6 | 53.47 |
| 10000x10000 | 381.470 | 1321.6 | 264.33 |

The updated results show the intended behavior for real GPU workloads:

- tiny matrices like `100x100` are still dominated by launch and synchronization overhead, so CPU remains faster there
- once the workload is large enough to matter, the no-mirror GPU path pulls ahead decisively
- the `1000x1000` and `10000x10000` MVM cases now outperform the CPU equivalents by a wide margin on this machine
- keeping the vector generation on GPU improves the pipeline further, especially once the matrix is large enough for the math to dominate
