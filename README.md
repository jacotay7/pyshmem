# pyshmem

pyshmem is a small shared-memory library that presents one API for CPU-backed NumPy buffers and optional CUDA-backed PyTorch mirrors.

The initial scaffold is intentionally narrow. The goal is to lock down a simple public API and a test contract before expanding the implementation surface.

Current design goals:

- safe cross-thread and cross-process access
- Windows, macOS, and Linux compatibility
- high enough throughput for real-time control style workloads

## Installation

```bash
pip install -e .
```

For test dependencies:

```bash
pip install -e .[test]
```

For manual CUDA tests:

```bash
pip install -e .[test,gpu]
```

## API

```python
import numpy as np
import pyshmem

writer = pyshmem.create(
	"demo_frame",
	shape=(4, 4),
	dtype=np.float32,
	gpu_device=None,
)

writer.write(np.ones((4, 4), dtype=np.float32))

reader = pyshmem.open("demo_frame")
frame = reader.read()
next_frame = reader.read_new(timeout=1.0)
```

Current public entry points:

- `pyshmem.SharedMemory`
- `pyshmem.create(name, *, shape, dtype=np.float32, size=None, gpu_device=None)`
- `pyshmem.open(name, *, gpu_device=None)`
- `pyshmem.unlink(name)`
- `pyshmem.gpu_available()`

Returned objects expose:

- `name`
- `shape`
- `dtype`
- `size`
- `gpu_device`
- `owner`
- `count`
- `write_time`
- `write_sequence`
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

Writes use a cross-platform file lock backend and safe reads use a versioned
snapshot path. That means:

- writes are serialized across processes
- explicit lock ownership is available through `acquire`, `release`, and
	`locked`
- `read(safe=True)` returns a consistent snapshot without forcing every reader
	to take the OS lock
- `read(safe=False)` still requires `with shm.locked():` because raw shared
	views cannot be made safe otherwise

Lifecycle semantics are intentionally destructive:

- any attached handle may destroy the segment
- `x.unlink()` destroys the backing shared memory
- `x.delete()` is an alias for `x.unlink()`
- `x.clear()` resets the current payload to zeros and records a new write
- `x.close()` is non-destructive, idempotent, and only releases the local handle

Closed handles are guarded explicitly. After `x.close()`, operations such as
`read`, `write`, `acquire`, `clear`, and metadata access raise a pyshmem-level
`RuntimeError` that instructs the caller to reopen the segment with
`pyshmem.open(...)`.

Missing segments also raise a clearer pyshmem-level `FileNotFoundError`
explaining that the caller likely needs `pyshmem.create(...)` first.

## Testing

CPU tests are part of CI:

```bash
pytest -m cpu
```

The CPU suite covers:

- creating and writing a stream in one process, then opening it in another
	process and validating shape, dtype, size, and data
- explicit lock behavior across processes
- concurrent write safety for consistent reader snapshots
- closed-handle misuse protection
- recovery after a lock-holding process exits abruptly
- continued usability after the original creator process exits

CI runs lint and CPU tests on Linux, macOS, and Windows.

Benchmark smoke tests are run separately in CI with:

```bash
pytest -m "cpu and benchmark" -q -s
```

CUDA tests are kept in the repository but marked separately so they can be run on a GPU machine:

```bash
pytest -m gpu
```

The GPU tests are intended to validate that GPU-enabled streams return `torch.Tensor` objects and can be opened from a second process.

## Performance

The current benchmark target is 50 kHz for a 128x128 CPU array round trip.

The repository includes a benchmark-marked test that measures repeated
`write` plus `read` round trips for a 128x128 `float32` array. The benchmark is
run in CI as a smoke test and can be enforced locally with:

```bash
pyshmem_ENFORCE_BENCHMARK=1 pyshmem_TARGET_HZ=50000 pytest -m "cpu and benchmark" -q -s
```

Hosted CI runners are not reliable performance labs, so the benchmark smoke
test records the path and keeps the hard 50 kHz enforcement opt-in.
