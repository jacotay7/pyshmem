# pyshmem

[![PyPI](https://img.shields.io/pypi/v/pyshmem)](https://pypi.org/project/pyshmem/)
[![Python](https://img.shields.io/pypi/pyversions/pyshmem)](https://pypi.org/project/pyshmem/)
[![CI](https://github.com/jacotay7/pyshmem/actions/workflows/ci.yml/badge.svg)](https://github.com/jacotay7/pyshmem/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/pyshmem/badge/?version=latest)](https://pyshmem.readthedocs.io/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

pyshmem is a Python library for low-latency, cross-process exchange of fixed
shape NumPy arrays and CUDA-backed PyTorch tensors. CPU and GPU streams share
the same small `create` / `open` / `write` / `read` API.

A pyshmem stream is a **capacity-one latest-value exchange**, not a queue. Each
write replaces the previous payload; readers get a consistent snapshot and can
inspect missed-publication counters when producers run faster than consumers.
It supports Linux and macOS; CUDA IPC is Linux-only. Windows is unsupported.

[Documentation](https://pyshmem.readthedocs.io/) ·
[API reference](https://pyshmem.readthedocs.io/en/latest/api.html) ·
[Source](https://github.com/jacotay7/pyshmem) ·
[Issues](https://github.com/jacotay7/pyshmem/issues) ·
[Changelog](CHANGELOG.md)

## Quick start

Create and publish from one process:

```python
import numpy as np
import pyshmem

writer = pyshmem.create("frames", shape=(480, 640), dtype=np.float32)
with writer.write_view() as frame:
    frame[...] = 1.0  # zero-copy, exception-safe publish
```

Attach and read from another:

```python
import pyshmem

reader = pyshmem.open("frames")
frame = reader.read()  # latest consistent snapshot
next_frame = reader.read_new(timeout=1)  # wait for the next publication
next_frame = reader.read_after(reader.last_read_count, timeout=1)
print(reader.missed_writes)

# Keep a payload and its generation metadata inseparable when it matters.
publication = reader.read_publication()
print(publication.frame_id, publication.count, publication.payload)

reader.close()
```

Use the same API for CUDA by creating with `gpu_device="cuda:0"`; reads return
a CUDA `torch.Tensor` when attached to the GPU. Destroy persistent streams with
`writer.unlink()` when they are no longer needed.

See the [quick start](https://pyshmem.readthedocs.io/en/latest/quickstart.html),
[usage guide](https://pyshmem.readthedocs.io/en/latest/usage.html), and
[GPU guide](https://pyshmem.readthedocs.io/en/latest/gpu.html) for lifecycle,
locking, asyncio, CPU mirrors, failure recovery, and CLI examples.

## Installation

CPU support:

```bash
pip install pyshmem
```

CUDA support (installs the PyTorch dependency):

```bash
pip install "pyshmem[gpu]"
```

See the [installation guide](https://pyshmem.readthedocs.io/en/latest/installation.html)
for supported Python versions, platform requirements, development setup, and
installation verification.

## Performance

The repository includes both single-process microbenchmarks and a calibrated,
spawned-process request/acknowledgement benchmark with an unsafe raw shared
memory lower-bound comparison.

On the primary Linux development machine (Python 3.12, NumPy 2.2.6, PyTorch
2.10, RTX 5090), the spawned-process 64 KiB benchmark measured:

| Implementation | Round trips/s | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| pyshmem (CPU) | 13,988 | 60.26 µs | 111.69 µs | 115.70 µs |
| pyshmem (GPU IPC) | 4,872 | 189.25 µs | 239.00 µs | 240.98 µs |
| Raw shared memory polling | 16,492 | 57.58 µs | 104.45 µs | 108.95 µs |

The raw baseline omits pyshmem's locking, metadata validation, discovery, and
consistent snapshots. The GPU row is a separate spawned process mapping the
producer's CUDA tensor over torch IPC and reading a consistent device snapshot
each round trip. Results are machine-specific; run the harness on the target
deployment host (add `--gpu` for the CUDA baseline):

```bash
python benchmarks/benchmark_ipc.py \
  --payload-bytes 65536 --minimum-seconds 1 --repeats 5 --gpu
```

See the [benchmark documentation](https://pyshmem.readthedocs.io/en/latest/benchmarks.html)
and the [versioned result](benchmarks/results/rtx5090-linux-2026-07-10.json) for
methodology, CPU/GPU measurements, and limitations.

## License and contact

pyshmem is licensed under [GPL-3.0-only](LICENSE). Applications that distribute
pyshmem or derivative work should evaluate the GPL's obligations.

Use [GitHub issues](https://github.com/jacotay7/pyshmem/issues) for bugs and
feature requests. See [CONTRIBUTING.md](CONTRIBUTING.md) for development,
[SUPPORT.md](SUPPORT.md) for compatibility and support scope, and
[SECURITY.md](SECURITY.md) for private vulnerability reporting.
