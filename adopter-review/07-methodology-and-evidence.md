# Methodology and evidence

## Post-review remediation evidence

After implementing the first remediation batch, the same environment produced:

```text
full CPU/GPU suite: 129 passed in 11.74s
branch-aware coverage suite: 123 passed, 81% total coverage
1,200 unique stream lifecycle probe: fd 5 -> 6, lock_states 0
failed write: InconsistentStreamError at sequence -1
replacement write: stable sequence 4, payload [3.0, 4.0]
CPU 128x128 roundtrip: 167155.1 Hz
GPU 128x128 sequence-safe roundtrip: 27516.1 Hz
```

Ruff, warning-strict Sphinx, isolated package build, and `twine check` also
passed. The original pre-fix evidence remains below for before/after comparison.

## Scope and non-destructive precautions

The review inspected revision `517ab7c`, built the package and docs, ran all
checked-in tests on the configured CUDA environment, read the implementation and
public material, executed isolated probes with `review_*` UUID names, and
checked current official documentation for relevant alternatives.

The machine already contained unrelated `/dev/shm` objects. `pyshmem.purge()`
was deliberately **not** executed because its source shows that it would remove
all `ps_*` objects and global orphaned PyTorch CUDA IPC files. Each review stream
was individually unlinked.

No alternative package had to be installed: the most useful performance
baseline was the standard library/NumPy/PyTorch stack already used internally.

## Commands and outcomes

### Environment

```text
conda env: cuda312
Python: 3.12.0
NumPy: 2.2.6
PyTorch: 2.10.0+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 5090, 32607 MiB
Driver: 580.159.03
/dev/shm: 16 GiB, about 1% used before review
```

### Functional suite

```bash
python -m pytest tests -q -ra --durations=20
```

Result: 120 passed in 10.73 seconds. The process emitted a CUDA IPC producer
lifecycle warning after pytest completed.

### Coverage

```bash
python -m pytest tests -m 'not benchmark' \
  --cov=pyshmem --cov-report=term-missing -q
```

Result: 115 passed; 83% total statement coverage (`_shared.py` 83%, CLI 74%).

### Static/package/docs checks

```bash
ruff check .
ruff format --check .
sphinx-build -W -b html docs /tmp/pyshmem-docs-review
python -m build --outdir /tmp/pyshmem-dist-review
python -m twine check /tmp/pyshmem-dist-review/*
python -m pip check
```

All passed. Archive inspection found the sdist contained test modules but not
`tests/conftest.py`.

### Built-in benchmarks

```bash
python -m pytest tests/test_benchmark.py -q -s
```

```text
pyshmem 128x128 CPU roundtrip: 169345.8 Hz (0.006 ms/op)
pyshmem 128x128 GPU roundtrip: 31034.8 Hz (0.032 ms/op)
CPU shared-memory MVM 1024: 9539.1 Hz (0.105 ms/op)
GPU host-upload MVM 1024: 23494.3 Hz (0.043 ms/op)
GPU device-resident MVM 1024: 29313.3 Hz (0.034 ms/op)
5 passed in 0.43s
```

### Descriptor-retention probe

In one fresh Python process, create and unlink 1,200 uniquely named one-element
streams, then compare `/proc/self/fd` and `_THREAD_LOCKS`:

```text
{'fd_before': 5, 'fd_after': 1206, 'lock_states': 1200,
 'gpu_open_locks': 0}
```

### Thread timeout probe

One thread held a stream lock for 0.5 seconds while another called
`acquire(timeout=0.05)`:

```text
{'requested_timeout_s': 0.05, 'result': 'acquired', 'elapsed_s': 0.501}
```

### Failed-write probe

Temporarily replace the implementation module's `np.copyto` with a function
that raises, then restore it and perform one normal write:

```text
first_error injected copy failure
sequence_after_failed_write 1 count 0
sequence_after_next_success 3 count 1 odd_means_stuck True
```

A following safe read spun in `_wait_for_stable_writer` until interrupted.

### GPU sequence-contract probe

Create a no-mirror CUDA stream, mark a write started, and call the separate
reader handle's default safe read before finishing:

```text
sequence_before_safe_read 1
safe_read_returned_while_odd (1024,) 1
```

This used private marking only to make the protocol state deterministic. The
public writer uses the same marker immediately before copying.

### GPU torn-read stress (negative result)

A producer alternated 160 writes of 256 MiB all-zero/all-one tensors while a
spawned consumer made 120 safe reads and checked min/max:

```text
{'reads': 120, 'mixed_snapshots': 0, 'samples': []}
```

This review therefore does not claim a torn GPU snapshot was empirically seen;
it claims the implementation does not enforce its advertised sequence protocol.

## Source locations central to findings

- Global lock/GPU caches and lock file creation: `src/pyshmem/_shared.py:102-120`
- Resource tracker/private POSIX handling: `src/pyshmem/_shared.py:246-274`
- Global purge behavior: `src/pyshmem/_shared.py:518-578`
- Metadata representation: `src/pyshmem/_shared.py:703-708`
- Sequence read/copy protocol: `src/pyshmem/_shared.py:753-811`
- Thread then file lock ordering: `src/pyshmem/_shared.py:813-861`
- Write sequence without failure recovery: `src/pyshmem/_shared.py:1189-1235`
- Polling reads: `src/pyshmem/_shared.py:1278-1305`
- GPU reduction pickle storage/load: `src/pyshmem/_shared.py:1443-1501`
- Same-process benchmark loops: `tests/test_benchmark.py`
- CI matrix and disabled benchmark enforcement: `.github/workflows/ci.yml`
- Stale former-name archives: `dist-release/pyshare-1.0.0*`

## External state checked on 2026-07-10

- PyPI listed 1.0.5, uploaded 2026-06-12 through trusted publishing, with
  provenance tied to revision `517ab7c`.
- GitHub showed one contributor identity, zero stars/forks, and restricted new
  issue creation.
- Hosted Read the Docs returned HTTP 200 but presented older GPU-opening advice
  than the repository source.

External facts can change; the dated observations above should not be treated as
permanent project attributes.
