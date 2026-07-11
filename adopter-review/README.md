# pyshmem critical adopter review

Review date: 2026-07-10

Reviewed revision: `517ab7c` (`main`, package version 1.0.5)

Environment: Linux, Python 3.12.0, NumPy 2.2.6, PyTorch 2.10.0+cu128,
NVIDIA GeForce RTX 5090

Implementation update: 2026-07-10, first remediation batch completed after the
review. See [Implementation status](00-implementation-status.md). Findings below
retain the original review evidence; status annotations record later fixes.

## Bottom line

pyshmem is a useful, unusually small wrapper around named CPU shared memory and
PyTorch CUDA IPC. Its CPU path is easy to understand, the public API is compact,
the tests are far better than the repository's size suggests, and packaging,
lint, documentation build, and the complete local CPU/GPU suite all passed.

At the reviewed revision I would not have adopted 1.0.5 as a production-grade
unified streaming layer. The first remediation batch has now fixed the five
immediate reliability blockers identified here: failed/crashed-write handling,
lock deadlines, per-name descriptor retention, purge ownership, and the GPU
sequence contract. The remaining adoption caveats concern memory ordering,
single-slot product semantics, GPU lifecycle/async performance, reproducible
benchmarks, and project-maintenance infrastructure.

The central product-positioning issue is that a pyshmem "stream" is one mutable
array slot. It has no queue, history, capacity, backpressure, consumer cursors,
delivery acknowledgement, or missed-frame count. That is a valid and valuable
primitive for cameras and control loops, but it is not what many adopters will
infer from "shared memory streams."

## Adoption scorecard

| Area | Assessment | Why |
|---|---|---|
| CPU latest-value exchange | Promising | Simple API, consistent-copy loop, strong local tests, good bandwidth |
| GPU latest-value exchange | Improved; further hardening open | Safe reads now honor the sequence protocol; CUDA event/stream ordering remains open |
| Stream/queue semantics | Missing | Single slot only; intermediate writes are overwritten |
| Failure recovery | Implemented | Failed/dead writers yield an explicit invalid state; a later full write repairs it |
| Long-lived dynamic workloads | Implemented | Per-name lock state is reference-counted and closed after the last handle |
| Cross-platform claim | Overstated | Discovery/purge are Linux-only; Windows persistence differs; GPU is Linux-only in practice |
| Maintenance posture | Early-stage | One contributor, restricted issue creation, no GPU CI, no compatibility policy |
| Documentation | Substantial but inconsistent | Good breadth, but README contradicts the code and hosted docs lag the release |
| Benchmarks | Directionally useful only | Same-process microbenchmarks, no baselines/statistics, public tables are not reproducible from a checked-in runner |

## Highest-priority findings

Status as of the implementation update: items 1-5 are implemented and covered
by regression tests. Items 6-7 remain open.

1. **P0 — failed/crashed writes poison the sequence protocol.** There is no
   `try/finally` or recovery state around odd/even sequence updates. A probe
   injected one `np.copyto` failure: sequence became 1; the next successful
   write ended at 3; `read(safe=True)` then waited forever.
2. **P0 — lock timeouts are incorrect for threads.** `RLock.acquire()` blocks
   without a timeout before the timed file-lock path. A requested 0.05 s
   timeout returned successfully after 0.501 s.
3. **P0 — distinct stream names leak file descriptors.** The global
   `_THREAD_LOCKS` cache never evicts or closes `_SharedLockState.file_handle`.
   A fresh process went from 5 to 1,206 descriptors after 1,200 create/unlink
   cycles.
4. **P0 — `purge()` exceeds its advertised ownership boundary.** It unlinks
   every `/dev/shm/ps_*` object without validating a pyshmem metadata signature
   and sweeps orphaned `cuda.shm.*` objects created by any PyTorch application,
   not only pyshmem.
5. **P1 — default no-mirror GPU reads are not safe snapshots.** The GPU clone
   path never reads or rechecks `write_sequence`. A deterministic probe showed
   `read(safe=True)` returning while the sequence was odd. A concurrent stress
   run did not observe torn values, so this is a contract/protocol defect rather
   than a demonstrated torn-copy failure on this GPU.
6. **P1 — the name "stream" overpromises.** There is no FIFO/ring buffer or
   way for a consumer to detect exactly how many updates it missed.
7. **P1 — public benchmark claims are not reproducible.** Default timed regions
   are milliseconds, despite the README saying every case ran at least 1.5 s;
   the public size sweep has no checked-in runner or raw result artifact.

## What was verified successfully

- `pytest tests -q`: **120 passed in 10.73 s**, CPU and GPU included.
- Non-benchmark suite with coverage: **115 passed**, 83% statement coverage
  overall (74% CLI, 83% core implementation).
- `ruff check .` and `ruff format --check .`: passed.
- `sphinx-build -W`: passed.
- isolated sdist/wheel build and `twine check`: passed.
- `pip check`: passed.
- Built-in benchmark smoke suite: 5 passed.
- Cross-process CUDA creation, attachment, locking, and lifecycle tests passed,
  although the run ended with PyTorch's warning that a producer terminated
  before all shared CUDA tensors were released.

Post-remediation verification:

- complete suite: **129 passed in 11.74 seconds**;
- branch-aware coverage run: **123 passed**, 81% total coverage;
- 1,200 unique create/unlink cycles: descriptor growth reduced from 1,201 to a
  bounded one-time resource-tracker descriptor, with zero retained lock states;
- failed-write probe: `InconsistentStreamError` at generation `-1`, followed by
  successful recovery at stable generation `4`;
- lint, formatting, warning-strict docs, sdist/wheel build, and `twine check`
  all passed;
- CPU 128x128 roundtrip: 167,155 Hz; sequence-safe GPU roundtrip: 27,516 Hz.

## Report map

- [Implementation status](00-implementation-status.md)
- [Feature set](01-feature-set.md)
- [Correctness and reliability](02-correctness-and-reliability.md)
- [Performance](03-performance.md)
- [Repository health](04-repository-health.md)
- [Alternatives](05-alternatives.md)
- [Prioritized recommendations](06-prioritized-recommendations.md)
- [Methodology and evidence](07-methodology-and-evidence.md)
