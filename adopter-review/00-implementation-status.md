# Implementation status

Status date: 2026-07-10

Baseline reviewed: `517ab7c`, pyshmem 1.0.5

Scope: first remediation batch following the critical adopter review

## Completed

| Review item | Status | Implementation and evidence |
|---|---|---|
| Failed writes poison the sequence | Done | Failed copies publish a negative invalid generation and raise `InconsistentStreamError`; a later complete write starts a fresh generation and repairs the stream. Regression coverage includes injected `np.copyto` failure. |
| Writer process dies mid-write | Done | Readers detect a dead recorded lock owner, acquire the released OS lock, mark the generation invalid, and raise instead of spinning forever. A spawned-process crash regression test covers the path. |
| Safe reads can wait forever | Done for active writes | `read(timeout=...)` bounds stable-write waiting. Dead/failed writers produce `InconsistentStreamError`. A live but indefinitely hung writer still requires the caller to supply a timeout. |
| Thread lock timeout ignored | Done | One monotonic deadline now covers both `threading.RLock` and the process file lock. The 50 ms regression test returns by deadline rather than waiting 500 ms. |
| Per-name file-descriptor leak | Done | `_SharedLockState` is reference-counted; the final local handle evicts the cache entry and closes its file. A 1,200-name probe retained zero lock states and only the process's bounded resource-tracker descriptor. |
| Purge deletes unrelated `ps_*` objects | Done | Discovery and purge require metadata version, stored friendly name, and exact name-to-hash validation. A regression test proves an unvalidated `ps_0123456789abcd` segment survives. |
| Purge sweeps global PyTorch IPC by default | Done | `purge()` no longer touches `cuda.shm.*` by default. Global dead-producer cleanup requires `include_cuda_orphans=True` or `pyshmem purge --include-cuda-orphans`. |
| No-mirror GPU `safe=True` ignores sequence | Done at host protocol level | GPU safe reads now wait for an even generation, clone and synchronize, then recheck the generation and retry on overlap. Cross-process CUDA-event ordering is still a deeper open item. |
| GPU `out=` silently ignored | Done | Unsupported GPU or unsafe `out=` combinations raise `ValueError`. CPU `out=` remains zero-allocation, not zero-copy. |
| README GPU-open contradiction | Done in repository docs | README now agrees with implementation: omitted `gpu_device` auto-attaches; `False` requests a CPU mirror. Hosted Read the Docs still needs a deployment/rebuild after push. |
| Python 3.9 declared but absent from CI | Done | Python 3.9 added to the OS/Python test matrix. |
| Incomplete sdist tests | Done | `MANIFEST.in` includes `tests/conftest.py`; rebuilt archive contains the fixture and all test modules. |
| Obsolete former-name distributions | Done | Tracked `dist-release/pyshare-1.0.0*` artifacts removed. |
| Persistent metadata representation | Done: format foundation | New streams use a documented v3 256-byte little-endian header with magic, feature flags, aligned fixed-width integer counters and dimensions. Readers retain v2 compatibility. Native acquire/release atomics remain open. |

## Verification record

```text
pytest tests -q -ra                 129 passed in 11.74s
pytest ... --cov-branch             123 passed; 81% total coverage
ruff check .                        passed
ruff format --check .               passed
sphinx-build -W                     passed
python -m build                     passed
twine check                         passed
benchmark smoke                     5 passed
```

The test process still emits PyTorch's warning that a producer terminated
before all shared CUDA tensors were released. The tests pass, but eliminating or
precisely isolating that lifecycle warning remains open.

## Remaining work, in priority order

### P1 correctness and contract

1. Define and enforce the interprocess memory model. Replace `float64` metadata
   counters with fixed-width integers and use acquire/release-capable atomic
   operations or a native synchronization layer.
2. Complete corruption validation for the documented v3 format. Magic,
   endianness, alignment, feature flags, and v2 compatibility are implemented;
   semantic field and segment-geometry validation is the next step.
3. Resolve unlink/recreate behavior while old handles remain open, including
   lock-inode generation and stale-handle semantics.
4. Replace or tightly isolate the pickled CUDA reduction trust boundary and
   private CPython/PyTorch APIs.
5. Expand fault testing: malformed metadata, fork inheritance, CUDA failure
   during publication, repeated writer kills, and prolonged contention stress.

### P1 product and validation

1. State prominently that the existing primitive is a capacity-one latest-value
   exchange, or add a capacity-N ring buffer with overrun reporting and
   backpressure/drop policies.
2. Build a reproducible spawned-process benchmark harness with raw shared-memory
   and PyTorch baselines, calibrated duration, repetitions, percentiles, and
   versioned JSON results.
3. Add actual GPU CI hardware and test the minimum/newest supported PyTorch
   versions; the current GitHub workflow remains CPU-only.
4. Rebuild hosted Read the Docs and enable a normal public issue-reporting path.

### P2 performance and ecosystem

1. Add CUDA stream/event-aware asynchronous operations and avoid whole-device
   synchronization.
2. Add reusable pinned staging buffers and direct host-to-shared-GPU copies.
3. Replace polling with waitable notifications plus an optional adaptive spin
   policy.
4. Add DLPack/array-interface adapters, capability-driven dtype support,
   namespaces, read-only handles, and producer heartbeat/staleness metadata.
5. Split the implementation by format, synchronization, lifecycle, CPU backend,
   and CUDA backend once the contracts above are fixed.
