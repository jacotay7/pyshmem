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
| Metadata corruption validation | Done | Open, discovery, and purge validate header/segment length, flags, reserved bytes, stored name, dtype, dimensions, shape, byte-size product, CPU/GPU rules, creator and lock fields, timestamps, unused dimensions, and actual payload segment size before mapping. |
| Interprocess memory model specified | Done: documented model | `docs/format.rst` now specifies encoding, alignment, the seqlock protocol, what pyshmem relies on (single-writer serialization, aligned 8-byte counter atomicity, program-order publication), what it does *not* provide (no hardware barriers), and the validated architectures (x86-64, aarch64). Platform docs and README narrow correctness claims accordingly. A regression test enforces 8-byte alignment of the hot-path counters. A native acquire/release atomic backend remains the open enforcement piece. |
| Unlink/recreate lock-inode generation | Done | `_SharedLockState` records the lock file's inode and rebinds a stale handle on each acquire when the pathname resolves to a new inode, so a stream destroyed and recreated while old handles are live reconverges on one shared lock instead of splitting into per-generation locks. Regression tests cover the refresh mechanism and post-recreate convergence (verified load-bearing against a disabled-refresh baseline); `docs/platforms.rst` documents the semantics. |
| Private resource-tracker API | Done | All segment open/create paths go through `_attach_segment()`, which uses the public `track=False` argument on Python 3.13+ and only falls back to the private `resource_tracker.unregister` on <=3.12. Tests assert the capability probe matches the `SharedMemory` signature and that `_attach_segment` branches correctly (track-false path never unregisters; fallback path does). The pickled CUDA reduction trust boundary is a separate, still-open item. |
| Fault testing: metadata/kills/contention | Done (CPU scope) | Added CPU regression tests for a truncated metadata segment (clean rejection), repeated writer kills each recovering to `InconsistentStreamError` then repairing, and concurrent multi-writer/reader contention proving no torn seqlock snapshots and no deadlock (stable across 10 repeats). |
| Fault testing: CUDA failure during publication | Done | GPU regression test injects a CUDA error at publication-time synchronize and asserts the write path's `_abort_write` leaves the stream invalid (`InconsistentStreamError` on read) and that a later good write repairs it. Validated on an RTX 5090. Fork inheritance remains the last open fault-testing item. |
| Pickle CUDA trust boundary | Done | GPU handle reconstruction no longer calls raw `pickle.loads` on the writable 0600 segment. `_RestrictedCudaUnpickler` permits only torch's known CUDA rebuild globals and inert dtype values, so a tampered payload raises `UnpicklingError` instead of executing code. Validated on an RTX 5090: a legit reduction still round-trips, a child opening a tampered handle fails with `disallowed global`, and a torch-independent CPU test covers the rejection path. Documented in `docs/format.rst` and CLAUDE.md. |

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

1. Enforce the specified memory model in hardware. Fixed-width aligned counters
   and the model itself are now documented and the alignment contract is
   test-guarded, so the remaining step is a native acquire/release atomic
   backend (or FFI synchronization layer) for weakly ordered architectures.
2. Extend format validation only when new fields/features are introduced. The
   current v3 semantic fields and segment geometry are validated; checksums or
   authenticated metadata remain optional future hardening.
3. Resolve remaining unlink/recreate edge cases while old handles remain open.
   Lock-inode generation and stale-handle reconvergence are handled; data and
   metadata segment recreation while consumers hold prior mappings still needs
   explicit documentation and coverage.
4. Residual private PyTorch reduction internals. The pickle trust boundary is
   now closed with an authenticating restricted unpickler, and the private
   `resource_tracker` reach-in is avoided on Python 3.13+ via public
   `track=False`. What remains is that reconstruction still depends on torch's
   private `rebuild_cuda_tensor`/`_lazy_init` internals, which carry no stable
   API guarantee across torch versions.
5. Expand fault testing. Malformed/truncated metadata, repeated writer kills,
   multi-writer contention (CPU), and CUDA-failure-during-publication (GPU) now
   have regression tests. Fork/spawn inheritance is the last open item.

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
