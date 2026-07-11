# Implementation status

Status date: 2026-07-10

Baseline reviewed: `517ab7c`, pyshmem 1.0.5

Scope: cumulative remediation following the critical adopter review

Platform decision: Windows support was intentionally removed. The supported
surface is Linux and macOS; CUDA support is Linux-only.

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
| Unlink/recreate generation safety | Done | New streams carry a random 128-bit instance id. Create and unlink serialize on a persistent per-name lock inode; handle-level unlink rejects a newer replacement with `StaleStreamError`. Tests cover stable lock identity, changing generation ids, old-mapping isolation, and replacement survival. |
| Private resource-tracker API | Done | All segment open/create paths go through `_attach_segment()`, which uses the public `track=False` argument on Python 3.13+ and only falls back to the private `resource_tracker.unregister` on <=3.12. Tests assert the capability probe matches the `SharedMemory` signature and that `_attach_segment` branches correctly (track-false path never unregisters; fallback path does). The pickled CUDA reduction trust boundary is a separate, still-open item. |
| Fault testing: metadata/kills/contention | Done (CPU scope) | Added CPU regression tests for a truncated metadata segment (clean rejection), repeated writer kills each recovering to `InconsistentStreamError` then repairing, and concurrent multi-writer/reader contention proving no torn seqlock snapshots and no deadlock (stable across 10 repeats). |
| Fault testing: CUDA failure during publication | Done | GPU regression test injects a CUDA error at publication-time synchronize and asserts the write path's `_abort_write` leaves the stream invalid (`InconsistentStreamError` on read) and that a later good write repairs it. Validated on an RTX 5090. |
| Fork-state hardening | Done | An `os.register_at_fork` child handler resets each inherited `_SharedLockState` (fresh re-entrant lock, cleared held flag, reopened private lock-file descriptor) and drops cached CUDA IPC tensors. A regression test forks while the parent holds the lock and asserts the child neither inherits held state nor shares the parent's lock (its acquire blocks on the parent), verified load-bearing against a no-reset baseline. Documented in `docs/platforms.rst`. |
| Pickle CUDA trust boundary | Done | GPU handle reconstruction no longer calls raw `pickle.loads` on the writable 0600 segment. `_RestrictedCudaUnpickler` permits only torch's known CUDA rebuild globals and inert dtype values, so a tampered payload raises `UnpicklingError` instead of executing code. Validated on an RTX 5090: a legit reduction still round-trips, a child opening a tampered handle fails with `disallowed global`, and a torch-independent CPU test covers the rejection path. Documented in `docs/format.rst` and CLAUDE.md. |
| Interprocess publication ordering enforced | Done | Safe reads use x86-64 TSO directly, runtime `libatomic` acquire/release operations elsewhere when available, and a process-shared OS-lock barrier fallback. Regression tests force native and fallback paths; payload copies remain outside the lock with sequence retry semantics. |
| Capacity-one contract and overrun reporting | Done | README and overview now lead with latest-value/capacity-one semantics. Each handle exposes `last_read_count`, per-read `missed_writes`, and cumulative `total_missed_writes`; regression coverage proves skipped publications are counted without changing the zero-queue design. |
| Spawned-process benchmark harness | Done | `benchmarks/benchmark_ipc.py` calibrates repeated spawn-based request/ack IPC runs and reports throughput plus p50/p95/p99 latency in versioned JSON. It includes an explicitly unsafe raw `multiprocessing.shared_memory` lower-bound baseline, a CI smoke test, and checked-in RTX 5090/Linux results. |
| Maintainer and adopter policies | Done | Added contributor workflow, security disclosure instructions, support/platform/API compatibility policy, and a repository changelog; README links all four. CODE_OF_CONDUCT and succession policy are deferred until the project has multiple participants. |
| Single-source versioning | Done | `pyproject.toml` is the only literal version. The public `__version__` and Sphinx release value use `importlib.metadata`; a test prevents installed metadata/runtime drift. |
| Dependency/security automation | Done | Dependabot tracks Python and GitHub Actions updates; CI runs strict `pip-audit` against installed runtime dependencies; CodeQL runs on pushes, pull requests, and weekly. Repository-host settings such as secret scanning remain external. |
| Release artifact gating | Done | The PyPI workflow builds once, validates distributions, then installs and runs the CPU suite against that exact wheel on Python 3.9 and 3.13. OIDC publishing depends on both wheel-test jobs succeeding. |
| README as adopter landing page | Done | Replaced the 462-line duplicated manual with a concise overview, CPU/GPU quick start, installation, reproducible performance summary, and license/contact sections. Each section links to the authoritative detailed docs. |
| Direct host-to-shared-GPU writes | Done | NumPy/CPU values remain host tensors until `shared_cuda_tensor.copy_`, eliminating the temporary GPU allocation and extra D2D copy. A regression test checks zero-copy NumPy wrapping on CPU; a 4 MB local probe improved from the audited 186.25 us to median 171.70 us. |
| Reusable pinned GPU staging | Done | `SharedMemory.pinned_buffer()` lazily allocates and reuses a correctly shaped/dtyped page-locked CPU tensor, writable through a zero-copy NumPy view. Regression coverage verifies reuse and round-trip correctness; repeated 4 MB writes measured median 149.67 us versus 171.70 us pageable. |
| Stream-local CUDA synchronization | Done | GPU read/write/clear records and synchronizes an event on the active CUDA stream instead of calling whole-device `torch.cuda.synchronize`. Publication remains synchronous and safe. A regression test forbids the global API while round-trip and clear operations succeed. Fully async cross-process publication remains open. |
| Capability-driven dtype support | Done | GPU dtype support is derived from attributes exposed by the installed PyTorch instead of a stale fixed subset. The persistent table was backward-compatibly extended with bool and complex64/128. CPU and real-CUDA tests cover bool/complex and torch 2.10 unsigned 16/32/64-bit round trips. |
| Read-only consumer handles | Done | `open(..., readonly=True)` returns a per-handle guarded consumer: `write`, `write_locked`, `clear`, `acquire`/`locked`, `pinned_buffer`, unsafe (`safe=False`) reads, and handle-level `unlink` raise `PermissionError`, while reads snapshot normally. The guard is per handle, not segment-level, so other writable handles and the owner keep publishing; `describe()` reports the flag. CPU and real-CUDA (RTX 5090) regression tests cover snapshot-then-reject behaviour. |

## Verification record

```text
pytest tests -q                     173 passed in 19.41s (RTX 5090)
pytest tests/test_cpu_api.py -q      112 passed in 7.88s
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

1. Maintain the implemented publication backends as architectures are added.
   x86-64 TSO, runtime `libatomic`, and the OS-lock fallback now enforce the
   specified model; no known supported-platform ordering gap remains.
2. Extend format validation only when new fields/features are introduced. The
   current v3 semantic fields and segment geometry are validated; checksums or
   authenticated metadata remain optional future hardening.
3. Residual private PyTorch reduction internals. The pickle trust boundary is
   now closed with an authenticating restricted unpickler, and the private
   `resource_tracker` reach-in is avoided on Python 3.13+ via public
   `track=False`. What remains is that reconstruction still depends on torch's
   private `rebuild_cuda_tensor`/`_lazy_init` internals, which carry no stable
   API guarantee across torch versions.
4. Fault testing is now covered: malformed/truncated metadata, repeated writer
   kills, multi-writer contention, CUDA-failure-during-publication, and
   fork-state inheritance all have regression tests. Prolonged multi-host or
   many-hour soak stress could still be added but is not a correctness gap.

### P1 product and validation

1. Add a spawned-process PyTorch/CUDA IPC baseline to the now-reproducible CPU
   harness when CUDA event-aware behavior is implemented.
2. Add actual GPU CI hardware and test the minimum/newest supported PyTorch
   versions; the current GitHub workflow remains CPU-only.
3. Rebuild hosted Read the Docs and enable a normal public issue-reporting path.

### P2 performance and ecosystem

1. Design fully asynchronous cross-process publication using IPC-capable CUDA
   events. Synchronous operations now wait only on their active stream and no
   longer synchronize the whole device.
2. Replace polling with waitable notifications plus an optional adaptive spin
   policy.
3. Add DLPack/array-interface adapters, namespaces, read-only handles, and
   producer heartbeat/staleness metadata. Dtype support is now capability-driven.
4. Split the implementation by format, synchronization, lifecycle, CPU backend,
   and CUDA backend once the contracts above are fixed.
