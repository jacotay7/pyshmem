# Changelog

All notable user-facing changes are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## Unreleased

## 1.4.0 - 2026-10-07

Measured on an 80-core Arm Neoverse-N1 (Linux 6.17, Python 3.13, NumPy 2.5,
RTX 4060), writer and reader pinned to separate cores, median of 3
interleaved runs; x86-64 figures exercise the x86 code path's Python cost on
the same host.

### Fixed

- **Notification waits no longer sleep through a publication.** On
  `notify=True` streams, `read_new`, `wait_for_count`, `read_after*` and
  `read_new_async` re-read the futex word just before parking instead of
  passing the sequence their wait condition had checked. A publication landing
  between the check and the park was then missed until the 50 ms park cap (or
  the caller's shorter timeout) expired. shmpipeline uses notify streams for
  every kernel trigger, where this showed up as ~10 ms p99 stalls: its smoke
  pipeline's end-to-end p99 drops from 10.5 ms to 0.12 ms, and the notify
  ping-pong's maximum from 50.8 ms to 0.2 ms.

### Performance

The public API, on-disk format and locking/consistency semantics are
unchanged; streams can be shared with processes running older releases.

- Per-publication metadata fields are bound once per handle as ctypes scalars
  (previously every atomic load or store exported the structured header to
  ctypes, ~2.6 µs each, and index access rebuilt a dict per call); libatomic
  loads/stores take the cached address directly.
- The stream lock calls `flock(2)` directly on POSIX (~1 µs per lock/unlock
  instead of ~8 µs through portalocker's flag validation and dispatch).
  portalocker uses the same `flock` primitive there and is still used on
  Windows, so mixed-version writers keep excluding each other (checked with
  four concurrent writers, two on 1.3.8: 142,769 writes, none lost, no torn
  reads).
- Stale-lock detection uses `fstat` on the open lock file instead of `stat` on
  its path; writers read back their own sequence without an atomic load.
- `locked()`, `write_view_locked()` and `locked_many()` return small context
  manager classes instead of generator/ExitStack contexts; `read_after*` no
  longer counts itself in flight twice; futex syscalls use typed prototypes;
  GPU synchronization passes the device index to `torch.cuda.current_stream`.

| Small payload (97 float32), Arm | 1.3.8 p50 / p99 (µs) | 1.4.0 p50 / p99 (µs) |
| --- | --- | --- |
| `write` | 46.6 / 57.9 | 11.2 / 20.0 |
| `read` | 11.2 / 14.7 | 3.6 / 3.9 |
| `read_publication` | 13.8 / 17.9 | 5.2 / 5.5 |
| `read_after_publication` (data ready) | 24.3 / 29.4 | 9.4 / 10.5 |
| `locked()` + `write_view_locked()` | 49.9 / 61.0 | 11.3 / 18.1 |
| `locked_many([one])` | 22.8 / 32.4 | 7.4 / 8.2 |
| `write`, x86-64 code path | 24.8 / 31.5 | 7.9 / 9.6 |

| Cross-process round trip, Arm | 1.3.8 p50 / p99 (µs) | 1.4.0 p50 / p99 (µs) |
| --- | --- | --- |
| `benchmarks/benchmark_ipc.py` CPU, 64 KiB | 299 / 405 | 115 / 196 |
| `benchmarks/benchmark_ipc.py` GPU, 64 KiB | 603 / 868 | 397 / 591 |
| notify streams, `wait_for_count`/`read_after`, 64 B | 469 / 774 | 63 / 86 |

## 1.3.8 - 2026-09-29

### Fixed

- **`open()`, `stat()` and stream discovery no longer reject a stream that
  another process is writing.** They validated the live metadata header in
  place, so a concurrent write could land between two reads of one field or
  between the lock owner and lock depth updates. About 0.2% of opens against a
  1 kHz writer failed (4% against a tight write loop) with errors such as
  `lock owner and depth metadata are inconsistent` or
  `invalid count in metadata: array(662)`, and `list_streams()` could briefly
  omit such a stream. Validation now runs on a private copy of the header and
  retries for up to 50 ms (enough for a writer preempted between two metadata
  updates), so persistent corruption is still rejected.

## 1.3.7 - 2026-09-29

### Changed

- **`import pyshmem` no longer imports torch.** torch is imported on the first
  GPU operation (`gpu_available()`, creating or opening a GPU stream, or
  reading `GPU_SUPPORTED_DTYPES`). With torch installed, this cuts
  `import pyshmem` from about 1 s to about 0.1 s, and CPU-only processes never
  load torch. `pyshmem._shared.torch` is now a lazy stand-in for the module.

## 1.3.6 - 2026-09-28

### Fixed

- Closing a handle while another thread was blocked on it in `read_new`,
  `read_after`, `wait_for_count` (or their `_publication`/async variants)
  unmapped the segments under the waiting reader and crashed the process.
  `close()` now wakes such readers (they raise `RuntimeError`) and waits for
  them to leave before unmapping; it takes a `timeout` (default 5 s).
  Non-blocking reads are unaffected and pay no extra cost.

## 1.3.5 - 2026-09-26

### Fixed

- `close()` no longer fails because another thread holds the stream's lock
  through a *different* handle (the lock is shared per name within a process,
  so a reader could not close its handle while a writer thread was mid-write).
  Closing a handle that another thread is using inside its lock scope is
  still refused.

## 1.3.4 - 2026-09-26

### Fixed

- A handle finalized by garbage collection while pyshmem was setting up
  another stream's lock state could deadlock the thread: the finalizer's
  `close()` waited on the non-reentrant registry lock its own thread held.
  Lock files are now opened outside that lock, and the lock is reentrant.

## 1.3.3 - 2026-09-26

### Fixed

- On Windows, process liveness checks (`producer_alive()`, `stat()`, and the
  abandoned-writer check that runs when a reader sees a write in progress)
  used `os.kill(pid, 0)`, which sends Ctrl+C to the whole console process
  group. They now use `OpenProcess`/`GetExitCodeProcess`.

## 1.3.2 - 2026-09-26

### Changed

- Reuse synchronous CUDA completion events per handle, process, thread, and
  active stream instead of allocating one event for every read, write, clear,
  or write-view publication. Handle close releases the event cache.

### Fixed

- Corrected stale contributor and developer documentation that still named
  GPL-3.0-only after the project moved to MIT.

## 1.3.1 - 2026-07-26

### Added

- Added immutable `Publication` snapshots and the `read_publication()`,
  `read_new_publication()`, and `read_after_publication()` APIs.  Each returns
  a payload with its matching completed count, `frame_id`, write time, and
  missed-publication count from one seqlock-verified generation, so consumers
  no longer have to combine a payload with separately sampled metadata.

### Changed

- Relicensed pyshmem under MIT after a first-party authorship and provenance
  audit. Third-party dependencies retain their own licenses.

## 1.2.0 - 2026-07-11

### Added

- Added a user `frame_id` publication token: `write()`, `write_locked()`,
  `write_view()`, and `write_view_locked()` accept `frame_id=`, and the new
  `SharedMemory.frame_id` property reads it back. The uint64 token is stamped
  atomically with the write sequence so a reader that observes a stable
  sequence sees the matching token, letting consumers establish cross-stream
  frame identity (e.g. a synchronized multi-camera fan-in). It reuses the
  previously reserved 8-byte slot in the v3 metadata header, is excluded from
  the header CRC, defaults to 0, and reads 0 on legacy v2 streams, so the
  on-disk format stays backward compatible.

## 1.1.1 - 2026-07-11

### Integration and performance

- Added `poll_interval` to `pyshmem.locked_many()`, forwarding the setting to
  each stream lock acquisition for low-latency multi-stream consumers.

## 1.1.0 - 2026-07-11

### Integration and performance

- Added exception-safe zero-copy `SharedMemory.write_view()` and
  `write_view_locked()` transactions for direct CPU/GPU publication.
- Added level-triggered `wait_for_count()` and `read_after()` APIs for
  consumers that track a publication generation.
- Added metadata-only `pyshmem.stat()` for attach/reuse decisions without
  mapping payload or CUDA IPC storage.
- Attached GPU consumers now read consistent snapshots from the device tensor
  even when a CPU mirror exists; CPU-only handles continue to consume mirrors.

## 1.0.6 - 2026-07-10

A large reliability, correctness, security, and ecosystem release.

### Reliability and correctness

- Made publication crash-safe: a failed or abandoned write publishes an invalid
  generation and readers raise `InconsistentStreamError` instead of returning
  torn data or spinning forever; a later complete write repairs the stream.
- Bounded safe reads with `read(timeout=...)`; a dead or failed writer raises
  `InconsistentStreamError` promptly rather than blocking indefinitely.
- Honored a single deadline across the thread lock and the cross-process file
  lock so lock-acquisition timeouts are respected.
- Reference-counted per-name lock state so create/close/unlink cycles no longer
  leak a file descriptor per stream, and reset inherited lock state safely after
  `fork()`.
- Added generation-safe unlink/recreate behavior and `StaleStreamError`; a stale
  handle cannot destroy a newer replacement, and live handles reconverge on one
  lock after a stream is recreated.
- Enforced interprocess publication ordering with architecture-aware atomics
  (x86-64 TSO, runtime `libatomic`) and a process-shared OS-lock fallback.
- Documented pyshmem as a capacity-one latest-value exchange and added
  per-handle `missed_writes` / `total_missed_writes` counters.
- Documented that `read_new` is edge-triggered and unsuitable for synchronous
  request/response ("ping-pong") exchanges; showed the level-triggered
  `count`-poll pattern for lock-step consumers.

### Format and security

- Introduced a documented v3 metadata format (magic, versioned fixed-width
  header, feature flags, explicit little-endian encoding) with strict corruption
  validation on open, discovery, and purge; legacy v2 metadata remains readable.
- Added a v3-metadata header CRC-32 (`header_crc` field + feature flag) covering
  the immutable header fields and name region to reject silent corruption or
  torn header writes; v2 and pre-flag v3 streams skip the check.
- Reconstructed GPU IPC handles through a restricted unpickler that permits only
  torch's known CUDA rebuild globals, so a tampered handle segment raises
  `UnpicklingError` instead of executing arbitrary code.
- Scoped `purge()` to segments whose stored name validates against their exact
  pyshmem hash; global dead-producer `cuda.shm.*` cleanup is now opt-in via
  `purge(include_cuda_orphans=True)` / `pyshmem purge --include-cuda-orphans`.
- Avoided the private `resource_tracker` reach-in by using the public
  `track=False` on Python 3.13+.

### GPU

- `open()` reconstructs a GPU stream as created: it auto-attaches to the CUDA
  device recorded in metadata, falls back to the CPU mirror when one exists, and
  accepts `gpu_device=False` to read the host mirror without attaching a tensor.
- Removed a temporary CUDA allocation and extra device copy from NumPy/CPU
  writes by copying directly into shared GPU storage.
- Added reusable `SharedMemory.pinned_buffer()` host staging for faster repeated
  host-to-GPU writes.
- Replaced whole-device CUDA synchronization with active-stream event waits for
  synchronous GPU reads, writes, and clears.
- Made GPU dtype support reflect installed PyTorch capabilities and added stable
  bool/complex codes to the CPU/persistent format.
- Made unsupported GPU or unsafe `out=` read combinations raise `ValueError`
  instead of being silently ignored.

### API and ergonomics

- Added `pyshmem.open(..., readonly=True)` for consumer handles that reject
  writes, clears, write-lock acquisition, unsafe zero-copy views, pinned-buffer
  allocation, and handle-level unlink with `PermissionError`.
- Added producer-liveness and staleness helpers: `SharedMemory.age`,
  `is_stale(max_age)`, `producer_alive()`, and `creator_pid`, plus new
  `describe()` lines. No producer-side heartbeat thread is required.
- Added DLPack support (`__dlpack__` / `__dlpack_device__`) so a handle is
  directly consumable by `np.from_dlpack`, `torch.from_dlpack`,
  `cupy.from_dlpack`, etc. The export is a seqlock-consistent snapshot (safe on
  read-only handles), not a live view.
- Added opt-in waitable notifications: `create(..., notify=True)` makes writers
  wake parked `read_new`/`read_new_async` consumers via a Linux futex instead of
  busy-polling (with a polling fallback off Linux/big-endian). Exposed via the
  `SharedMemory.notify` property; default streams are unaffected.

### Tooling, packaging, and docs

- Added a reproducible spawned-process IPC benchmark with versioned JSON
  results, including a spawned-process GPU IPC baseline (`--gpu` / `--no-gpu`,
  auto-detected) reported as `pyshmem_gpu` alongside the CPU and raw baselines.
- Made `pyproject.toml` package metadata the single version source used by the
  runtime package and documentation.
- Added Dependabot, CodeQL, and runtime dependency-vulnerability auditing, and
  gated PyPI publication on CPU tests of the exact wheel artifact under the
  minimum and newest supported Python versions.
- Added maintenance, support, security, and compatibility policies, and reworked
  the README into a concise landing page linked to the detailed docs.

## 1.0.5

- Current PyPI baseline before the repository audit remediation series.

Earlier release history is available from GitHub Releases and PyPI.
