# Changelog

All notable user-facing changes are documented here. The project follows
[Semantic Versioning](https://semver.org/).

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
