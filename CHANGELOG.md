# Changelog

All notable user-facing changes are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## Unreleased

- Documented pyshmem as a capacity-one latest-value exchange and added
  per-handle missed-publication counters.
- Added generation-safe unlink/recreate behavior and `StaleStreamError`.
- Enforced interprocess publication ordering with architecture-aware atomics
  and an OS-lock fallback.
- Added a reproducible spawned-process IPC benchmark and versioned JSON result.
- Added maintenance, support, security, and compatibility policies.
- Made `pyproject.toml` package metadata the single version source used by the
  runtime package and documentation.
- Added Dependabot, CodeQL, and runtime dependency-vulnerability auditing.
- Gated PyPI publication on CPU tests of the exact wheel artifact under the
  minimum and newest supported Python versions.
- Reworked README into a concise project, quick-start, installation,
  performance, license, and contact landing page linked to detailed docs.
- Removed a temporary CUDA allocation and extra device copy from NumPy/CPU
  writes by copying directly into shared GPU storage.
- Added reusable `SharedMemory.pinned_buffer()` host staging for faster repeated
  host-to-GPU writes.
- Replaced whole-device CUDA synchronization with active-stream event waits for
  synchronous GPU reads, writes, and clears.
- Made GPU dtype support reflect installed PyTorch capabilities and added stable
  bool/complex codes to the CPU/persistent format.
- Added `pyshmem.open(..., readonly=True)` for consumer handles that reject
  writes, clears, write-lock acquisition, unsafe zero-copy views, pinned-buffer
  allocation, and handle-level unlink with `PermissionError`.
- Documented that `read_new` is edge-triggered and unsuitable for synchronous
  request/response ("ping-pong") exchanges; showed the level-triggered
  `count`-poll pattern for lock-step consumers.
- Added a v3-metadata header CRC-32 (`header_crc` field + feature flag) covering
  the immutable header fields and name region, validated on open, discovery, and
  purge to reject silent corruption or torn header writes. Backward-compatible:
  version 2 and pre-flag version 3 streams skip the check.

## 1.0.5

- Current PyPI baseline before the repository audit remediation series.

Earlier release history is available from GitHub Releases and PyPI.
