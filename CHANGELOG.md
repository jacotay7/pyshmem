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

## 1.0.5

- Current PyPI baseline before the repository audit remediation series.

Earlier release history is available from GitHub Releases and PyPI.
