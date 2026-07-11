# Repository-health critique

## Implementation update

Completed repository-health items from the first batch:

- Python 3.9 was added to the CI matrix;
- `tests/conftest.py` was added to the sdist manifest;
- stale tracked `pyshare-1.0.0` wheel/sdist artifacts were removed;
- repository docs were corrected for GPU auto-attachment, safe-read recovery,
  GPU `out=`, and purge scope;
- fault, lock-deadline, descriptor-lifecycle, purge-ownership, and GPU sequence
  regression tests were added.

Still open: real GPU CI, dependency/security automation, type checking,
governance/support files, hosted-doc deployment, single-source versioning,
release gating, and architectural/private-API work.

## Positive signals

- The source is small enough to audit and has a surprisingly extensive test
  suite: 120 tests passed locally with real CUDA.
- CPU CI spans Ubuntu, macOS, and Windows on Python 3.10-3.13.
- Lint, format, docs-as-warnings, package build, wheel smoke install, and twine
  validation are automated.
- PyPI publishing uses OIDC trusted publishing and PyPI shows provenance
  attestations for 1.0.5.
- Documentation covers installation, API, platform behavior, GPU modes, CLI,
  lifecycle, and benchmarks.

## Maturity and governance concerns

PyPI classifies the package as `Production/Stable`, but public GitHub showed zero
stars/forks, one maintainer, no open issues, and **issue creation restricted** at
review time. The repository history is authored by one person. Small projects
can be excellent, but an adopter has no visible bus-factor mitigation or even a
normal path to report defects.

Missing community/maintenance material includes CONTRIBUTING, SECURITY,
CODE_OF_CONDUCT, support policy, compatibility/deprecation policy, changelog
file, architecture decision records, and maintainer/release succession. The GPL
3-only license is valid but will rule out many proprietary adopters; the README
should make the consequence prominent rather than leaving it to metadata.

## Architecture concerns

### One 1,616-line implementation module

CPU transport, CUDA IPC, metadata encoding, locking, discovery, destructive
cleanup, lifecycle, polling, and public object behavior all live in
`_shared.py`. This reduces file count but increases change coupling and makes it
hard to reason separately about platform backends and invariants.

Suggested boundaries:

- format/schema and validation;
- CPU segment backend;
- CUDA/PyTorch backend;
- synchronization/notification backend;
- registry/discovery/cleanup;
- public stream/latest-value API.

### Private internal dependencies in both directions

The implementation uses private CPython and PyTorch details. More concerning,
`CLAUDE.md` states that a separate `shmpipeline` project intentionally imports
pyshmem private functions, globals, names, and `_array`. That makes the private
surface de facto public without versioning or integration tests in this repo.
Either promote a supported low-level API or remove the coupling through a small
adapter/protocol package.

### Unspecified shared-memory format

The metadata is a positional `float64[32]` array with magic represented only by
version number. There is no public format specification, endianness, checksum,
feature flags, atomic/memory-order definition, migration test, or compatibility
matrix. Persistent named memory makes format stability more important than for
ordinary in-process classes.

## CI/CD gaps

1. **No GPU workflow exists.** GPU tests run only when a developer invokes them
   locally. README text referring to manual/self-hosted GPU CI is not backed by
   a GPU test workflow in `.github/workflows`.
2. **Python 3.9 is declared but not tested.** The CI matrix begins at 3.10.
3. **No sanitizer/stress/fault-injection job.** There is no repeated concurrent
   reader/writer stress, kill-during-write recovery test, descriptor-leak test,
   fork test, or metadata-corruption fuzzing.
4. **No dependency/security automation.** No Dependabot/Renovate, dependency
   audit, CodeQL, or secret scanning configuration is present.
5. **No type checking.** The package has partial annotations but no mypy/pyright
   gate and no `py.typed` marker for consumers.
6. **Benchmarks never gate regressions.** Hosted smoke tests explicitly disable
   the only rate assertion.
7. **Release publishing is not gated on CI in the publish workflow.** A GitHub
   release triggers a fresh build/publish but does not declare a dependency on
   a specific successful CI artifact or test the built artifact's actual CPU
   behavior before upload.
8. **Actions are tag-pinned, not commit-pinned.** This is common but weaker for
   supply-chain hardening.

## Packaging hygiene

- `dist-release/` contains tracked `pyshare-1.0.0` wheel/sdist artifacts from
  the package's former name. They are stale, confusing, and unnecessary in
  source control.
- The freshly built sdist includes all three test modules but omits
  `tests/conftest.py`, which supplies fixtures those tests require. It therefore
  ships a visibly incomplete test suite.
- Version `1.0.5` is duplicated in `pyproject.toml`, `pyshmem.__init__`, and
  `docs/conf.py`, inviting drift.
- `IMPROVEMENTS.md` is explicitly ignored, so maintenance thinking in that file
  is not shared with adopters or contributors.
- There is no lockfile/constraints strategy for CI. `torch>=2.2` has no upper
  compatibility bound even though implementation details are serialized and
  private APIs are used.

## Documentation and public-facing inconsistencies

1. README lines 337-345 say users must pass `gpu_device` and omission gives a
   CPU-only handle. Current code and the earlier README quick start say omission
   auto-attaches to the recorded GPU. These are mutually exclusive instructions.
2. The hosted Read the Docs `latest/gpu.html` still said "Always pass
   gpu_device" on 2026-07-10, while repository docs say auto-attach. The hosted
   documentation is behind the released source behavior.
3. The user guide says CPU accepts "any NumPy dtype"; the fixed table accepts
   only 11.
4. The platform/persistence message is easy to overread: Linux CPU segments can
   persist without a process, CUDA allocations cannot provide the same
   no-producer semantics, Windows drops the last-handle object, and discovery is
   Linux-specific.
5. Benchmark methodology and results appear in README but not as versioned raw
   artifacts with a reproducible sweep command.
6. `out` is described as zero-copy in implementation documentation even though
   it performs a copy; on GPU it is silently ignored.

## Coverage interpretation

83% statement coverage is healthy for a young project but not sufficient
evidence for concurrency correctness. Important uncovered areas include cleanup
exceptions, malformed metadata, platform fallbacks, attach failures, and
several recovery paths. Branch coverage is not configured. The most serious
bugs found in this review are invariant/lifetime bugs that high statement
coverage did not expose.
