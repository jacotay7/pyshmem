# Repository-health critique

## Implementation update

Completed repository-health items from the first batch:

- Python 3.9 was added to the CI matrix;
- `tests/conftest.py` was added to the sdist manifest;
- stale tracked `pyshare-1.0.0` wheel/sdist artifacts were removed;
- repository docs were corrected for GPU auto-attachment, safe-read recovery,
  GPU `out=`, and purge scope;
- fault, lock-deadline, descriptor-lifecycle, purge-ownership, and GPU sequence
  regression tests were added;
- a spawned-process IPC benchmark has a CI smoke test and a versioned JSON
  result.

Still open: real GPU CI, type checking, hosted-doc deployment, and
architectural/private-API work. Contributor, security, support/compatibility,
and changelog policies have now been added.

## Positive signals

- The source is small enough to audit and has a surprisingly extensive test
  suite: 120 tests passed locally with real CUDA.
- CPU CI spans Ubuntu and macOS on Python 3.9-3.13. Windows support was
  intentionally removed.
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

The repository now includes CONTRIBUTING, SECURITY, support/compatibility, and
changelog files. CODE_OF_CONDUCT, architecture decision records, and
maintainer/release succession remain premature for a one-person project. The GPL
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
4. **Resolved in-repository:** Dependabot covers pip and Actions, CI runs a
   strict runtime `pip-audit`, and CodeQL runs on changes and weekly. GitHub
   secret scanning remains a repository-host setting.
5. **No type checking.** The package has partial annotations but no mypy/pyright
   gate and no `py.typed` marker for consumers.
6. **Benchmarks never gate regressions.** Hosted smoke tests explicitly disable
   the only rate assertion.
7. **Resolved for CPU:** the publish workflow builds once, then tests that exact
   wheel on Python 3.9 and 3.13. OIDC upload depends on both jobs. GPU release
   gating still requires an external CUDA runner.
8. **Actions are tag-pinned, not commit-pinned.** This is common but weaker for
   supply-chain hardening.

## Packaging hygiene

- `dist-release/` contains tracked `pyshare-1.0.0` wheel/sdist artifacts from
  the package's former name. They are stale, confusing, and unnecessary in
  source control.
- The freshly built sdist includes all three test modules but omits
  `tests/conftest.py`, which supplies fixtures those tests require. It therefore
  ships a visibly incomplete test suite.
- **Resolved:** `pyproject.toml` is the only literal version source; runtime and
  Sphinx obtain it through `importlib.metadata`, with a regression test.
- `IMPROVEMENTS.md` is explicitly ignored, so maintenance thinking in that file
  is not shared with adopters or contributors.
- There is no lockfile/constraints strategy for CI. `torch>=2.2` has no upper
  compatibility bound even though implementation details are serialized and
  private APIs are used.

## Documentation and public-facing inconsistencies

1. **Resolved:** README and repository docs consistently say omission
   auto-attaches to the recorded GPU and `gpu_device=False` selects a CPU mirror.
2. The hosted Read the Docs `latest/gpu.html` still said "Always pass
   gpu_device" on 2026-07-10, while repository docs say auto-attach. The hosted
   documentation is behind the released source behavior.
3. **Resolved:** dtype documentation lists the actual supported table.
4. **Resolved:** platform docs distinguish persistent CPU segments, CUDA
   producer lifetime, Linux-only discovery, and explicitly unsupported Windows.
5. **Resolved for CPU:** a reproducible process benchmark and versioned JSON
   result are checked in; CUDA still needs an equivalent baseline.
6. **Resolved:** `out` is documented as zero-allocation and unsupported GPU or
   unsafe combinations raise rather than being ignored.

## Coverage interpretation

83% statement coverage is healthy for a young project but not sufficient
evidence for concurrency correctness. Important uncovered areas include cleanup
exceptions, malformed metadata, platform fallbacks, attach failures, and
several recovery paths. Branch coverage is not configured. The most serious
bugs found in this review are invariant/lifetime bugs that high statement
coverage did not expose.
