# Support and compatibility

Community support is provided through GitHub issues. Include a minimal
reproducer and environment details. There is no paid support, service-level
agreement, or guaranteed response time.

The supported platforms are Linux and macOS with the Python versions declared
in `pyproject.toml`. Windows is unsupported. CPU behavior is exercised in
GitHub Actions; CUDA behavior is validated manually on the development GPU
until a GPU runner is configured.

Public names exported by `pyshmem.__all__` follow semantic versioning.
Deprecations should remain for at least one minor release before removal.
Private names, serialized PyTorch CUDA reduction details, and direct imports
from `pyshmem._shared` are not compatibility promises. Metadata format v3 is a
persistent compatibility contract and readers retain v2 compatibility.

See `SECURITY.md` for private vulnerability reports and `CONTRIBUTING.md` for
development checks.
