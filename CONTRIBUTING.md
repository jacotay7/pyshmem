# Contributing

Bug reports and focused pull requests are welcome. Before opening a change,
search existing issues and describe the operating system, Python/NumPy version,
and (for GPU behavior) CUDA, driver, device, and PyTorch versions.

Set up a development environment with:

```bash
python -m pip install -e ".[test,docs]"
```

Before submitting a pull request, run:

```bash
ruff check .
ruff format --check .
pytest -m cpu
sphinx-build -W -b html docs docs/_build/html
python -m build
twine check dist/*
```

Run the full `pytest` suite on a CUDA host for GPU changes. New behavior should
include regression tests and user-facing contract changes should update the
README, Sphinx docs, and `CHANGELOG.md`. Do not commit generated `docs/_build`
content or distribution artifacts.

The project supports Linux and macOS. Windows support is intentionally out of
scope. Changes to the persistent metadata format require backward-compatible
read support, validation tests, and an update to `docs/format.rst`.

By contributing, you agree that your contribution is licensed under the
repository's GPL-3.0-only license.
