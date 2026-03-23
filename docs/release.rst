Release and Publishing
======================

PyPI publishing
---------------

The repository includes a trusted-publishing workflow at
``.github/workflows/pypi.yml``.

It performs two phases:

- build source and wheel distributions
- run ``twine check`` before publishing to PyPI

The publish job uses GitHub's OIDC-based trusted publishing flow through the
``pypa/gh-action-pypi-publish`` action.

Read the Docs
-------------

Read the Docs is configured through ``.readthedocs.yaml``. The RTD build:

- uses Python 3.12
- installs the package with the ``docs`` extra
- builds this Sphinx site from ``docs/conf.py``

Before tagging a release
------------------------

- run the test suite
- ensure the README and docs reflect the current API
- verify the project version in ``pyproject.toml``
- create and publish a GitHub release to trigger the PyPI workflow