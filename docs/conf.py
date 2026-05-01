from __future__ import annotations

import os
import sys


ROOT = os.path.abspath("..")
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

project = "pyshmem"
author = "Jacob Taylor"
copyright = "2026, Jacob Taylor"
release = "1.0.1"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "class"

html_theme = "furo"
html_title = "pyshmem documentation"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
