import doctest


project = "ifnt"
html_theme = "sphinx_book_theme"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]
exclude_patterns = [
    "venv",
    "README.rst",
]
intersphinx_mapping = {
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
}
doctest_global_setup = """
import ifnt
import jax
from jax import numpy as jnp
"""
doctest_default_flags = (
    doctest.ELLIPSIS | doctest.DONT_ACCEPT_TRUE_FOR_1 | doctest.NORMALIZE_WHITESPACE
)
nitpick_ignore = [
    # https://github.com/sphinx-doc/sphinx/issues/10974.
    ("py:class", "ifnt.util.F"),
]
