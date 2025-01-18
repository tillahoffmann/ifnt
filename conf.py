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
from jax import scipy as jsp
"""
doctest_default_flags = (
    doctest.ELLIPSIS | doctest.DONT_ACCEPT_TRUE_FOR_1 | doctest.NORMALIZE_WHITESPACE
)
nitpick_ignore = [
    # https://github.com/sphinx-doc/sphinx/issues/10974.
    ("py:class", "ifnt.util.F"),
    # Not documented by jax.
    ("py:class", "jax._src.core.NamedShape"),
    ("py:class", "jax._src.typing.SupportsDType"),
    # Only a problem for Python 3.9.
    ("py:class", "Array"),
    ("py:class", "ArrayLike"),
    ("py:class", "DTypeLikeFloat"),
    ("py:class", "DTypeLikeUInt"),
    ("py:class", "NamedShape"),
    ("py:class", "RealArray"),
    ("py:class", "Shape"),
]
suppress_warnings = [
    "ref.footnote",  # jax._src.random.orthogonal:15 has an unreferenced footnote.
]
