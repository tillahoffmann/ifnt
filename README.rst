🚀 ifnt
=======

.. image:: https://github.com/tillahoffmann/ifnt/actions/workflows/build.yml/badge.svg
    :target: https://github.com/tillahoffmann/ifnt/actions/workflows/build.yml
.. image:: https://readthedocs.org/projects/ifnt/badge/?version=latest
    :target: https://ifnt.readthedocs.io/en/latest/?badge=latest
.. image:: https://img.shields.io/pypi/v/jax-ifnt
    :target: https://pypi.org/project/jax-ifnt

Execute runtime assertions, indexing checks, and more if :code:`jax` code is not traced.

Overview
--------

ifnt comprises three core modules, :code:`ifnt.util` with utility functions, :code:`ifnt.testing` for runtime assertions, and :code:`ifnt.random` for stateful random number generation.

    >>> import ifnt
    >>> import jax
    >>> from jax import numpy as jnp
    >>>
    >>> def safe_log(x):
    ...     ifnt.testing.assert_array_less(0, x)
    ...     return jnp.log(x)
    >>>
    >>> safe_log(-1)
    Traceback (most recent call last):
    ...
    AssertionError: Arrays are not strictly ordered `x < y`
    <BLANKLINE>
    Mismatched elements: 1 / 1 (100%)
    Max absolute difference among violations: 1
    Max relative difference among violations: 1.
    x: array(0)
    y: array(-1)
    >>> jax.jit(safe_log)(-1)
    Array(nan, dtype=float32, weak_type=True)

Installation
------------

.. code-block:: bash

    $ pip install jax-ifnt

Relationship to chex
--------------------

DeepMind's `chex <https://github.com/google-deepmind/chex>`_ provides similar, often complementary, assertions. While chex requires runtime assertions to be "functionalized" with :code:`chex.chexify`, ifnt will skip assertions in traced code. This facilitates, for example, verifying that indices are not out of bounds.
