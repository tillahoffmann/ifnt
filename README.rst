ifnt
====

.. image:: https://github.com/tillahoffmann/ifnt/actions/workflows/build.yml/badge.svg
    :target: https://github.com/tillahoffmann/ifnt/actions/workflows/build.yml
.. image:: https://readthedocs.org/projects/ifnt/badge/?version=latest
    :target: https://ifnt.readthedocs.io/en/latest/?badge=latest

Execute runtime assertions, indexing checks, and more if :code:`jax` code is not traced.

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
    AssertionError: Arrays are not less-ordered
    <BLANKLINE>
    Mismatched elements: 1 / 1 (100%)
    Max absolute difference: 1
    Max relative difference: 1.
    x: array(0)
    y: array(-1)
    >>> jax.jit(safe_log)(-1)
    Array(nan, dtype=float32, weak_type=True)

ifnt comprises three modules:

- :mod:`ifnt.random` for stateful random number generation, obviating repeated :func:`jax.random.split` calls.
- :mod:`ifnt.testing` for numpy-like runtime assertions for code that is not traced.
- :mod:`ifnt.util` containing utility functions, such as decorators :func:`~ifnt.util.skip_if_traced` or :func:`~ifnt.util.raise_if_traced`.

Installation
------------

.. code-block:: bash

    $ pip install ifnt
