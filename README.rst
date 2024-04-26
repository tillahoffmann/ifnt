ifnt
====

Execute runtime assertions, indexing checks, and more if :code:`jax` code is not traced.

ifnt.util
---------

:mod:`ifnt.util` includes basic functionality for constructing functions that are skipped if traced, conditioning on traced values, and safe indexing.

.. automodule:: ifnt.util
    :members:

ifnt.testing
------------

:mod:`ifnt.testing` wraps common assertions from :mod:`numpy.testing` and implements stochastic tests through :func:`~ifnt.testing.assert_samples_close`.

.. automodule:: ifnt.testing
    :members:
