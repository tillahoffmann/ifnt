📖 Reference
============

⚙️ ifnt.util
------------

:mod:`ifnt.util` includes basic functionality for constructing functions that are skipped if traced, conditioning on traced values, safe indexing, and printing non-traced tensors.

.. automodule:: ifnt.util
    :members:

🧪 ifnt.testing
---------------

:mod:`ifnt.testing` wraps common assertions from :mod:`numpy.testing` and implements stochastic tests through :func:`~ifnt.testing.assert_samples_close`.

.. automodule:: ifnt.testing
    :members:

🎲 ifnt.random
--------------

:mod:`ifnt.random` facilitates stateful random number generation to avoid repeated calls to :func:`jax.random.split`.

.. automodule:: ifnt.random
    :members:
