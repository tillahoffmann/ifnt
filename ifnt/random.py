import functools
from jax import numpy as jnp
from jax import random
from typing import Callable, TypeVar
from .util import is_traced


C = TypeVar("C", bound=Callable[..., jnp.ndarray])


def _wrap_random(func: C) -> C:
    @functools.wraps(func)
    def _inner(self: "JaxRandomState", *args, **kwargs):
        return func(self.get_key(), *args, **kwargs)

    return _inner


class JaxRandomState:
    """
    Utility class for sampling random variables using the JAX interface with automatic
    random state handling.

    Args:
        seed: Initial random number generator seed.

    .. warning::

        This implementation is stateful and does not support :func:`jax.jit`
        compilation.

    Example:

        >>> rng = ifnt.random.JaxRandomState(7)
        >>> rng.normal()
        Array(-1.4622004, dtype=float32)
        >>> rng.normal()
        Array(2.0224454, dtype=float32)
        >>> jax.jit(rng.normal)()
        Traceback (most recent call last):
        ...
        RuntimeError: `JaxRandomState.get_key()` does not support jit compilation.
    """

    def __init__(self, seed: int) -> None:
        self._key = random.PRNGKey(seed)

    def get_key(self) -> jnp.ndarray:
        """
        Get a random key and update the state.
        """
        self._key, key = random.split(self._key)
        if is_traced(key):
            raise RuntimeError(
                "`JaxRandomState.get_key()` does not support jit compilation."
            )
        return key

    multivariate_normal = _wrap_random(random.multivariate_normal)
    normal = _wrap_random(random.normal)
