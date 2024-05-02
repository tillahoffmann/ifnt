import functools
import inspect
from jax import numpy as jnp
from jax import random
from typing import Callable, TypeVar
from .util import is_traced


C = TypeVar("C", bound=Callable[..., jnp.ndarray])


def _wrap_random(func: C) -> C:
    signature = inspect.signature(func)
    for param in signature.parameters:
        assert (
            param == "key"
        ), f"The first parameter must be `key` but got `{param}` for `{func}`."
        break

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

    ball = _wrap_random(random.ball)
    bernoulli = _wrap_random(random.bernoulli)
    beta = _wrap_random(random.beta)
    binomial = _wrap_random(random.binomial)
    bits = _wrap_random(random.bits)
    categorical = _wrap_random(random.categorical)
    cauchy = _wrap_random(random.cauchy)
    chisquare = _wrap_random(random.chisquare)
    choice = _wrap_random(random.choice)
    dirichlet = _wrap_random(random.dirichlet)
    double_sided_maxwell = _wrap_random(random.double_sided_maxwell)
    exponential = _wrap_random(random.exponential)
    f = _wrap_random(random.f)
    fold_in = _wrap_random(random.fold_in)
    gamma = _wrap_random(random.gamma)
    generalized_normal = _wrap_random(random.generalized_normal)
    geometric = _wrap_random(random.geometric)
    gumbel = _wrap_random(random.gumbel)
    laplace = _wrap_random(random.laplace)
    loggamma = _wrap_random(random.loggamma)
    logistic = _wrap_random(random.logistic)
    lognormal = _wrap_random(random.lognormal)
    maxwell = _wrap_random(random.maxwell)
    multivariate_normal = _wrap_random(random.multivariate_normal)
    normal = _wrap_random(random.normal)
    orthogonal = _wrap_random(random.orthogonal)
    pareto = _wrap_random(random.pareto)
    permutation = _wrap_random(random.permutation)
    poisson = _wrap_random(random.poisson)
    rademacher = _wrap_random(random.rademacher)
    randint = _wrap_random(random.randint)
    rayleigh = _wrap_random(random.rayleigh)
    t = _wrap_random(random.t)
    triangular = _wrap_random(random.triangular)
    truncated_normal = _wrap_random(random.truncated_normal)
    uniform = _wrap_random(random.uniform)
    wald = _wrap_random(random.wald)
    weibull_min = _wrap_random(random.weibull_min)
