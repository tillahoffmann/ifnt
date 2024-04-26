from jax import numpy as jnp
from jax.scipy import stats
import numpy as np
from .util import skip_if_traced


assert_array_less = skip_if_traced(np.testing.assert_array_less)
assert_allclose = skip_if_traced(np.testing.assert_allclose)


def assert_shape(x: jnp.ndarray, shape: tuple) -> None:
    """
    Assert an array has the desired shape.

    Args:
        x: Array to check.
        shape: Desired shape.

    Returns:
        Input array :code:`x`.

    Example:

        >>> ifnt.testing.assert_shape(jnp.arange(3), (3,))
        >>> ifnt.testing.assert_shape(jnp.arange(3), ())
        Traceback (most recent call last):
        ...
        AssertionError: Expected shape `()` but got `(3,)`.
    """
    actual = x.shape
    assert actual == shape, f"Expected shape `{shape}` but got `{actual}`."


@skip_if_traced
def assert_positive_definite(x: jnp.ndarray, atol: float = 0.0) -> None:
    """
    Assert that a matrix is positive definite.

    Args:
        x: Matrix to check.
        atol: Absolute tolerance for the smallest eigenvalue, i.e., the smallest
            eigenvalue must not be smaller than :code:`-atol`.

    Example:

        >>> x = jnp.eye(3)
        >>> ifnt.testing.assert_positive_definite(x)
        >>> ifnt.testing.assert_positive_definite(-x)
        Traceback (most recent call last):
        ...
        AssertionError: Matrices are not positive definite: min(eigenvalues) = -1.0.
    """
    min_eigvals = jnp.linalg.eigvalsh(x).min(axis=-1)
    if (min_eigvals < -atol).any():
        raise AssertionError(
            f"Matrices are not positive definite: min(eigenvalues) = {min_eigvals}."
        )


@skip_if_traced
def assert_samples_close(
    samples: jnp.ndarray, expected: jnp.ndarray, q: float = 0.001
) -> None:
    """
    Assert that i.i.d samples are close to a target using a normal approximation of the
    sample mean. If samples are not scalars, a
    `Bonferroni correction <https://en.wikipedia.org/wiki/Bonferroni_correction>`__ is
    applied to the tail probability :code:`q` to avoid incorrect rejection of the null
    hypothesis because of sampling noise.

    Args:
        samples: Samples with shape :code:`(n_samples, ...)`.
        expected: Target with shape :code:`(...)`.
        q: Tail probability where to reject the null that :code:`expected` is the mean
            of the distribution that generated :code:`samples`.

    Example:

        >>> samples = jax.random.normal(jax.random.key(7), (1000,))
        >>> ifnt.testing.assert_samples_close(samples, jnp.zeros(()))
        >>> ifnt.testing.assert_samples_close(samples, jnp.ones(()))
        Traceback (most recent call last):
        ...
        AssertionError: Sample mean 0.01749... with standard error 0.030881... is not
        consistent with the expected value 1.0 (z-score = 31.81521...).
    """
    size = samples.shape[0]
    assert samples.shape[1:] == expected.shape
    mean = samples.mean(axis=0)
    stderr = samples.std(axis=0) / jnp.sqrt(size - 1)

    # We consider the ratio between the standard error and target variable. If this
    # ratio is large, then we can't test the quantity well because it's consistent with
    # noise. We exclude zeros because the notion of a relative error isn't well-defined.
    fltr = expected != 0
    scale = stderr[fltr] / expected[fltr]
    n_weak = jnp.sum(scale > 3)
    assert n_weak == 0

    z = (expected - mean) / stderr
    # Compute p-value and apply Bonferroni correction (see
    # https://en.wikipedia.org/wiki/Bonferroni_correction).
    p = stats.norm.cdf(z)
    p = expected.size * jnp.minimum(p, 1 - p)
    if (p < q).any():
        raise AssertionError(
            f"Sample mean {mean} with standard error {stderr} is not consistent with "
            f"the expected value {expected} (z-score = {z})."
        )


@skip_if_traced
def assert_allfinite(x: jnp.ndarray) -> None:
    """
    Assert all elements are finite.

    Args:
        x: Array to check.

    Example:

        >>> x = jnp.arange(3)
        >>> ifnt.testing.assert_allfinite(x)
        >>> ifnt.testing.assert_allfinite(jnp.log(x))
        Traceback (most recent call last):
        ...
        AssertionError: Array with shape `(3,)` has 1 non-finite elements.
    """
    finite = jnp.isfinite(x)
    if not finite.all():
        n_not_finite = x.size - finite.sum()
        raise AssertionError(
            f"Array with shape `{x.shape}` has {n_not_finite} non-finite elements."
        )
