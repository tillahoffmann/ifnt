import functools
import jax
from jax import numpy as jnp
from jax.scipy import stats
import numpy as np
import warnings
from .util import broadcast_over_dict, skip_if_traced


assert_array_less = skip_if_traced(np.testing.assert_array_less)
assert_allclose = skip_if_traced(np.testing.assert_allclose)
assert_array_equal = skip_if_traced(np.testing.assert_array_equal)


def assert_shape(x: jnp.ndarray, shape: tuple) -> None:
    """
    Assert an array has the desired shape.

    Args:
        ACTUAL: array to check.
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
    actual = jnp.shape(x)
    assert actual == shape, f"Expected shape `{shape}` but got `{actual}`."


@broadcast_over_dict
@functools.singledispatch
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
    samples: jnp.ndarray,
    expected: jnp.ndarray,
    q: float = 0.001,
    on_weak: str = "raise",
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> None:
    """
    Assert that i.i.d samples are close to a target using a normal approximation of the
    sample mean. If samples are not scalars, a
    `Bonferroni correction <https://en.wikipedia.org/wiki/Bonferroni_correction>`__ is
    applied to the tail probability :code:`q` to avoid incorrect rejection of the null
    hypothesis (that :code:`expected` is the mean of the distribution that generated
    :code:`samples`) because of sampling noise. The assertion also passes without
    failure if all samples are close to the target as determined by :code:`rtol` and
    :code:`atol`.

    Args:
        samples: Samples with shape :code:`(n_samples, ...)`.
        expected: Target with shape :code:`(...)`.
        q: Tail probability where to reject the null that :code:`expected` is the mean
            of the distribution that generated :code:`samples`.
        on_weak: Action if the sample size is too small to confidently reject the null
            hypothesis.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Example:

        >>> samples = jax.random.normal(jax.random.key(7), (1000,))
        >>> ifnt.testing.assert_samples_close(samples, 0.0)
        >>> ifnt.testing.assert_samples_close(samples, 1.0)
        Traceback (most recent call last):
        ...
        AssertionError: Sample mean 0.01749... with standard error 0.030881... is not
        consistent with the expected value 1.0 (z-score = 31.81521...).
        >>> ifnt.testing.assert_samples_close(samples, -1.0)
        Traceback (most recent call last):
        ...
        AssertionError: Sample mean 0.01749... with standard error 0.03088... is not
        consistent with the expected value -1.0 (z-score = -32.94792...).
    """
    allowed_actions = {"raise", "warn", "ignore"}
    if on_weak not in allowed_actions:
        raise ValueError(f"`on_weak` must be one of {allowed_actions}.")

    size = samples.shape[0]
    assert_shape(expected, samples.shape[1:])
    mean = samples.mean(axis=0)
    stderr = samples.std(axis=0) / jnp.sqrt(size - 1)

    # We consider the ratio between the standard error and target variable. If this
    # ratio is large, then we can't test the quantity well because it's consistent with
    # noise. We exclude zeros because the notion of a relative error isn't well-defined.
    ratio = jnp.where(expected, jnp.abs(expected / stderr), float("inf"))
    if jnp.any(ratio < 1 / 3):
        message = (
            "The target value is small compared with the standard error of the "
            f"samples; min(expected / stderr) = {ratio.min():.3g}. Consider increasing "
            "the sample size or set `on_weak = 'ignore'`."
        )
        if on_weak == "raise":
            raise ValueError(message)
        elif on_weak == "warn":
            warnings.warn(message)

    z = (expected - mean) / stderr
    # Compute p-value and apply Bonferroni correction (see
    # https://en.wikipedia.org/wiki/Bonferroni_correction).
    p = stats.norm.cdf(z)
    p = jnp.size(expected) * jnp.minimum(p, 1 - p)
    within_tol = jnp.isclose(samples, expected, rtol=rtol, atol=atol).all(axis=0)
    if ((p < q) & ~within_tol).any():
        raise AssertionError(
            f"Sample mean {mean} with standard error {stderr} is not consistent with "
            f"the expected value {expected} (z-score = {z})."
        )


@broadcast_over_dict
@functools.singledispatch
@skip_if_traced
def assert_allfinite(x: jnp.ndarray) -> None:
    """
    Assert all elements are finite.

    Args:
        x: Array or a dictionary of arrays to check.

    Example:

        >>> x = jnp.arange(3)
        >>> ifnt.testing.assert_allfinite(x)
        >>> ifnt.testing.assert_allfinite(jnp.log(x))
        Traceback (most recent call last):
        ...
        AssertionError: Array with shape `(3,)` has 1 non-finite elements.
        >>> ifnt.testing.assert_allfinite({"a": x, "b": jnp.log(x)})
        Traceback (most recent call last):
        ...
        AssertionError: <function assert_allfinite at 0x...> failed for key `b`. Array
        with shape `(3,)` has 1 non-finite elements.
    """
    finite = jnp.isfinite(x)
    if not finite.all():
        n_not_finite = jnp.size(x) - finite.sum()
        raise AssertionError(
            f"Array with shape `{jnp.shape(x)}` has {n_not_finite} non-finite elements."
        )


def is_toeplitz(x: jnp.ndarray, rtol: float = 1e-7, atol: float = 0.0) -> jnp.ndarray:
    """
    Determine if a matrix is Toeplitz, i.e., has the same value on all diagonals.

    Args:
        x: Matrix or batch of matrices to check with shape (..., n, m).
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        Boolean array with shape (...) indicating whether :code:`x` is Toeplitz.

    Example:

        >>> t = jsp.linalg.toeplitz(jnp.arange(3))
        >>> t
        Array([[0, 1, 2],
               [1, 0, 1],
               [2, 1, 0]], dtype=int32)
        >>> ifnt.testing.is_toeplitz(t)
        Array(True, dtype=bool)
        >>> a = t.at[0, 0].set(7)
        >>> a
        Array([[7, 1, 2],
               [1, 0, 1],
               [2, 1, 0]], dtype=int32)
        >>> ifnt.testing.is_toeplitz(a)
        Array(False, dtype=bool)
    """
    assert x.ndim >= 2, "Input must have at least two dimensions."
    batch_shape = x.shape[:-2]
    result = jnp.ones(batch_shape, dtype=bool)
    return jax.lax.fori_loop(
        0,
        x.shape[-2] - 1,
        lambda i, result: result
        & jnp.allclose(x[..., i, :-1], x[..., i + 1, 1:], rtol, atol),
        result,
    )


@skip_if_traced
def assert_toeplitz(x: jnp.ndarray, rtol: float = 1e-7, atol: float = 0.0) -> None:
    """
    Assert that a matrix is Toeplitz, i.e., has the same value on all diagonals.

    Args:
        x: Matrix or batch of matrices to check.
        rtol: Relative tolerance.
        atol: Absolute tolerance.


    Example:

        >>> t = jsp.linalg.toeplitz(jnp.arange(3))
        >>> t
        Array([[0, 1, 2],
               [1, 0, 1],
               [2, 1, 0]], dtype=int32)
        >>> ifnt.testing.assert_toeplitz(t)
        >>> a = t.at[0, 0].set(7)
        >>> a
        Array([[7, 1, 2],
               [1, 0, 1],
               [2, 1, 0]], dtype=int32)
        >>> ifnt.testing.assert_toeplitz(a)
        Traceback (most recent call last):
        ...
        AssertionError:
        Arrays are not equal
        Matrix is not Toeplitz.
        Mismatched elements: 1 / 1 (100%)
        ACTUAL: array(False)
        DESIRED: array(True)
    """
    assert_array_equal(is_toeplitz(x, rtol, atol), True, "Matrix is not Toeplitz.")


def is_circulant(x: jnp.ndarray, rtol: float = 1e-7, atol: float = 0.0) -> jnp.ndarray:
    """
    Determine if a matrix is circulant, i.e., has periodic boundary conditions and the
    same values on all diagonals.

    Args:
        x: Matrix or batch of matrices to check with shape (..., n, m).
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        Boolean array with shape (...) indicating whether :code:`x` is circulant.

    Example:

        >>> c = jsp.linalg.toeplitz(jnp.array([2, 1, 0, 1]))
        >>> c
        Array([[2, 1, 0, 1],
               [1, 2, 1, 0],
               [0, 1, 2, 1],
               [1, 0, 1, 2]], dtype=int32)
        >>> ifnt.testing.is_circulant(c)
        Array(True, dtype=bool)
        >>>
        >>> t = jsp.linalg.toeplitz(jnp.arange(3))
        >>> t
        Array([[0, 1, 2],
               [1, 0, 1],
               [2, 1, 0]], dtype=int32)
        >>> ifnt.testing.is_circulant(t)
        Array(False, dtype=bool)
    """
    assert x.ndim >= 2, "Input must have at least two dimensions."
    assert x.shape[-2] == x.shape[-1], "Input must be square."
    batch_shape = x.shape[:-2]
    result = jnp.ones(batch_shape, dtype=bool)
    return jax.lax.fori_loop(
        0,
        x.shape[-2] - 1,
        lambda i, result: result
        & jnp.allclose(x[..., i], jnp.roll(x[..., i + 1], -1, axis=-1), rtol, atol),
        result,
    )


@skip_if_traced
def assert_circulant(x: jnp.ndarray, rtol: float = 1e-7, atol: float = 0.0) -> None:
    """
    Assert that a matrix is circulant, i.e., has periodic boundary conditions and the
    same values on all diagonals.

    Args:
        x: Matrix or batch of matrices to check.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Example:

        >>> c = jsp.linalg.toeplitz(jnp.array([2, 1, 0, 1]))
        >>> c
        Array([[2, 1, 0, 1],
               [1, 2, 1, 0],
               [0, 1, 2, 1],
               [1, 0, 1, 2]], dtype=int32)
        >>> ifnt.testing.assert_circulant(c)
        >>>
        >>> t = jsp.linalg.toeplitz(jnp.arange(3))
        >>> t
        Array([[0, 1, 2],
               [1, 0, 1],
               [2, 1, 0]], dtype=int32)
        >>> ifnt.testing.assert_circulant(t)
        Traceback (most recent call last):
        ...
        AssertionError:
        Arrays are not equal
        Matrix is not circulant.
        Mismatched elements: 1 / 1 (100%)
        ACTUAL: array(False)
        DESIRED: array(True)
    """
    assert_array_equal(is_circulant(x, rtol, atol), True, "Matrix is not circulant.")
