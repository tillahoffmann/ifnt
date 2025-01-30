"""
Functions are primarily tested in doctests. This test suite only tests edge cases.
"""

import functools
import ifnt
import jax
from jax import numpy as jnp
from jax import scipy as jsp
import pytest
from time import sleep
from typing import Callable, Type


def _access_guarded_array(x, idx):
    ifnt.index_guard(x)[idx]


@pytest.mark.parametrize(
    "func, args, kwargs, error",
    [
        (ifnt.testing.assert_allfinite, (1.0,), {}, None),
        (ifnt.testing.assert_allfinite, (jnp.nan,), {}, AssertionError),
        (ifnt.testing.assert_allfinite, ({"a": 1, "b": 3},), {}, None),
        (ifnt.testing.assert_allfinite, ({"a": 1, "b": jnp.nan},), {}, AssertionError),
        (_access_guarded_array, (jnp.zeros(3), 4), {}, IndexError),
    ],
)
def test_ifnt(
    func: Callable, args: tuple, kwargs: dict, error: Type[Exception]
) -> None:
    # Check the call fails as expected.
    if error:
        with pytest.raises(error):
            func(*args, **kwargs)
    else:
        func(*args, **kwargs)

    # Jitted should always pass (but may lead to undefined behavior).
    jax.jit(func)(*args, **kwargs)


def test_disable() -> None:
    with pytest.raises(AssertionError):
        ifnt.testing.assert_allclose(1, 2)
    assert ifnt.util.IS_ENABLED
    with ifnt.disable():
        assert not ifnt.util.IS_ENABLED
        ifnt.testing.assert_allclose(1, 2)
    assert ifnt.util.IS_ENABLED
    with pytest.raises(AssertionError):
        ifnt.testing.assert_allclose(1, 2)


def test_assert_samples_close_weak() -> None:
    with pytest.raises(ValueError, match="Consider increasing the sample size"):
        ifnt.testing.assert_samples_close(jnp.arange(3), 0.001)
    with pytest.warns(match="Consider increasing the sample size"):
        ifnt.testing.assert_samples_close(jnp.arange(3), 0.001, on_weak="warn")
    ifnt.testing.assert_samples_close(jnp.arange(3), 0.001, on_weak="ignore")


def test_assert_samples_close_tol() -> None:
    samples = ifnt.random.JaxRandomState(17).normal((1000,)) * 1e-6
    ifnt.testing.assert_samples_close(samples, 0)
    with pytest.raises(AssertionError):
        ifnt.testing.assert_samples_close(samples, 1e-3)
    ifnt.testing.assert_samples_close(samples, 1e-3, atol=1e-2)


def test_print(capsys: pytest.CaptureFixture) -> None:
    def target(x):
        ifnt.print("hello", x)
        return x + 3

    assert target(1) == 4
    out, _ = capsys.readouterr()
    assert out == "hello 1\n"

    jitted = jax.jit(target)
    assert jitted(1) == 4
    out, _ = capsys.readouterr()
    assert not out

    assert target(2) == 5
    out, _ = capsys.readouterr()
    assert out == "hello 2\n"


def test_toeplitz() -> None:
    toeplitz = (
        jsp.linalg.toeplitz(jnp.arange(3), -jnp.arange(5))
        + jnp.arange(7)[:, None, None]
    )
    is_toeplitz = ifnt.testing.is_toeplitz(toeplitz)
    assert is_toeplitz.shape == (7,)
    assert is_toeplitz.all()
    ifnt.testing.assert_toeplitz(toeplitz)

    not_toeplitz = toeplitz.at[0, 1].set(0)
    with pytest.raises(AssertionError, match="Matrix is not Toeplitz."):
        ifnt.testing.assert_toeplitz(not_toeplitz)


def test_circulant() -> None:
    circulant = (
        jsp.linalg.toeplitz(jnp.array([2, 1, 0, 1])) + jnp.arange(7)[:, None, None]
    )
    is_circulant = ifnt.testing.is_circulant(circulant)
    assert is_circulant.shape == (7,)
    assert is_circulant.all()
    ifnt.testing.assert_circulant(circulant)

    toeplitz_not_circulant = (
        jsp.linalg.toeplitz(jnp.arange(3)) + jnp.arange(7)[:, None, None]
    )
    with pytest.raises(AssertionError, match="Matrix is not circulant."):
        ifnt.testing.assert_circulant(toeplitz_not_circulant)


@pytest.mark.parametrize("jit_value", [False, True])
@pytest.mark.parametrize("jit_index", [False, True])
def test_index_guard_arg_jitted(jit_value: bool, jit_index: bool) -> None:
    value = jnp.arange(9)
    idx = jnp.arange(3, 7)

    def fn(value, idx):
        return ifnt.index_guard(value)[idx]

    kwargs = {"value": value, "idx": idx}
    if not jit_value:
        fn = functools.partial(fn, value=kwargs.pop("value"))
    if not jit_index:
        fn = functools.partial(fn, idx=kwargs.pop("idx"))
    ifnt.testing.assert_allclose(jax.jit(fn)(**kwargs), idx)


def test_time_based_seed() -> None:
    # Try twice because we might just be at the boundary of a second.
    close = False
    for _ in range(2):
        x1 = ifnt.random.JaxRandomState().normal()
        x2 = ifnt.random.JaxRandomState().normal()
        if jnp.allclose(x1, x2):
            close = True
            break
    assert close

    # Sleep and try again to get a different value.
    sleep(1.1)
    x3 = ifnt.random.JaxRandomState().normal()
    assert not jnp.allclose(x1, x3)
