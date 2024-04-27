"""
Functions are primarily tested in doctests. This test suite only tests edge cases.
"""

import ifnt
import jax
from jax import numpy as jnp
import pytest
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
