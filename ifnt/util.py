import builtins
import contextlib
import functools
import jax._src.basearray
from jax.core import Tracer
from jax import numpy as jnp
import numpy as np
import os
from typing import Any, Callable, TypeVar


IS_ENABLED = "IFNT_DISABLED" not in os.environ


@contextlib.contextmanager
def disable(do_disable: bool = True):
    """
    Disable all :code:`ifnt` behavior even if values are not traced.

    Args:
        do_disable: Disable :code:`ifnt` if truth-y.

    Example:

        >>> ifnt.testing.assert_allclose(1, 2)
        Traceback (most recent call last):
        ...
        AssertionError:
        Not equal to tolerance rtol=1e-07, atol=0
        <BLANKLINE>
        Mismatched elements: 1 / 1 (100%)
        Max absolute difference among violations: 1
        Max relative difference among violations: 0.5
        ACTUAL: array(1)
        DESIRED: array(2)
        >>> with ifnt.disable():
        ...     ifnt.testing.assert_allclose(1, 2)
        >>> ifnt.testing.assert_allclose(1, 2)
        Traceback (most recent call last):
        ...
        AssertionError:
        Not equal to tolerance rtol=1e-07, atol=0
        <BLANKLINE>
        Mismatched elements: 1 / 1 (100%)
        Max absolute difference among violations: 1
        Max relative difference among violations: 0.5
        ACTUAL: array(1)
        DESIRED: array(2)
    """
    global IS_ENABLED
    previous = IS_ENABLED
    IS_ENABLED = not do_disable
    yield
    IS_ENABLED = previous


def is_traced(*xs: Any) -> bool:
    """
    Return if any of the arguments are traced.

    .. warning::

        :func:`is_traced` always returns :code:`True` if the :code:`IFNT_DISABLED`
        environment variable is set or the function is called within a :func:`disable`
        context. Consequently any code that is not executed when traced is skipped.

    Args:
        xs: Value or values to check.

    Returns:
        If any of the values are traced.

    Example:

        >>> def f(x):
        ...     return ifnt.is_traced(x)
        >>> x = jnp.zeros(3)
        >>> f(x)
        False
        >>> jax.jit(f)(x)
        Array(True, dtype=bool)
        >>> with ifnt.disable():
        ...     f(x)
        True
    """
    # Early termination if ifnt is disabled.
    if not IS_ENABLED:
        return True
    # Check if any of the elements is a tracer.
    if any(isinstance(x, Tracer) for x in xs):
        return True
    # Check if any flattened element is a tracer.
    for x in xs:
        leaves, _ = jax.tree.flatten(x)
        if any(isinstance(leaf, Tracer) for leaf in leaves):
            return True
    return False


F = TypeVar("F", bound=Callable)


def skip_if_traced(func: F) -> F:
    """
    Skip a function if any of its arguments are traced. The decorated function does not
    return a value, even if the original function did.

    Args:
        func: Function to skip if any of its arguments are traced.

    Example:

        >>> @ifnt.skip_if_traced
        ... def assert_positive(x):
        ...     assert x.min() > 0
        >>>
        >>> assert_positive(-jnp.zeros(5))
        Traceback (most recent call last):
        ...
        AssertionError
        >>> jax.jit(assert_positive)(-jnp.zeros(5))
    """

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        if is_traced(*args) or is_traced(*kwargs.values()):
            return
        func(*args, **kwargs)

    return _wrapper


def raise_if_traced(func: F) -> F:
    """
    Raise an error if any of the function's arguments are traced.

    Args:
        func: Function to fail if any of its arguments are traced.

    Example:

        >>> @ifnt.raise_if_traced
        ... def multiply(x):
        ...     return 2 * x
        >>>
        >>> multiply(jnp.arange(3))
        Array([0, 2, 4], dtype=int32)
        >>> jax.jit(multiply)(jnp.arange(3))
        Traceback (most recent call last):
        ...
        RuntimeError: Cannot execute `multiply` because one or more of its arguments are
        traced.
    """

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        if is_traced(*args) or is_traced(*kwargs.values()):
            raise RuntimeError(
                f"Cannot execute `{func.__name__}` because one or more of its "
                "arguments are traced."
            )
        return func(*args, **kwargs)

    return _wrapper


class index_guard:
    """
    Safe indexing that checks out of bounds when not traced.

    Args:
        x: Array to guard.

    Example:

        >>> x = jnp.arange(3)
        >>> x[7]
        Array(2, dtype=int32)
        >>> ifnt.index_guard(x)[7]
        Traceback (most recent call last):
        ...
        IndexError: index 7 is out of bounds for axis 0 with size 3
        >>> ifnt.index_guard(x).at[1].set(7)
        Array([0, 7, 2], dtype=int32)
    """

    ACTIVE = True

    def __init__(self, x) -> None:
        self.x = x

    def __getitem__(self, index):
        # Get the underlying array if `x` is an index helper obtained using `x.at`.
        array: jnp.ndarray = getattr(self.x, "array", self.x)
        if not is_traced(array, index) and self.ACTIVE:
            # Create a dummy array with the right shape.
            shape = array.shape
            strides = (0,) * array.ndim
            dummy = np.lib.stride_tricks.as_strided(
                np.empty_like(array, shape=(1,)), shape, strides
            )
            dummy[index]

        return self.x[index]

    @property
    def at(self) -> "jax._src.basearray._IndexUpdateHelper":
        return self.__class__(self.x.at)

    def __repr__(self) -> str:
        return f"index_guard({self.x})"


def broadcast_over_dict(func: F) -> F:
    """
    Broadcast a function over values of a dictionary.

    Args:
        func: Function to broadcast.

    Example:

        >>> from functools import singledispatch
        >>> @ifnt.util.broadcast_over_dict
        ... @singledispatch
        ... def add_one(x):
        ...     return x + 1
        >>> add_one({"a": 1, "b": 2})
        {'a': 2, 'b': 3}
    """
    register = getattr(func, "register", None)
    if not register:
        raise TypeError("Function to broadcast must be a `singledispatch` function.")

    @register
    def wrapper(x: dict, *args, **kwargs):
        result = {}
        for key, value in x.items():
            try:
                result[key] = func(value, *args, **kwargs)
            except Exception as ex:
                message = f"{func} failed for key `{key}`."
                if ex.args and isinstance(ex.args[0], str):
                    ex.args = (f"{message} {ex.args[0]}",) + ex.args[1:]
                else:
                    ex.args = (message,) + ex.args
                raise

        return result

    return func


@skip_if_traced
def print(*objects, sep=" ", end="\n", file=None, flush=False) -> None:
    """
    If none of the arguments are traced, print :code:`objects` separated by :code:`sep`
    followed by :code:`end` to :code:`stdout` or a file.

    Args:
        *objects: Objects to print.
        sep: Separator between objects.
        end: End string.
        file: Stream with :code:`write` method to print objects to.
        flush: Forcibly flush the stream.
    """
    builtins.print(*objects, sep=sep, end=end, file=file, flush=flush)
