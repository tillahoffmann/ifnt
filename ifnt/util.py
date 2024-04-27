import contextlib
import functools
from jax.core import Tracer
from jax import numpy as jnp
import numpy as np
from typing import Any, Callable, TypeVar


def is_traced(*xs: Any) -> bool:
    """
    Return if any of the arguments are traced.

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
    """
    return any(isinstance(x, Tracer) for x in xs)


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
    """

    ACTIVE = True

    def __init__(self, x) -> None:
        self.x = x

    def __getitem__(self, index):
        # Get the underlying array if `x` is an index helper obtained using `x.at`.
        array: jnp.ndarray = getattr(self.x, "array", self.x)
        if not is_traced(array) and self.ACTIVE:
            # Create a dummy array with the right shape.
            shape = array.shape
            strides = (0,) * array.ndim
            dummy = np.lib.stride_tricks.as_strided(
                np.empty_like(array, shape=(1,)), shape, strides
            )
            dummy[index]

        return self.x[index]

    @classmethod
    @contextlib.contextmanager
    def set_active(cls, active: bool):
        """
        Activate or deactivate all index guards using a context manager.

        Args:
            active: If index guards are active.

        Example:

            >>> x = jnp.arange(3)
            >>> guarded = ifnt.index_guard(x)
            >>> guarded[7]
            Traceback (most recent call last):
            ...
            IndexError: index 7 is out of bounds for axis 0 with size 3
            >>> with ifnt.index_guard.set_active(False):
            ...     guarded[7]
            Array(2, dtype=int32)
            >>> guarded[7]
            Traceback (most recent call last):
            ...
            IndexError: index 7 is out of bounds for axis 0 with size 3
        """
        outer = cls.ACTIVE
        cls.ACTIVE = active
        yield
        cls.ACTIVE = outer


def broadcast_over_dict(func: F) -> F:
    """
    Broadcast a function over values of a dictionary.
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
