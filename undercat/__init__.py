"""Library implementing the "Reader" functor."""

from __future__ import annotations

import builtins
from collections.abc import Iterable
from dataclasses import dataclass
import functools
import operator as ops
from typing import Any, Callable, Generic, NoReturn, Optional, TypeVar


__version__ = '0.1.0'


S = TypeVar('S')
A = TypeVar('A')
B = TypeVar('B')


@dataclass(frozen=True, repr=False)
class Reader(Generic[S, A]):
    """Class that wraps a function func : S -> A."""
    func: Callable[[S], A]

    @classmethod
    def const(cls, val: A) -> Reader[S, A]:
        """Given a value, returns a Reader that is a constant function returning that value."""
        return Reader(lambda _: val)

    @classmethod
    def mktuple(cls, *readers: Reader[S, A]) -> Reader[S, tuple[A, ...]]:
        """Converts multiple Readers into a single Reader that produces a tuple of values, one for each of the input Readers."""
        return Reader(lambda val: tuple(reader(val) for reader in readers))

    def __call__(self, val: S) -> A:
        """Calls the wrapped function."""
        return self.func(val)

    def map(self, func: Callable[[A], B]) -> Reader[S, B]:
        """Left-composes a function onto the wrapped function, returning a new Reader."""
        return Reader(lambda val: func(self(val)))

    def map_binary(self, operator: Callable[[A, A], B], other: Reader[S, A]) -> Reader[S, B]:
        """Given a binary operator and another Reader, returns a new Reader that applies the operator to the output of this Reader and the other Reader."""
        return Reader(lambda val: operator(self(val), other(val)))

    # ARITHMETIC OPERATORS

    def __add__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map_binary(ops.add, other)

    def __sub__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map_binary(ops.sub, other)

    def __mul__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map_binary(ops.mul, other)

    def __truediv__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map_binary(ops.truediv, other)

    def __mod__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map_binary(ops.mod, other)

    def __floordiv__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map_binary(ops.floordiv, other)

    def __pow__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map_binary(ops.pow, other)

    def __matmul__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map_binary(ops.matmul, other)

    def __neg__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map(ops.neg)  # type: ignore[arg-type]

    def __pos__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map(ops.pos)  # type: ignore[arg-type]

    def __invert__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map(ops.inv)  # type: ignore[arg-type]

    # LOGICAL OPERATORS

    def __and__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map_binary(ops.and_, other)

    def __or__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map_binary(ops.or_, other)

    def __xor__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map_binary(ops.xor, other)

    # COMPARISON OPERATORS

    def __lt__(self, other: Reader[S, A]) -> Reader[S, bool]:
        return self.map_binary(ops.lt, other)  # type: ignore[arg-type]

    def __le__(self, other: Reader[S, A]) -> Reader[S, bool]:
        return self.map_binary(ops.le, other)  # type: ignore[arg-type]

    def __eq__(self, other: Any) -> NoReturn:
        # forbid equality operator since it could ambiguously be assumed to return a bool or a Reader
        raise NotImplementedError

    def __ne__(self, other: Any) -> NoReturn:
        raise NotImplementedError

    def __ge__(self, other: Reader[S, A]) -> Reader[S, bool]:
        return self.map_binary(ops.ge, other)  # type: ignore[arg-type]

    def __gt__(self, other: Reader[S, A]) -> Reader[S, bool]:
        return self.map_binary(ops.gt, other)  # type: ignore[arg-type]



def const(val: A) -> Reader[S, A]:
    """Given a value, returns a Reader that is a constant function returning that value."""
    return Reader.const(val)


def mktuple(*readers: Reader[S, A]) -> Reader[S, tuple[A, ...]]:
    """Converts multiple Readers into a single Reader that produces a tuple of values, one for each of the input Readers."""
    return Reader.mktuple(*readers)


def map(func: Callable[[A], B], reader: Reader[S, A]) -> Reader[S, B]:  # noqa: A001
    """Left composes a function onto a Reader, returning a new Reader."""
    return reader.map(func)


##############
# REDUCTIONS #
##############

def reduce(readers: Iterable[Reader[S, A]], operator: Callable[[A, A], A], initial: Optional[A] = None) -> Reader[S, A]:
    """Given a sequence of Readers and a binary operator, produces a new Reader that reduces the operator over the values produced by the input Readers.
    An initial value can optionally be provided to handle the case where an empty sequence is acted on."""
    if initial is None:
        func = functools.partial(functools.reduce, operator)
    else:
        func = functools.partial(functools.reduce, operator, initial=initial)
    return mktuple(*readers).map(func)


def all(readers: Iterable[Reader[S, A]]) -> Reader[S, bool]:  # noqa: A001
    """Given a sequence of Readers, produces a new Reader that evaluates the `all` function over the values output by the Readers."""
    return reduce(readers, ops.and_, initial=True)  # type: ignore[arg-type]


def any(readers: Iterable[Reader[S, A]]) -> Reader[S, bool]:  # noqa: A001
    """Given a sequence of Readers, produces a new Reader that evaluates the `any` function over the values output by the Readers."""
    return reduce(readers, ops.or_, initial=False)  # type: ignore[arg-type]


def sum(readers: Iterable[Reader[S, A]], start: Optional[A] = None) -> Reader[S, A]:  # noqa: A001
    """Given a sequence of Readers, produces a new Reader that evaluates the sum of the values output by the Readers."""
    return reduce(readers, ops.add, initial=start)


def prod(readers: Iterable[Reader[S, A]], start: Optional[A] = None) -> Reader[S, A]:
    """Given a sequence of Readers, produces a new Reader that evaluates the product of the values output by the Readers."""
    return reduce(readers, ops.mul, initial=start)


def min(readers: Iterable[Reader[S, A]], default: Optional[A] = None) -> Reader[S, A]:  # noqa: A001
    """Given a sequence of Readers, produces a new Reader that evaluates the minimum of the values output by the Readers."""
    return reduce(readers, builtins.min, initial=default)  # type: ignore[arg-type]


def max(readers: Iterable[Reader[S, A]], default: Optional[A] = None) -> Reader[S, A]:  # noqa: A001
    """Given a sequence of Readers, produces a new Reader that evaluates the maximum of the values output by the Readers."""
    return reduce(readers, builtins.max, initial=default)  # type: ignore[arg-type]


# __contains__
# __getitem__
# bold: override __getattr__ for item access
