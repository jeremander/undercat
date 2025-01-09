"""Library implementing the "Reader" functor."""

from __future__ import annotations

from typing import Callable, Generic, NamedTuple, TypeVar


__version__ = '0.1.0'


S = TypeVar('S')
A = TypeVar('A')
B = TypeVar('B')


class Reader(NamedTuple, Generic[S, A]):
    """Class that wraps a function func : S -> A."""
    func: Callable[[S], A]

    def __call__(self, val: S) -> A:
        """Calls the wrapped function."""
        return self.func(val)

    def map(self, func: Callable[[A], B]) -> Reader[S, B]:
        """Left-composes a function onto the wrapped function, returning a new Reader."""
        return Reader(lambda val: func(self.func(val)))


def map(func: Callable[[A], B], reader: Reader[S, A]) -> Reader[S, B]:  # noqa: A001
    """Left composes a function onto a Reader, returning a new Reader."""
    return reader.map(func)

# lots of binary operations
# all/any/sum/prod/reduce
