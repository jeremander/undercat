"""Library implementing the "Reader" functor."""

from __future__ import annotations

import builtins
from collections.abc import Iterable, Sequence
from dataclasses import is_dataclass
import functools
import operator as ops
from typing import Any, Callable, Generic, Optional, TypeVar


__version__ = '0.1.1'


S = TypeVar('S')
A = TypeVar('A')
B = TypeVar('B')


####################
# HELPER FUNCTIONS #
####################


def _get_attr_type(cls: type, attr: str) -> Optional[type]:
    """Given a type, if it is a dataclass or namedtuple type, checks whether it possesses the given attribute.
    If not, raises a TypeError. Otherwise, returns the type of the attribute.
    If the input type is not a dataclass or nameduple type, returns None."""
    if isinstance(cls, type):
        if is_dataclass(cls):
            if (field := cls.__dataclass_fields__.get(attr)) is not None:
                return field.type  # type: ignore[return-value]
            raise TypeError(f'invalid field {attr!r} for type {cls.__name__!r}')
        if hasattr(cls, '_fields'):  # assume it's a namedtuple
            fields = set(cls._fields)
            if attr in fields:
                # may or may not have an annotation
                return cls.__annotations__.get(attr)
            raise TypeError(f'invalid field {attr!r} for type {cls.__name__!r}')
    return None


def _getattr_simple(attr: str) -> Callable[[Any], Any]:
    assert isinstance(attr, str)
    if (not attr.isalnum()) or attr[0].isdigit():
        raise ValueError(f'invalid attribute name {attr!r}')
    return ops.attrgetter(attr)


def _getattr_nested(attrs: Sequence[str], type: Optional[type] = None) -> Callable[[Any], Any]:
    assert len(attrs) > 0
    attr = attrs[0]
    outer = _getattr_simple(attr)
    # validate attribute statically, if applicable
    inner_type = _get_attr_type(type, attr)  # type: ignore[arg-type]
    if len(attrs) == 1:
        return outer
    # recursively call this function to get a function for the nested accesses
    inner = _getattr_nested(attrs[1:], type=inner_type)
    # left-compose inner with outer
    return lambda val: inner(outer(val))


def _getattr(attr: str, *args: Any, type: Optional[type] = None) -> Callable[[Any], Any]:
    if not isinstance(attr, str):
        raise TypeError('attr must be a string')
    attrs = attr.split('.')
    func = _getattr_nested(attrs, type=type)
    if args:
        if len(args) > 1:
            num_args = 1 + len(args)
            raise TypeError(f'getattr expected at most 2 arguments, got {num_args}')
        def _getattr(val: Any) -> Any:
            try:
                return func(val)
            except AttributeError:  # return the default
                return args[0]
        return _getattr
    else:  # no default
        return func


##########
# READER #
##########

class Reader(Generic[S, A]):
    """Class that wraps a function func : S -> A."""

    def __init__(self, func: Callable[[S], A]) -> None:
        self.func = func

    @classmethod
    def const(cls, val: A) -> Reader[S, A]:
        """Given a value, returns a Reader that is a constant function returning that value."""
        return Reader(lambda _: val)

    @classmethod
    def make_tuple(cls, *readers: Reader[S, A]) -> Reader[S, tuple[A, ...]]:
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

    def __neg__(self) -> Reader[S, A]:
        return self.map(ops.neg)  # type: ignore[arg-type]

    def __pos__(self) -> Reader[S, A]:
        return self.map(ops.pos)  # type: ignore[arg-type]

    def __invert__(self) -> Reader[S, A]:
        return self.map(ops.inv)  # type: ignore[arg-type]

    # LOGICAL OPERATORS

    def __and__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map_binary(ops.and_, other)

    def __or__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map_binary(ops.or_, other)

    def __xor__(self, other: Reader[S, A]) -> Reader[S, A]:
        return self.map_binary(ops.xor, other)

    def truthy(self) -> Reader[S, bool]:
        """Returns a Reader that evaluates the `bool` function on this Reader's output."""
        return self.map(bool)

    def falsy(self) -> Reader[S, bool]:
        """Returns a Reader that evaluates the logical negation (`not` operator) on this Reader's output."""
        return self.map(ops.not_)

    # COMPARISON OPERATORS

    def __lt__(self, other: Reader[S, A]) -> Reader[S, bool]:
        return self.map_binary(ops.lt, other)  # type: ignore[arg-type]

    def __le__(self, other: Reader[S, A]) -> Reader[S, bool]:
        return self.map_binary(ops.le, other)  # type: ignore[arg-type]

    def __ge__(self, other: Reader[S, A]) -> Reader[S, bool]:
        return self.map_binary(ops.ge, other)  # type: ignore[arg-type]

    def __gt__(self, other: Reader[S, A]) -> Reader[S, bool]:
        return self.map_binary(ops.gt, other)  # type: ignore[arg-type]

    def equals(self, other: Reader[S, A]) -> Reader[S, bool]:
        """Returns a Reader that evaluates whether the output of this Reader equals the output of the other."""
        return self.map_binary(ops.eq, other)

    def not_equals(self, other: Reader[S, A]) -> Reader[S, bool]:
        """Returns a Reader that evaluates whether the output of this Reader does not equal the output of the other."""
        return self.equals(other).falsy()

    # OTHER OPERATORS

    def contains(self, element: Any) -> Reader[S, bool]:
        """Returns a Reader returning True if the given element is in the value returned by this Reader."""
        return self.map(lambda val: element in val)  # type: ignore[operator]

    def getitem(self, index: Any) -> Reader[S, Any]:
        """Returns a Reader that returns value[index], where value is the value returned by this Reader."""
        return self.map(ops.itemgetter(index))  # type: ignore[arg-type]

    def getattr(self, attr: str, *args: Any, type: Optional[type] = None) -> Reader[S, Any]:
        """Returns a Reader that returns getattr(value, attr, [default]), where value is the value returned by this Reader.
        attr may contain multiple fields separated by '.' (example x.y.z), which will perform nested attribute retrieval.
        The second argument to this method, if present, is the default value to use if an attribute does not exist."""
        return self.map(_getattr(attr, *args, type=type))


#######################
# READER CONSTRUCTORS #
#######################


def const(val: A) -> Reader[S, A]:
    """Given a value, returns a Reader that is a constant function returning that value."""
    return Reader.const(val)


def make_tuple(*readers: Reader[S, A]) -> Reader[S, tuple[A, ...]]:
    """Converts multiple Readers into a single Reader that produces a tuple of values, one for each of the input Readers."""
    return Reader.make_tuple(*readers)


def itemgetter(index: Any) -> Reader[S, Any]:
    """Given an index object, returns a Reader that takes a value and returns returns value[index]."""
    return Reader(ops.itemgetter(index))  # type: ignore[arg-type]


def attrgetter(attr: str, *args: Any, type: Optional[type] = None) -> Reader[S, Any]:
    """Given an attribute string and optional default, returns a Reader that takes a value and returns returns getattr(value, attr, [default]).
    attr may contain multiple fields separated by '.' (example x.y.z), which will perform nested attribute retrieval."""
    return Reader(_getattr(attr, *args, type=type))


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
        func = lambda iterable: functools.reduce(operator, iterable, initial)  # type: ignore[assignment]
    return make_tuple(*readers).map(func)


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
