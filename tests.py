from collections import namedtuple
from dataclasses import dataclass
import operator as ops
import re
from typing import Any, NamedTuple

import pytest

import undercat as uc
from undercat import Reader


def square(x):
    return x ** 2

r_square = Reader(square)

def add_one(x):
    return x + 1

r_add_one = Reader(add_one)

r_id = Reader(lambda val: val)
r_not = Reader(ops.not_)
r_item1 = uc.itemgetter(1)
r_attr = uc.attrgetter('attr')


class Vec(tuple[float]):
    """Class for a vector of floats that implements uses the dot product as the @ operator."""

    def __new__(cls, *args):
        return super().__new__(cls, args)

    def __matmul__(self, other):
        return sum(xi * yi for (xi, yi) in zip(self, other))


@dataclass
class Obj:
    attr: Any

r_attr_typed = uc.attrgetter('attr', type=Obj)

class Outer:
    def __init__(self, val1):
        self.val1 = val1

@dataclass
class OuterDC:
    val1: Obj
    val2: int

class OuterTypedNT(NamedTuple):
    val1: Obj
    val2: int

OuterUntypedNT = namedtuple('OuterUntypedNT', ['val1', 'val2'])


@pytest.mark.parametrize(['reader', 'input_val', 'output_val'], [
    # Reader
    (r_square, 3, 9),
    (r_add_one, 0, 1),
    # const
    (uc.const(5), 0, 5),
    (uc.const(5), 1, 5),
    # itemgetter
    (r_item1, 3, TypeError('not subscriptable')),
    (r_item1, [], IndexError('out of range')),
    (r_item1, [1, 2, 3], 2),
    (uc.itemgetter('key'), {}, KeyError('key')),
    (uc.itemgetter('key'), {'key': 3}, 3),
    # attrgetter
    (r_attr, 3, AttributeError("'int' object has no attribute 'attr'")),
    (r_attr_typed, 3, AttributeError("'int' object has no attribute 'attr'")),
    (r_attr, Obj(3), 3),
    (r_attr_typed, Obj(3), 3),
    (uc.attrgetter('fake'), Obj(3), AttributeError("'Obj' object has no attribute 'fake'")),
    (uc.attrgetter('attr.attr'), Obj(Obj(3)), 3),
    (uc.attrgetter('fake', type=Outer), Outer(Obj(3)), AttributeError("'Outer' object has no attribute 'fake'")),
    (uc.attrgetter('val1.attr', type=Outer), Outer(Obj(3)), 3),
    (uc.attrgetter('val1.fake', type=Outer), Outer(Obj(3)), AttributeError("'Obj' object has no attribute 'fake'")),
    (uc.attrgetter('val1.attr', type=OuterDC), OuterDC(Obj(3), 1), 3),
    (uc.attrgetter('val2.attr', type=OuterDC), OuterDC(Obj(3), 1), AttributeError("'int' object has no attribute 'attr'")),
    (uc.attrgetter('val1.attr', type=OuterTypedNT), OuterDC(Obj(3), 1), 3),
    (uc.attrgetter('val1.attr', type=OuterUntypedNT), OuterDC(Obj(3), 1), 3),
    (uc.attrgetter('val1.fake', type=OuterUntypedNT), OuterDC(Obj(3), 1), AttributeError("'Obj' object has no attribute 'fake'")),
    # make_tuple
    (uc.make_tuple(uc.const(1), uc.const(2)), 0, (1, 2)),
    # map
    (r_square.map(add_one), 3, 10),
    (uc.map(add_one, r_square), 3, 10),
    (r_add_one.map(square), 3, 16),
    (uc.map(square, r_add_one), 3, 16),
    (uc.const(5).map(square), 0, 25),
    # map_binary
    (r_add_one.map_binary(ops.add, r_square), 3, 13),
    # arithmetic operators
    (-r_square, 3, -9),
    (+r_square, 3, 9),
    (~r_square, 3, -10),
    (~uc.const(True), 100, -2),  # ~True is -2
    (r_square + r_add_one, 3, 13),
    (r_square - r_add_one, 3, 5),
    (r_square * r_add_one, 3, 36),
    (r_square / r_add_one, 3, 2.25),
    (r_square % r_add_one, 3, 1),
    (r_square // r_add_one, 3, 2),
    (r_square ** r_add_one, 3, 9 ** 4),
    (uc.const(Vec(1, 2)) @ uc.const(Vec(3, 4)), None, 11),
    # logical operators
    (uc.const(True).truthy(), 0, True),
    (uc.const(False).truthy(), 1, False),
    (uc.const(True).falsy(), 0, False),
    (uc.const(False).falsy(), 0, True),
    (r_id.falsy(), True, False),
    (r_id.falsy(), False, True),
    (r_not.falsy(), True, True),
    (r_not.falsy(), False, False),
    (uc.const(3) & uc.const(2), 0, 2),
    (uc.const(3) and uc.const(2), 0, 2),
    (uc.const(True) & uc.const(False), 0, False),
    (uc.const(True) & uc.const(True), 0, True),
    (uc.const(True) and uc.const(False), 0, False),
    (uc.const(True) and uc.const(True), 0, True),
    (r_not and uc.const(True), False, True),
    (r_not and r_not, False, True),
    (uc.const(3) | uc.const(2), 0, 3),
    (uc.const(3) or uc.const(2), 0, 3),
    (uc.const(True) | uc.const(False), 0, True),
    (uc.const(False) | uc.const(False), 0, False),
    (uc.const(True) or uc.const(False), 0, True),
    (uc.const(False) or uc.const(False), 0, False),
    (uc.const(3) ^ uc.const(2), 0, 1),
    (uc.const(True) ^ uc.const(False), 0, True),
    (uc.const(False) ^ uc.const(False), 0, False),
    (uc.const(True) ^ uc.const(True), 0, False),
    # comparison operators
    (r_square < r_add_one, 3, False),
    (r_square < r_square, 3, False),
    (r_square <= r_add_one, 3, False),
    (r_square <= r_square, 3, True),
    (r_square >= r_add_one, 3, True),
    (r_square >= r_square, 3, True),
    (r_square > r_add_one, 3, True),
    (r_square > r_square, 3, False),
    (r_square.equals(r_square), 3, True),
    (r_square.equals(r_id), 3, False),
    (r_square.not_equals(r_square), 3, False),
    (r_square.not_equals(r_id), 3, True),
    # other operators
    (r_id.contains('a'), 3, TypeError('not iterable')),
    (r_id.contains('a'), 'abc', True),
    (r_id.contains('a'), 'bc', False),
    (r_id.contains('a'), ['a'], True),
    (r_id.contains('a'), ['ab'], False),
    (r_id.getitem(0), 3, TypeError('not subscriptable')),
    (r_id.getitem(0), [], IndexError('out of range')),
    (r_id.getitem(0), [1, 2, 3], 1),
    (r_id.getitem(slice(None, 2)), 3, TypeError('not subscriptable')),
    (r_id.getitem(slice(None, 2)), [], []),
    (r_id.getitem(slice(None, 2)), [1, 2, 3], [1, 2]),
    (r_id.getitem('key'), 3, TypeError('not subscriptable')),
    (r_id.getitem('key'), [], TypeError('list indices must be integers or slices')),
    (r_id.getitem('key'), {}, KeyError('key')),
    (r_id.getitem('key'), {'key': 'val'}, 'val'),
    (r_id.getattr('attr'), 3, AttributeError("'int' object has no attribute 'attr'")),
    (r_id.getattr('attr'), Obj(3), 3),
    (r_id.getattr('attr', None), Obj(3), 3),
    (r_id.getattr('other', None), Obj(3), None),
    (r_id.getattr('attr'), Obj(Obj(3)), Obj(3)),
    (r_id.getattr('attr').getattr('attr'), Obj(3), AttributeError("'int' object has no attribute 'attr'")),
    (r_id.getattr('attr').getattr('attr'), Obj(Obj(3)), 3),
    (r_id.getattr('attr.attr'), 3, AttributeError("'int' object has no attribute 'attr'")),
    (r_id.getattr('attr.attr'), Obj(3), AttributeError("'int' object has no attribute 'attr'")),
    (r_id.getattr('attr.attr'), Obj(Obj(3)), 3),
    (r_id.getattr('attr.attr', None), 3, None),
    (r_id.getattr('attr.attr', None), Obj(3), None),
    (r_id.getattr('attr.attr', None), Obj(Obj(3)), 3),
    # reductions
    (uc.all([]), False, True),
    (uc.all([]), True, True),
    (uc.all([r_id, r_id, r_id]), False, False),
    (uc.all([r_id, r_id, r_id]), True, True),
    (uc.all([r_id, r_not, r_id]), False, False),
    (uc.all([r_id, r_not, r_id]), True, False),
    (uc.any([]), False, False),
    (uc.any([]), True, False),
    (uc.any([r_id, r_id, r_id]), False, False),
    (uc.any([r_id, r_id, r_id]), True, True),
    (uc.any([r_id, r_not, r_id]), False, True),
    (uc.any([r_id, r_not, r_id]), True, True),
    (uc.sum([]), None, TypeError('no initial value')),
    (uc.sum([r_id, r_square, r_add_one]), 3, 16),
    (uc.sum([uc.const(1), uc.const('2')]), None, TypeError('unsupported operand type')),
    (uc.sum([uc.const('1'), uc.const('2')]), None, '12'),
    (uc.sum([uc.const([1]), uc.const([]), uc.const([2])]), None, [1, 2]),
    (uc.prod([]), None, TypeError('no initial value')),
    (uc.prod([r_id, r_square, r_add_one]), 3, 108),
    (uc.min([]), None, TypeError('no initial value')),
    (uc.min([r_id, r_square, r_add_one]), 3, 3),
    (uc.max([]), None, TypeError('no initial value')),
    (uc.max([r_id, r_square, r_add_one]), 3, 9),
])
def test_reader(reader, input_val, output_val):
    """Tests that a (reader, input) pair produces what we expect."""
    if isinstance(output_val, Exception):  # expect an error
        with pytest.raises(type(output_val), match=str(output_val)):
            _ = reader(input_val)
    else:
        assert reader(input_val) == output_val

def test_bool_operators():
    """Tests that the `bool` and `not` operators return a bool when evaluated on a Reader.
    (This may be unexpected, as one might think they return a Reader.)"""
    for reader in [uc.const(False), uc.const(True), r_id]:
        assert bool(reader) is True
        assert (not reader) is False

def test_reader_equality():
    """Tests that the `__eq__` and `__ne__` operators return a bool when evaluated on a pair of Readers.
    (This may be unexpected, as one might think they return a Reader.)
    Equality is just identity of objects."""
    assert r_square is r_square
    assert (r_square == r_square) is True
    assert (r_square != r_square) is False
    assert uc.const(1) is not uc.const(2)
    assert (uc.const(1) != uc.const(2)) is True
    assert (uc.const(1) == uc.const(2)) is False
    assert uc.const(1) is not uc.const(1)
    assert (uc.const(1) != uc.const(1)) is True
    assert (uc.const(1) == uc.const(1)) is False

def test_invalid_operators():
    """Tests that certain operators are invalid when called on a Reader."""
    type_err = lambda match: pytest.raises(TypeError, match=match)
    # __iter__
    with type_err('not iterable'):
        _ = list(r_square)  # type: ignore[call-overload]
    # __contains__
    with type_err('not iterable'):
        _ = 3 in r_square
    with type_err('not iterable'):
        _ = 3 not in r_square
    with type_err('not iterable'):
        _ = 3 not in uc.const([1, 2, 3])
    # __getitem__
    with type_err('not subscriptable'):
        _ = r_square[1]
    with type_err('not subscriptable'):
        _ = r_square['key']
    with type_err('not subscriptable'):
        _ = r_square[:2]
    # getattr with too many args
    with type_err('getattr expected at most 2 arguments, got 3'):
        _ = r_id.getattr('attr', 1, None)
    # getattr with invalid attr string
    with type_err('attr must be a string'):
        _ = r_id.getattr(3)

@pytest.mark.parametrize(['attr', 'seg'], [
    ('', ''),
    (' ', ' '),
    ('+', '+'),
    ('a+', 'a+'),
    ('a ', 'a '),
    (' a', ' a'),
    ('1', '1'),
    ('1a', '1a'),
    ('a.b+', 'b+'),
    ('a.1b', '1b'),
    ('a.1b.c', '1b'),
])
def test_invalid_attr_name(attr, seg):
    """Tests attrgetter when the attribute name is invalid."""
    with pytest.raises(ValueError, match=re.escape(f'invalid attribute name {seg!r}')):
        _ = uc.attrgetter(attr)

def test_invalid_getattr_typed():
    """Tests attrgetter when the provided type is invalid."""
    err = "invalid field 'fake' for type 'Obj'"
    for (attr, tp) in [('fake', Obj), ('val1.fake', OuterDC), ('val1.fake', OuterTypedNT)]:
        with pytest.raises(TypeError, match=err):
            _ = uc.attrgetter(attr, type=tp)
    for tp in [OuterDC, OuterTypedNT, OuterUntypedNT]:
        with pytest.raises(TypeError, match=f"invalid field 'fake' for type '{tp.__name__}'"):
            _ = uc.attrgetter('fake', type=tp)
