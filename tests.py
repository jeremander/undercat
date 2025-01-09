import operator as ops

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


class Vec(tuple[float]):
    """Class for a vector of floats that implements uses the dot product as the @ operator."""

    def __new__(cls, *args):
        return super().__new__(cls, args)

    def __matmul__(self, other):
        return sum(xi * yi for (xi, yi) in zip(self, other))


@pytest.mark.parametrize(['reader', 'input_val', 'output_val'], [
    # Reader
    (r_square, 3, 9),
    (r_add_one, 0, 1),
    # const
    (uc.const(5), 0, 5),
    (uc.const(5), 1, 5),
    # mktuple
    (uc.mktuple(uc.const(1), uc.const(2)), 0, (1, 2)),
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
    assert r_square == r_square
    assert not (r_square != r_square)
    assert uc.const(1) is not uc.const(2)
    assert uc.const(1) != uc.const(2)
    assert not (uc.const(1) == uc.const(2))
    assert uc.const(1) is not uc.const(1)
    assert uc.const(1) != uc.const(1)
    assert not (uc.const(1) == uc.const(1))
