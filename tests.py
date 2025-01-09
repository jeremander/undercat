import pytest

import undercat as uc
from undercat import Reader


def square(x):
    return x ** 2

def add_one(x):
    return x + 1


@pytest.mark.parametrize(['reader', 'input_val', 'output_val'], [
    (Reader(square), 3, 9),
    (Reader(square).map(add_one), 3, 10),
    (uc.map(add_one, Reader(square)), 3, 10),
    (Reader(add_one).map(square), 3, 16),
    (uc.map(square, Reader(add_one)), 3, 16),
])
def test_reader(reader, input_val, output_val):
    """Tests that a (reader, input) pair produces what we expect."""
    assert reader(input_val) == output_val
