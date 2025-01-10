# Undercat

**Undercat** is a small Python library implementing a functional programming construct called the *Reader functor*. This pattern is particularly useful for dependency injection, composing functions, and operating on immutable, context-aware computations.

The Reader functor is ideal when you need to:

- Pass a shared dependency or context (e.g., configuration, environment) through multiple computations without explicitly threading it through function calls.
- Create pipelines of computation where each step depends on a shared state.
- Enhance code readability by abstracting common operations such as accessing attributes or combining results.

## Features

- **Function composition**: Compose computations using a clean, declarative style.
- **Attribute and item access**: Retrieve nested attributes or indexed elements with ease.
- **Operator overloading**: Perform arithmetic, logical, and comparison operations directly on Reader objects.

## Installation

`pip install undercat`

or (using [uv](https://docs.astral.sh/uv/))

`uv pip install undercat`

## Quickstart

### 1. Basic Usage

Define a `Reader` that retrieves data from a context and operates on it:

```python
import undercat as uc

class Context:
    """A context with a dict attribute."""
    def __init__(self, value):
        self.data = {'key': value}

# Define a Reader to get a nested value from a context.
get_value = uc.attrgetter('data').getitem('key')

context = Context(42)

# Execute the Reader with a context.
get_value(context)  # output: 42
```

### 2. Function Composition

`Reader`s can be composed using `.map` to transform outputs:

```python
get_value_and_increment = get_value.map(lambda x: x + 1)

# Execute the modified Reader.
get_value_and_increment(context)  # output: 43
```

### 3. Combining Readers

Combine multiple `Reader`s into one using operators:

```python
# Define a Reader that squares its input.
square = uc.Reader(lambda x: x * x)

# Define a Reader that leaves its input alone.
identity = uc.Reader(lambda x: x)

# Combine Readers arithmetically.
combined = square + identity

# Execute the Reader.
# Its constituent Readers act independently on the same input; the results are then added.
combined(3)  # output: 12 (3 * 3 + 3)
combined(5)  # output: 30 (5 * 5 + 5)

# Combine Readers to make a new Reader that produces a tuple.
combined = uc.make_tuple(square, identity)

combined(3)  # output: (9, 3)

# Create a Reader that produces a constant value for any input.
const10 = uc.const(10)

# Create a Reader that multiplies the output of its constituent Readers.
combined = uc.prod([const10, identity, square])

combined(3)  # Output: 270 (10 * 3 * 9)
```

## API Overview

### Core Methods

- `Reader(func)`: Wrap a function into a `Reader`.
- `uc.const(val)`: Create a `Reader` that always returns a constant value.
- `uc.attrgetter(attr, [default])`: Access an attribute (or nested attributes) from a context.
- `uc.make_tuple(*readers)`: Combine multiple `Reader`s into one that returns a tuple.

### Operators

`Reader`s support most logical/arithmetic operations:

- Arithmetic: `+`, `-`, `*`, `**`, `/`, `//`, `%`, `@`
- Boolean/bitwise logic: `and`, `or`, `&`, `|`, `^`, `~`
- Comparisons: `<`, `<=`, `>=`, `>`
    - For equality you cannot use `==` and `!=`; instead use `reader.equals(other)` and `reader.not_equals(other)`.

You can also apply operations to sequences of `Reader`s using `uc.sum`, `uc.prod`, `uc.all`, `uc.any`, `uc.min`, `uc.max`, and `uc.reduce`.

## License

This library is open-source and licensed under the [MIT License](LICENSE).

Contributions are welcome!
