from apriori.ico.core.operator import operator
from apriori.ico.core.sink import sink
from apriori.ico.core.source import source


# Define simple number processing pipeline
@source()
def numbers() -> list[int]:
    """Source: Generate numbers 1 to 5"""
    return [1, 2, 3, 4, 5]


@operator()
def square(x: int) -> int:
    """Transform: Square the number"""
    return x * x


@operator()
def to_string(x: int) -> str:
    """Convert: Number to formatted string"""
    return f"Result: {x}"


@sink()
def print_output(text: str) -> None:
    """Output: Display the result"""
    print(text)


# Compose with elegant | syntax
pipeline = numbers | (square | to_string).stream() | print_output

# Show pipeline structure
pipeline.describe()
