from ico.core.operator import IcoOperator
from ico.core.process import IcoProcess


def test_fibonacci_process() -> None:
    """
    Test that IcoProcess correctly performs iterative context mutation.

    This example models Fibonacci sequence generation using a simple
    iterative process that updates the context tuple (a, b) → (b, a + b)
    over a fixed number of iterations.
    """

    # Define Fibonacci update step: (a, b) → (b, a + b)
    fib_step = IcoOperator[tuple[int, int], tuple[int, int]](
        lambda c: (c[1], c[0] + c[1])
    )

    # Create process that applies fib_step 8 times
    fib_process = IcoProcess(fib_step, num_iterations=8)

    # Run process starting from context (0, 1)
    result = fib_process((0, 1))

    # The 8th iteration gives (21, 34)
    assert result == (21, 34)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
