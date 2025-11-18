from __future__ import annotations

from collections.abc import Callable
from typing import Generic, final

from apriori.ico.core.dsl.operator import (
    IcoOperator,
)
from apriori.ico.core.types import C, IcoNodeProtocol, IcoNodeType


@final
class IcoProcess(
    Generic[C],
    IcoOperator[C, C],
):
    """
    Repeatedly applies an operator to the same context.

    Represents a fixed update rule applied several times to the same state.
    Useful for iterative refinement, optimization, or simulation steps.

    ICO form:
        C → C → C   (repeated `steps` times)

    Example: Fibonacci sequence as a process: an iterative ICO process can
    model recursion or stateful computations.

    >>> fib_process = IcoProcess(lambda c: (c[1], c[0] + c[1]), num_iterations=8)
    >>> fib_process((0, 1))
    (21, 34)

    Flow structure:
        process[C → C]
        └── operator[fib_step]
    """

    __slots__ = ("num_iterations", "body")

    body: Callable[[C], C]
    num_iterations: int

    def __init__(
        self,
        body: Callable[[C], C],
        num_iterations: int,
        name: str | None = None,
    ):
        body_fn = body

        super().__init__(
            fn=body,
            name=name,
            node_type=IcoNodeType.process,
            children=[body] if isinstance(body, IcoNodeProtocol) else [],
        )
        self.body = body_fn
        self.num_iterations = num_iterations

    def __call__(self, context: C) -> C:
        for _ in range(self.num_iterations):
            context = self.fn(context)
        return context
