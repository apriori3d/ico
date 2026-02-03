from __future__ import annotations

from collections.abc import Callable
from typing import Generic, final

from apriori.ico.core.context_operator import C
from apriori.ico.core.operator import (
    IcoOperator,
    wrap_operator,
)
from apriori.ico.core.signature import IcoSignature


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

    num_iterations: int
    body: IcoOperator[C, C]

    def __init__(
        self,
        body: Callable[[C], C],
        *,
        num_iterations: int,
        name: str | None = None,
    ):
        # body_fn = body
        body = wrap_operator(body)

        super().__init__(
            fn=self._process_fn,
            name=name or "process",
            children=[body],
        )
        self.body = body
        self.num_iterations = num_iterations

    def _process_fn(self, context: C) -> C:
        for _ in range(self.num_iterations):
            context = self.body(context)
        return context

    @property
    def signature(self) -> IcoSignature:
        """Infer ICO signature of this operator."""
        signature = super().signature

        return signature if signature.infered else self.body.signature
