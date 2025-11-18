from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Generic, final

from apriori.ico.core.dsl.operator import (
    IcoOperator,
)
from apriori.ico.core.types import (
    C,
    I,
    IcoNodeProtocol,
    IcoNodeType,
    O,
)


@final
class IcoPipeline(
    Generic[I, C, O],
    IcoOperator[I, O],
):
    """
    A transformation pipeline following the ICO convention.

    ICO form:
        I → C → O
        context: I → C
        body:    [C → C]
        output:   C → O

    Each component can be any callable or nested IcoOperator.

    The `body` operates directly on the context `C`, possibly mutating it.
    This allows stateful or iterative transformations (e.g. model updates,
    accumulated metrics, cached buffers) within the same pipeline lifecycle.

    Example:
        |> from apriori.ico.core import IcoOperator, IcoPipeline

        |> to_float = IcoOperator(float)
        |> scale = IcoOperator(lambda x: x * 2)
        |> to_string = IcoOperator(str)

        |> pipeline = IcoPipeline(
        ...     context=to_float,
        ...     body=[scale],
        ...     output=to_string,
        ... )
        |> print(pipeline("21.0"))
        '42.0'

        # ICO structure:
        # pipeline
        # ├── operator[to_float]
        # ├── operator[scale]
        # └── operator[to_string]
    """

    __slots__ = ("context", "body", "output")

    context: Callable[[I], C]
    body: Sequence[Callable[[C], C]]
    output: Callable[[C], O]

    def __init__(
        self,
        context: Callable[[I], C],
        body: Sequence[Callable[[C], C]],
        output: Callable[[C], O],
        name: str | None = None,
    ):
        context_fn = context
        body_fn = body
        output_fn = output

        # Collect children for ICO structure
        children: list[IcoNodeProtocol] = []

        if isinstance(context, IcoNodeProtocol):
            children.append(context)

        for step in body:
            if isinstance(step, IcoNodeProtocol):
                children.append(step)

        if isinstance(output, IcoNodeProtocol):
            children.append(output)

        super().__init__(
            fn=self._run_pipeline,
            name=name,
            node_type=IcoNodeType.pipeline,
            children=children,
        )
        self.context = context_fn
        self.body = body_fn
        self.output = output_fn

    def _run_pipeline(self, item: I) -> O:
        ctx = self.context(item)

        for step in self.body:
            ctx = step(ctx)

        result = self.output(ctx)

        return result

    def __len__(self) -> int:
        return len(self.body)
