from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import ClassVar, Generic, final

from apriori.ico.core.context_operator import (
    C,
    IcoContextOperator,
    wrap_context_operator,
)
from apriori.ico.core.operator import (
    I,
    IcoOperator,
    O,
    wrap_operator,
)


@final
class IcoContextPipeline(
    Generic[I, C, O],
    IcoContextOperator[I, C, O],
):
    type_name: ClassVar[str] = "Context Pipeline"

    __slots__ = ("apply", "body")

    apply: IcoContextOperator[I, C, O]
    body: Sequence[IcoOperator[O, O]]

    def __init__(
        self,
        apply: Callable[[I, C], O],
        *body: Callable[[O], O],
        name: str | None = None,
    ):
        if len(body) == 0:
            raise ValueError("Pipeline body must contain at least one operator.")

        apply_op = wrap_context_operator(apply)
        body_ops = [wrap_operator(step) for step in body]
        children = [apply_op] + body_ops

        super().__init__(
            fn=self._run_pipeline,
            name=name,
            children=children,
        )
        self.apply = apply_op
        self.body = body_ops

    def _run_pipeline(self, item: I, context: C) -> O:
        output = self.apply(item, context)
        for step in self.body:
            output = step(output)
        return output

    def __len__(self) -> int:
        return len(self.body)
