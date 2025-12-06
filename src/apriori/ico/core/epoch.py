from collections.abc import Callable, Iterator
from typing import Generic

from apriori.ico.core.context_operator import (
    C,
    I,
    IcoContextOperator,
    wrap_context_operator,
)
from apriori.ico.core.operator import IcoOperator, wrap_operator


class IcoEpoch(
    Generic[I, C],
    IcoOperator[C, C],
):
    type_name: str = "epoch"

    source: IcoOperator[None, Iterator[I]]
    context_operator: IcoContextOperator[I, C, C]

    def __init__(
        self,
        source: Callable[[None], Iterator[I]],
        context_operator: Callable[[I, C], C],
        *,
        name: str | None = None,
    ) -> None:
        source_op = wrap_operator(source)
        context_op = wrap_context_operator(context_operator)

        super().__init__(
            fn=self._process_fn,
            name=name,
            children=[source_op, context_op],
        )
        self.source = source_op
        self.context_operator = context_op

    def _process_fn(self, context: C) -> C:
        for item in self.source(None):
            context = self.context_operator(item, context)
        return context
