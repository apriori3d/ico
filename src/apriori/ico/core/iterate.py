from collections.abc import Iterator
from typing import Generic

from apriori.ico.core.operator import I, IcoOperator, O


class IcoIterateOperator(
    Generic[I, O],
    IcoOperator[Iterator[I], Iterator[O]],
):
    _operator: IcoOperator[I, O]

    def __init__(
        self,
        operator: IcoOperator[I, O],
    ) -> None:
        super().__init__(
            fn=self._iterate_fn,
            name=f"iterate({operator.name})]",
            children=[operator],
        )
        self._operator = operator

    def _iterate_fn(self, items: Iterator[I]) -> Iterator[O]:
        for item in items:
            yield self._operator(item)


def iterate(operator: IcoOperator[I, O]) -> IcoOperator[Iterator[I], Iterator[O]]:
    return IcoIterateOperator(operator)
