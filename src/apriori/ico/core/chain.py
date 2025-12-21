from typing import Generic

from apriori.ico.core.operator import O2, I, IcoOperator, O


class IcoChain(
    Generic[I, O, O2],
    IcoOperator[I, O2],
):
    """Chained ICO Operator: (I → O, O → O2) == I → O2."""

    _first: IcoOperator[I, O]
    _second: IcoOperator[O, O2]

    def __init__(
        self,
        first: IcoOperator[I, O],
        second: IcoOperator[O, O2],
    ):
        super().__init__(
            fn=self._chained_fn,
            name=f"{first} | {second}",
            parent=None,
            children=[first, second],
        )
        self._first = first
        self._second = second

    def _chained_fn(self, item: I) -> O2:
        return self._second(self._first(item))


def chain(a: IcoOperator[I, O], b: IcoOperator[O, O2]) -> IcoOperator[I, O2]:
    return IcoChain(a, b)
