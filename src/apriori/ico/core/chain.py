from typing import Any, Generic

from apriori.ico.core.operator import O2, I, IcoOperator, O
from apriori.ico.core.signature import IcoSignature


class IcoChain(
    Generic[I, O, O2],
    IcoOperator[I, O2],
):
    """Chained ICO Operator: (I → O, O → O2) == I → O2."""

    _left: IcoOperator[I, O]
    _right: IcoOperator[O, O2]

    def __init__(
        self,
        left: IcoOperator[I, O],
        right: IcoOperator[O, O2],
    ):
        super().__init__(
            fn=self._chained_fn,
            name=f"{left} | {right}",
            parent=None,
            children=[left, right],
        )
        self._left = left
        self._right = right

    def _chained_fn(self, item: I) -> O2:
        return self._right(self._left(item))

    @property
    def signature(self) -> IcoSignature:
        left = self._left.signature
        right = self._right.signature

        if left.infered and right.infered:
            return IcoSignature(
                i=left.i,
                c=None,
                o=right.o,
            )
        return IcoSignature(
            i=Any,
            c=None,
            o=Any,
            infered=False,
        )


def chain(
    left: IcoOperator[I, O],
    right: IcoOperator[O, O2],
) -> IcoOperator[I, O2]:
    return IcoChain(left, right)
