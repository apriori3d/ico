from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, final

from apriori.ico.core.dsl.operator import (
    IcoOperator,
    wrap_operator,
)
from apriori.ico.core.types import I, IcoOperatorProtocol, NodeType, O


@final
class IcoStream(
    Generic[I, O],
    IcoOperator[Iterator[I], Iterator[O]],
):
    """
    Applies a body operator to each element in a data stream.

    ICO form:
        Iterator[I] → Iterator[O]

    Example:
        scale = IcoOperator[float, float](lambda x: x * 2)
        stream = IcoStream(scale)
        result = list(stream((1, 2, 3)))  # [2, 4, 6]
    """

    __slots__ = ("body",)

    body: IcoOperatorProtocol[I, O]

    def __init__(
        self,
        body: Callable[[I], O],
        *,
        name: str | None = None,
    ):
        body_op = wrap_operator(body)

        super().__init__(
            fn=self._stream_fn,
            name=name,
            node_type=NodeType.stream,
            children=[body_op],
        )
        self.body = body_op

    def _stream_fn(self, items: Iterator[I]) -> Iterator[O]:
        for item in items:
            yield self.body(item)
