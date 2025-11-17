from collections.abc import Callable, Iterator
from typing import Generic

from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.types import I, NodeType


class IcoSink(
    Generic[I],
    IcoOperator[Iterator[I], None],
):
    """A terminal operator, consumes output and ends the data flow.
    ICO form:
        Iterator[I] → ()
    """

    def __init__(
        self, fn: Callable[[Iterator[I]], None], name: str | None = None
    ) -> None:
        super().__init__(fn=fn, name=name or "IcoSink", node_type=NodeType.sink)
