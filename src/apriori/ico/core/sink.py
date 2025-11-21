from collections.abc import Callable, Iterator
from typing import Generic

from apriori.ico.core.operator import (
    I,
    IcoOperator,
)


class IcoSink(
    Generic[I],
    IcoOperator[Iterator[I], None],
):
    """A terminal operator, consumes output and ends the data flow.
    ICO form:
        Iterator[I] → ()
    """

    def __init__(
        self,
        fn: Callable[[Iterator[I]], None],
        name: str | None = None,
    ) -> None:
        super().__init__(
            fn=fn,
            name=name or "sink",
        )
