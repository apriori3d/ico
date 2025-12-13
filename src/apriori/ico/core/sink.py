from collections.abc import Callable, Iterator
from typing import ClassVar, Generic

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

    type_name: ClassVar[str] = "Sink"

    def __init__(
        self,
        consume_fn: Callable[[I], None],
        name: str | None = None,
    ) -> None:
        super().__init__(
            fn=self._sink_fn,
            name=name,
            original_fn=consume_fn,
        )
        self.consume_fn = consume_fn

    def _sink_fn(self, items: Iterator[I]) -> None:
        for item in items:
            self.consume_fn(item)


# ─────────────────────────────────────────────
# Decorator
# ─────────────────────────────────────────────


def sink() -> Callable[[Callable[[I], None]], IcoSink[I]]:
    def decorator(fn: Callable[[I], None]) -> IcoSink[I]:
        return IcoSink(fn)

    return decorator
