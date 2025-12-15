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
    consumer: Callable[[I], None]

    def __init__(
        self,
        consumer: Callable[[I], None],
        name: str | None = None,
    ) -> None:
        super().__init__(fn=self._sink_fn, name=name)
        self.consumer = consumer

    def _sink_fn(self, items: Iterator[I]) -> None:
        for item in items:
            self.consumer(item)


# ─────────────────────────────────────────────
# Decorator
# ─────────────────────────────────────────────


def sink(
    *,
    name: str | None = None,
) -> Callable[[Callable[[I], None]], IcoSink[I]]:
    def decorator(fn: Callable[[I], None]) -> IcoSink[I]:
        return IcoSink(fn, name=name)

    return decorator
