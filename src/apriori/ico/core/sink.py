from collections.abc import Callable, Iterator
from typing import Any, Generic

from apriori.ico.core.operator import (
    I,
    IcoOperator,
)
from apriori.ico.core.signature import IcoSignature


class IcoSink(
    Generic[I],
    IcoOperator[Iterator[I], None],
):
    """A terminal operator, consumes output and ends the data flow.
    ICO form:
        Iterator[I] → ()
    """

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

    @property
    def signature(self) -> IcoSignature:
        from apriori.ico.core.signature_utils import (
            infer_from_callable,
        )

        signature = super().signature

        if signature.infered:
            return IcoSignature(
                i=Iterator[signature.i],
                c=None,
                o=None,
            )

        # Infer from consumer callable
        consumer_signature = infer_from_callable(self.consumer)

        if consumer_signature is not None:
            return IcoSignature(
                i=Iterator[consumer_signature.i],
                c=None,
                o=None,
            )

        return IcoSignature(
            i=Iterator[Any],
            c=None,
            o=None,
            infered=False,
        )


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
