from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Generic, final

from apriori.ico.core.operator import (
    I,
    IcoOperator,
    wrap_operator,
)
from apriori.ico.core.signature import IcoSignature


@final
class IcoPipeline(
    Generic[I],
    IcoOperator[I, I],
):
    __slots__ = "body"

    body: Sequence[IcoOperator[I, I]]

    def __init__(
        self,
        *body: Callable[[I], I],
        name: str | None = None,
    ):
        if len(body) == 0:
            raise ValueError("Pipeline body must contain at least one operator.")

        body_ops = [wrap_operator(step) for step in body]

        super().__init__(
            fn=self._run_streamline,
            name=name or "streamline",
            children=body_ops,
        )
        self.body = body_ops

    def _run_streamline(self, item: I) -> I:
        item = item
        for step in self.body:
            item = step(item)
        return item

    def __len__(self) -> int:
        return len(self.body)

    @property
    def signature(self) -> IcoSignature:
        """Infer ICO signature of this operator."""
        signature = super().signature

        return signature if signature.infered else self.body[0].signature
