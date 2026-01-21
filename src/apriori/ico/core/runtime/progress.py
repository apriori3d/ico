from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, final

from apriori.ico.core.operator import I
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.monitor import IcoMonitor
from apriori.ico.core.tree_utils import TreePathIndex


@final
@dataclass(slots=True, frozen=True)
class IcoProgressEvent(IcoRuntimeEvent):
    advance: float

    @staticmethod
    def create(advance: float = 1) -> IcoProgressEvent:
        return IcoProgressEvent(trace=TreePathIndex(), advance=advance)


class IcoProgress(Generic[I], IcoMonitor[I]):
    __slots__ = ("total",)

    total: float

    def __init__(
        self,
        total: float,
        *,
        name: str | None = None,
    ) -> None:
        IcoMonitor.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, name=name
        )
        self.total = total

    def _before_call(self, item: I) -> None:
        super()._before_call(item)

        self.bubble_event(IcoProgressEvent.create(advance=1), from_child=self)

    def _after_call(self, item: I) -> None:
        self.state_model.ready()
