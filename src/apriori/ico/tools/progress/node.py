from dataclasses import dataclass
from typing import Generic, final

from apriori.ico.core.operator import I, IcoOperator
from apriori.ico.core.runtime.discovery import (
    IcoDiscovarableNode,
    IcoRegistrationEvent,
)
from apriori.ico.core.runtime.event import IcoRuntimeEvent


@final
@dataclass(slots=True, frozen=True)
class IcoProgressRegistrationEvent(IcoRegistrationEvent):
    total: float


@final
@dataclass(slots=True, frozen=True)
class IcoProgressEvent(IcoRuntimeEvent):
    node_id: int
    advance: float


class IcoProgress(
    Generic[I],
    IcoOperator[I, I],
    IcoDiscovarableNode,
):
    __slots__ = ("total",)

    total: float

    def __init__(
        self,
        total: float,
        *,
        name: str | None = None,
    ) -> None:
        IcoDiscovarableNode.__init__(self, runtime_name=name)
        IcoOperator.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            fn=self._progress_fn,
            name=name or "Progress",
        )
        self.total = total

    def _progress_fn(self, item: I) -> I:
        self.state_model.running()
        assert self.registered_id is not None

        self.bubble_event(IcoProgressEvent(node_id=self.registered_id, advance=1))

        self.state_model.ready()
        return item

    def _register_node(self):
        """Implement discovery contract"""
        assert self.registered_id is not None

        self.bubble_event(
            IcoProgressRegistrationEvent(
                node_type=type(self),
                node_name=self.runtime_name,
                node_id=self.registered_id,
                total=self.total,
            )
        )
