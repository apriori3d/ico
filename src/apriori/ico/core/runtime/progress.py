from dataclasses import dataclass
from typing import Generic, final

from apriori.ico.core.operator import I
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.monitor import IcoMonitor
from apriori.ico.core.runtime.tool import IcoDiscoverableNode, IcoRegistrationEvent
from apriori.ico.core.tree_utils import TreePathIndex


@final
@dataclass(slots=True, frozen=True)
class IcoProgressRegistrationEvent(IcoRegistrationEvent):
    total: float


@final
@dataclass(slots=True, frozen=True)
class IcoProgressEvent(IcoRuntimeEvent):
    node_path: TreePathIndex
    advance: float


class IcoProgress(
    Generic[I],
    IcoMonitor[I],
    IcoDiscoverableNode,
):
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
        IcoDiscoverableNode.__init__(self, runtime_name=name)
        self.total = total

    def _before_call(self, item: I) -> None:
        assert self.registered_id is not None
        self.state_model.running()
        self.bubble_event(IcoProgressEvent(node_path=self.registered_id, advance=1))

    def _after_call(self, item: I) -> None:
        self.state_model.ready()

    def _register_node(self):
        """Implement discovery contract"""
        assert self.registered_id is not None

        self.bubble_event(
            IcoProgressRegistrationEvent(
                node_type=type(self),
                node_name=self.runtime_name,
                node_path=self.registered_id,
                total=self.total,
            )
        )
