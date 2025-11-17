import multiprocessing
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import TYPE_CHECKING, Any

from apriori.ico.core import (
    IcoLifecycleEvent,
)
from apriori.ico.core.runtime.channels.messages import (
    ShutdownPayload,
    WorkerMessage,
)
from apriori.ico.core.runtime.types import IcoRuntimeMixin
from apriori.ico.core.types import IcoNodeType, IcoOperatorProtocol

if TYPE_CHECKING:
    WorkerQueue = Queue[WorkerMessage[Any]]
else:
    WorkerQueue = Queue  # noqa: F401


# ──── Simple Echo Operator ────


@dataclass
class EchoOperator(IcoOperatorProtocol[int, int]):
    name: str = "Echo"
    node_type: IcoNodeType = IcoNodeType.operator
    children: list[IcoOperatorProtocol[Any, Any]] = field(default_factory=list)
    fn: Callable[[int], int] = lambda x: x

    def __call__(self, item: int) -> int:
        return item

    async def run_async(self, item: int) -> int:
        return item

    @staticmethod
    def create() -> IcoOperatorProtocol[int, int]:
        return EchoOperator()


# ──── Failing with Exception Operator ────


@dataclass
class FailingOperator(IcoOperatorProtocol[int, int]):
    name: str = "Fail"
    node_type: IcoNodeType = IcoNodeType.operator
    children: list[IcoOperatorProtocol[Any, Any]] = field(default_factory=list)
    fn: Callable[[int], int] = lambda x: x

    def __call__(self, item: int) -> int:
        raise ValueError("Intentional failure")

    async def run_async(self, item: int) -> int:
        raise ValueError("Intentional async failure")

    @staticmethod
    def create() -> IcoOperatorProtocol[int, int]:
        return FailingOperator()


# ──── Recording Event Operator ────


class LifecycleEventsRecordingOperator(
    IcoOperatorProtocol[Any, Any],
    IcoRuntimeMixin,  # Added lifecycle support to allow event recording
):
    """
    An operator that records all lifecycle events it receives and bypasses data flow.
    ICO form:
        Any → Any
    """

    name: str = "EventRecorder"
    node_type: IcoNodeType = IcoNodeType.operator
    children: list[IcoOperatorProtocol[Any, Any]]
    fn: Callable[[Any], Any]

    received_events: list[IcoLifecycleEvent]

    def __init__(self) -> None:
        IcoRuntimeMixin.__init__(self)
        super().__init__()
        # Bypass data flow
        self.fn = lambda x: x
        self.children = []
        self.received_events = []

    def __call__(self, item: Any) -> Any:
        return self.fn(item)

    async def run_async(self, item: Any) -> Any:
        return await self(item)

    def on_event(self, event: IcoLifecycleEvent) -> None:
        self.received_events.append(event)


# ──── Worker Shutdown Helper ────


def shutdown_worker(
    in_q: WorkerQueue, out_q: WorkerQueue, proc: multiprocessing.Process
) -> None:
    in_q.put(WorkerMessage.create(ShutdownPayload()))
    proc.join(timeout=2)

    assert not proc.is_alive()
    assert proc.exitcode == 0, f"ProcessWorker exited with code {proc.exitcode}"
