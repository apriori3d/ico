from enum import Enum, auto
from typing import Any, Protocol

from typing_extensions import runtime_checkable


class ProgressProtocol(Protocol):
    @property
    def tasks(self) -> list[Any]: ...

    def add_task(
        self,
        description: str,
        start: bool = True,
        total: float | None = 100.0,
        completed: int = 0,
        visible: bool = True,
        **fields: Any,
    ) -> int: ...

    def advance(self, task_id: int, advance: float = 1) -> None: ...

    def update(
        self,
        task_id: int,
        *,
        total: float | None = None,
        completed: float | None = None,
        advance: float | None = None,
        description: str | None = None,
        visible: bool | None = None,
        refresh: bool = False,
        **fields: Any,
    ) -> None: ...

    def remove_task(self, task_id: int) -> None: ...

    def print(self, *objects: Any, **kw_args: Any) -> None: ...

    def log(self, *objects: Any, **kw_args: Any) -> None: ...


@runtime_checkable
class SupportsProgress(Protocol):
    progress: ProgressProtocol


@runtime_checkable
class SupportsTaskProgress(Protocol):
    progress: ProgressProtocol
    task: int


class TaskStructure(Enum):
    hidden = auto()
    inline = auto()
    tree = auto()
    undefined = auto()
