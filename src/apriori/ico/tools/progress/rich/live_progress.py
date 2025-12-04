from types import TracebackType
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    Progress,
    Task,
    TaskID,
)
from typing_extensions import Self


class ProgressWithStatus:
    def __init__(self, progress: Progress, console: Console | None = None) -> None:
        # All protocol methods are forwarded to the internal progress instance
        self.progress = progress

        # Status needs to be set before being used in _make_layout
        self._status_task_id: int | None = None
        self._prev_status: str = ""

        # Create live display for status updates
        self.live = Live(
            self._make_layout(),
            refresh_per_second=10,
            # Need to pass console to Live, otherwise it creates its own instance
            # which breaks printing from progress
            console=console,
        )

    def _make_layout(self) -> Group:
        # Collect current status of the tracked task
        if self._status_task_id is not None:
            status = next(
                task for task in self.tasks if task.id == self._status_task_id
            ).fields.get("status", "")
            self._prev_status = status
        else:
            status = self._prev_status

        status_text = self.live.console.render_str(status)

        # Add status text on the second line below progress bars
        return Group(
            self.progress,
            status_text,
        )

    @property
    def tasks(self) -> list[Task]:
        return self.progress.tasks

    def add_task(
        self,
        description: str,
        total: int,
        show_status: bool = False,
        **fields: Any,
    ) -> TaskID:
        task = self.progress.add_task(description, total=total, **fields)
        # Track this task as status task if requested
        if show_status:
            self._status_task_id = task
        return task

    def advance(self, task_id: TaskID, advance: int = 1) -> None:
        self.progress.advance(task_id, advance=advance)

    def update(
        self,
        task_id: TaskID,
        *,
        total: float | None = None,
        completed: float | None = None,
        advance: float | None = None,
        description: str | None = None,
        visible: bool | None = None,
        refresh: bool = False,
        status: str | None = None,
        **fields: Any,
    ) -> None:
        if status is not None:
            self._status_task_id = task_id
            fields["status"] = status

        self.progress.update(
            task_id,
            total=total,
            completed=completed,
            advance=advance,
            description=description,
            visible=visible,
            refresh=refresh,
            **fields,
        )
        # Update live display to reflect changes
        self.live.update(self._make_layout())

    def remove_task(self, task_id: TaskID) -> None:
        self.progress.remove_task(task_id)

        # Clear status task if it was removed
        if self._status_task_id == task_id:
            self._status_task_id = None

        # Update live display to reflect changes
        self.live.update(self._make_layout())

    def print(self, *objects: Any, **kw_args: Any) -> None:
        self.progress.print(*objects, **kw_args)

    def log(self, *objects: Any, **kw_args: Any) -> None:
        self.progress.log(*objects, **kw_args)

    def __enter__(self) -> Self:
        # Enter live display context
        self.live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Exit live display context
        self.live.__exit__(exc_type, exc_val, exc_tb)
