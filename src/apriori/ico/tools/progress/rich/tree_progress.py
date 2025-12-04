from typing import Any

from rich.console import Console
from rich.progress import GetTimeCallable, Progress, ProgressColumn, TaskID


class TreeProgress(Progress):
    def __init__(
        self,
        *columns: str | ProgressColumn,
        console: Console | None = None,
        auto_refresh: bool = True,
        refresh_per_second: float = 10,
        speed_estimate_period: float = 30.0,
        transient: bool = False,
        redirect_stdout: bool = True,
        redirect_stderr: bool = True,
        get_time: GetTimeCallable | None = None,
        disable: bool = False,
        expand: bool = False,
    ):
        super().__init__(
            *columns,
            console=console,
            auto_refresh=auto_refresh,
            refresh_per_second=refresh_per_second,
            speed_estimate_period=speed_estimate_period,
            transient=transient,
            redirect_stdout=redirect_stdout,
            redirect_stderr=redirect_stderr,
            get_time=get_time,
            disable=disable,
            expand=expand,
        )

        # Mapping of task IDs to their parent task IDs for hierarchy
        self._task_parent: dict[TaskID, TaskID | None] = {}

        # Currently active task for context in print
        self._active_task: TaskID | None = None

        # Note: inaccurate design in rich.Progress - print() is not virtual
        self.print = self._print

    # Progress methods with hierarchical task support

    def add_task(
        self,
        description: str,
        start: bool = True,
        total: float | None = 100.0,
        completed: int = 0,
        visible: bool = True,
        parent_task: TaskID | None = None,
        is_last_subtask: bool = True,
        **fields: Any,
    ) -> TaskID:
        task_prefix = self._get_task_prefix(parent_task, is_last_subtask)
        task_id = super().add_task(
            description=f"{task_prefix}{description}",
            start=start,
            total=total,
            completed=completed,
            visible=visible,
            is_last_subtask=is_last_subtask,
            **fields,
        )
        self._task_parent[task_id] = parent_task
        return task_id

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
        **fields: Any,
    ) -> None:
        # Append hierarchical prefix to description
        if description is not None:
            task = self._tasks[task_id]
            # 'Is the last subtask' flag will affect node branch prefix
            is_last_subtask = task.fields.get("is_last_subtask", True)
            parent_task_id = self._task_parent.get(task_id)

            task_prefix = self._get_task_prefix(parent_task_id, is_last_subtask)
            description = f"{task_prefix}{description}"

        # Update active task to use in task context retrieval in print
        self._active_task = task_id

        super().update(
            task_id,
            total=total,
            completed=completed,
            advance=advance,
            description=description,
            visible=visible,
            refresh=refresh,
            **fields,
        )

    def remove_task(self, task_id: TaskID) -> None:
        super().remove_task(task_id)

        # Clean up parent mapping
        self._task_parent.pop(task_id, None)

        # Clear active task if it was removed
        if self._active_task == task_id:
            self._active_task = None

    def _print(self, *objects: Any, **kw_args: Any) -> None:
        """Print text lines with context prefix of the currently active task."""

        if self._active_task is None:
            self.console.print(*objects, **kw_args)
            return

        # Find string objects to add context prefix
        indent_objects: list[str] = []
        str_objects = [obj for obj in objects if isinstance(obj, str)]
        other_objects = [obj for obj in objects if not isinstance(obj, str)]

        # Add context prefix to each line of the string object
        if len(str_objects) > 0:
            print_context = self._get_print_context()
            for str_obj in str_objects:
                indent_objects.append(
                    "".join(
                        f"{print_context}{line}"
                        for line in str_obj.splitlines(keepends=True)
                    )
                )

        # Print all objects
        self.console.print(*indent_objects, *other_objects, **kw_args)

    # Hierarchical task helpers

    def _get_task_prefix(
        self,
        parent_task_id: TaskID | None,
        is_last_subtask: bool | None,
    ) -> str:
        """Get the hierarchical prefix for a task based on its tree position."""

        if parent_task_id is None:
            # Root task has no prefix
            return ""

        # Get the task path to the root
        task_path = self._get_task_path(parent_task_id)

        # Append branch prefix for a new task, depending on whether it's the last subtask
        task_branch_prefix = "└──" if is_last_subtask else "├──"

        return f"{task_path}{task_branch_prefix}"

    def _get_task_path(self, task_id: TaskID) -> str:
        """Get the hierarchical path prefix for a task up to the root."""

        prefix = ""
        # Each level represented by task_id and parent_task_id,
        parent_task_id = self._task_parent.get(task_id)

        # Traverse up to the root task and insert prefix for each level
        while parent_task_id is not None:
            task = self._tasks[task_id]
            # Determine the indentation prefix based on the node type
            is_last_node = task.fields.get("is_last_subtask", True)
            indent_prefix = "   " if is_last_node else "│  "
            # Insert prefix at the beginning
            prefix = f"{indent_prefix}{prefix}"
            # Move up to the next level
            task_id = parent_task_id
            parent_task_id = self._task_parent.get(task_id)

        return prefix

    def _get_print_context(self) -> str:
        """Get the context string for the currently active task."""

        context_parts: list[str] = []
        # Traverse up from the active task to the root, collecting context fields
        task_id = self._active_task

        while task_id is not None:
            task = self._tasks[task_id]
            context = task.fields.get("context", None)
            if context is not None:
                context_parts.append(str(context))
            task_id = self._task_parent.get(task_id)

        # Concatenate context parts in reverse order (root to leaf)
        print_context = " ".join(reversed(context_parts))
        return print_context + " " if print_context else ""
