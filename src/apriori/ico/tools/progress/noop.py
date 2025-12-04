from typing import Any, final


@final
class NoOpProgress:
    @property
    def tasks(self) -> list[Any]:
        return []

    def add_task(
        self,
        description: str,
        start: bool = True,
        total: float | None = 100.0,
        completed: int = 0,
        visible: bool = True,
        **fields: Any,
    ) -> int:
        return 0

    def advance(self, task_id: int, advance: float = 1) -> None:
        pass

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
    ) -> None:
        pass

    def remove_task(self, task_id: int) -> None:
        pass

    def print(self, *objects: Any, **kw_args: Any) -> None:
        pass

    def log(self, *objects: Any, **kw_args: Any) -> None:
        pass
