from typing import Generic

from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.tools.monitor_node import IcoMonitorNode
from apriori.ico.tools.progress.mixin import ProgressMixin


class ProgressNode(
    Generic[I, O],
    IcoMonitorNode[I, O],
    ProgressMixin,
):
    total: int | None
    _task: int | None

    def __init__(
        self,
        wrapped_operator: IcoOperator[I, O],
        *,
        total: int | None = None,
        name: str | None = None,
    ):
        super().__init__(
            wrapped_operator=wrapped_operator,
            name=name or f"RichProgress({wrapped_operator.name})",
        )
        self.total = total

    def assign_task(self, task: int) -> None:
        self._task = task

    def _after_call(self, item: I, result: O) -> None:
        super()._after_call(item, result)

        if self.task is None:
            raise RuntimeError(
                "Progress task not set. Use Progress tool to assign task for this flow."
            )
