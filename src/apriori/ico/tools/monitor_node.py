from typing import Generic

from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.runtime.node import IcoRuntimeNode


class IcoMonitor(
    Generic[I, O],
    IcoOperator[I, O],
    IcoRuntimeNode,
):
    operator: IcoOperator[I, O]

    def __init__(
        self,
        operator: IcoOperator[I, O],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            fn=self._monitor_fn,
            name=name or f"Monitor({operator})",
        )
        self.operator = operator

    def _monitor_fn(self, item: I) -> O:
        self._before_call(item)
        result = self.operator(item)
        self._after_call(item, result)
        return result

    def _before_call(self, item: I) -> None:
        pass

    def _after_call(self, item: I, result: O) -> None:
        pass
