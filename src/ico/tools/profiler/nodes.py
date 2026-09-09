from typing import Any

from ico.core.context_operator import IcoContextOperator, IcoContextOperatorProtocol
from ico.core.operator import IcoOperator, IcoOperatorProtocol


class IcoOperatorProfiler(IcoOperator[Any, Any]):
    operator: IcoOperatorProtocol[Any, Any]

    def __init__(
        self, operator: IcoOperatorProtocol[Any, Any], *, name: str | None = None
    ) -> None:
        super().__init__(
            self._profiled_fn, children=[operator], name=name or "Profiler"
        )
        self.operator = operator

    def _profiled_fn(self, item: Any) -> Any:
        result = self.operator(item)
        return result


class IcoContextOperatorProfiler(IcoContextOperator[Any, Any, Any]):
    def __init__(
        self,
        operator: IcoContextOperatorProtocol[Any, Any, Any],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(
            self._profiled_fn, children=[operator], name=name or "Profiler"
        )
        self.operator = operator

    def _profiled_fn(self, item: Any, context: Any) -> Any:
        result = self.operator(item, context)
        return result
