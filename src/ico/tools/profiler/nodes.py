from collections.abc import Iterator
from typing import Any

from ico.core.context_operator import IcoContextOperator, IcoContextOperatorProtocol
from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.core.stream import IcoStream
from ico.tools.profiler.profiler import IcoProfilerNode


class IcoOperatorProfiler(IcoOperator[Any, Any], IcoProfilerNode):
    operator: IcoOperatorProtocol[Any, Any]

    def __init__(
        self, operator: IcoOperatorProtocol[Any, Any], *, name: str | None = None
    ) -> None:
        IcoOperator.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            self._profiled_fn,
            children=[operator],
            name=name or "Profiler",
        )
        IcoProfilerNode.__init__(self)
        self.operator = operator

    def _profiled_fn(self, item: Any) -> Any:
        result = self.operator(item)
        return result


class IcoContextOperatorProfiler(IcoContextOperator[Any, Any, Any], IcoProfilerNode):
    def __init__(
        self,
        operator: IcoContextOperatorProtocol[Any, Any, Any],
        *,
        name: str | None = None,
    ) -> None:
        IcoContextOperator.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            self._profiled_fn,
            children=[operator],
            name=name or "Profiler",
        )
        IcoProfilerNode.__init__(self)
        self.operator = operator

    def _profiled_fn(self, item: Any, context: Any) -> Any:
        result = self.operator(item, context)
        return result


class IcoStreamProfiler(IcoStream[Any, Any], IcoProfilerNode):
    def __init__(self, stream: IcoStream[Any, Any], *, name: str | None = None) -> None:
        IcoStream.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, stream.body, name=name or "Profiler"
        )
        IcoProfilerNode.__init__(self)

    def _profiled_fn(self, items: Iterator[Any]) -> Iterator[Any]:
        for item in items:
            # Here you could add profiling logic, e.g., timing each item
            result = self.body(item)
            yield result
