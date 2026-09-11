import abc
from dataclasses import dataclass, field
from typing import ClassVar, Literal

IcoProfileMetricType = Literal["cpu_time"]


@dataclass(slots=True)
class IcoProfilerMetric(abc.ABC):
    metric_type: ClassVar[IcoProfileMetricType]
    values: list[float] = field(default_factory=list[float])

    @abc.abstractmethod
    def before_call(self) -> None: ...

    @abc.abstractmethod
    def after_call(self) -> None: ...


@dataclass(slots=True)
class CpuTimeMetric(IcoProfilerMetric):
    metric_type: ClassVar[IcoProfileMetricType] = "cpu_time"
    _start_time: float = 0.0

    def before_call(self) -> None:
        import time

        self._start_time = time.time()

    def after_call(self) -> None:
        import time

        self.values.append(time.time() - self._start_time)


def create_metrics(metric_types: list[IcoProfileMetricType]) -> list[IcoProfilerMetric]:
    metrics: list[IcoProfilerMetric] = []
    for metric_type in metric_types:
        if metric_type == "cpu_time":
            metrics.append(CpuTimeMetric())
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
    return metrics
