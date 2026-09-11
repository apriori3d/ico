from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, final

import numpy as np

from ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
    IcoRuntimeCommand,
)
from ico.core.runtime.node import IcoRuntimeNode
from ico.core.runtime.state import (
    BaseStateModel,
    IcoRuntimeState,
    IdleState,
    ReadyState,
    StateTransitionMap,
)
from ico.tools.profiler.metrics import (
    IcoProfileMetricType,
    IcoProfilerMetric,
    create_metrics,
)

# ────────────────────────────────────────────────
# Metrics controller
# ────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class MetricStats:
    metric_type: IcoProfileMetricType
    mean: float
    std: float


class IcoProfilerMetricsController:
    metrics: list[IcoProfilerMetric]

    def __init__(self, metrics: list[IcoProfilerMetric]) -> None:
        self.metrics = metrics

    def before_call(self) -> None:
        for metric in self.metrics:
            metric.before_call()

    def after_call(self) -> None:
        for metric in self.metrics:
            metric.after_call()

    def reset(self) -> None:
        for metric in self.metrics:
            metric.values = []

    def collect(self) -> list[MetricStats]:
        return [
            MetricStats(
                metric_type=metric.metric_type,
                mean=np.array(metric.values).mean(),
                std=np.array(metric.values).std(),
            )
            for metric in self.metrics
        ]


# ────────────────────────────────────────────────
# Runtime Commands
# ────────────────────────────────────────────────


@final
@dataclass(slots=True, frozen=True)
class IcoSetupProfilerNodeCommand(IcoRuntimeCommand):
    metrics_type: list[IcoProfileMetricType]


# ────────────────────────────────────────────────
# Runtime states
# ────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class PendingState(IcoRuntimeState):
    """Agent worker state before activation completion.

    PendingState represents agent workers that are initializing but not yet
    ready for execution. Used during agent startup and resource allocation.
    """

    name: ClassVar[str] = "Pending"


class ProfileNodeStateModel(BaseStateModel):
    transitions: ClassVar[StateTransitionMap] = {
        IcoActivateCommand: PendingState,
        IcoSetupProfilerNodeCommand: ReadyState,
        IcoDeactivateCommand: IdleState,
    }


# ────────────────────────────────────────────────
# Runtime Node
# ────────────────────────────────────────────────


class IcoProfilerNode(IcoRuntimeNode):
    metrics_controller: IcoProfilerMetricsController | None = None

    def __init__(
        self,
        runtime_name: str | None = None,
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
        state_model: BaseStateModel | None = None,
    ) -> None:
        IcoRuntimeNode.__init__(
            self,
            runtime_name,
            runtime_parent,
            runtime_children,
            state_model or ProfileNodeStateModel(),
        )

    def on_command(self, command: IcoRuntimeCommand) -> None:
        if isinstance(command, IcoSetupProfilerNodeCommand):
            self.metrics_controller = IcoProfilerMetricsController(
                create_metrics(command.metrics_type)
            )
        elif isinstance(command, IcoDeactivateCommand):
            self.metrics_controller = None

        return super().on_command(command)
