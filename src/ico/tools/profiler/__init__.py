"""Profiler: where the time of a run goes, per node of the flow.

See ``docs/design/profiler.md``.
"""

from ico.tools.profiler.instrument import instrument
from ico.tools.profiler.metrics import CpuTime, IcoMetric, WallTime
from ico.tools.profiler.remote import IcoProfileEvent
from ico.tools.profiler.stats import StatsView
from ico.tools.profiler.table import MetricView, NodeProfile
from ico.tools.profiler.tool import (
    MetricInfo,
    ProcessInfo,
    ProfilerTool,
    ProfileSnapshot,
)

__all__ = [
    "CpuTime",
    "IcoMetric",
    "IcoProfileEvent",
    "MetricInfo",
    "MetricView",
    "NodeProfile",
    "ProcessInfo",
    "ProfileSnapshot",
    "ProfilerTool",
    "StatsView",
    "WallTime",
    "instrument",
]
