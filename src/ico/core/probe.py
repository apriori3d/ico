"""Probe hooks, for tools such as profilers.

Two hooks, both ``None`` normally:

- ``probe`` of an operator (``IcoOperator``, ``IcoAsyncOperator``): when set, the operator calls the probe with its
  callable and its arguments instead of calling the callable.
- ``runner_probe`` of a worker (``IcoWorker``): when set, the worker's
  runner calls ``runner_probe.wrap(flow)`` on the flow it built from
  ``flow_factory`` and runs the flow that ``wrap`` returns.

Nothing else changes: ``fn``, ``flow_factory``, signatures, ``describe`` and
plans are the same with and without probes. Without a probe, the cost is
one attribute check per call.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ico.core.operator import IcoOperatorProtocol


class IcoProbe(Protocol):
    """Called instead of an operator's callable when set as its ``probe``."""

    def call(self, fn: Callable[[Any], Any], item: Any) -> Any:
        """Call ``fn(item)`` of an ``IcoOperator`` and return its result."""
        ...

    async def call_async(self, fn: Callable[[Any], Awaitable[Any]], item: Any) -> Any:
        """Await ``fn(item)`` of an ``IcoAsyncOperator``."""
        ...


class IcoRunnerProbe(Protocol):
    """Wraps the flow a worker's runner built, when set as the worker's
    ``runner_probe``. It travels to the runner process, so it must be
    pickleable."""

    def wrap(
        self, flow: IcoOperatorProtocol[Any, Any]
    ) -> IcoOperatorProtocol[Any, Any]:
        """The flow the runner runs instead of ``flow``."""
        ...
