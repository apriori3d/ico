"""Probe hook of operator calls, for profilers.

An operator calls its callable directly unless a probe is set on the
instance (``node._probe``); then it calls the probe with its callable and
its arguments. Nothing else changes: ``fn``, the signature, ``describe`` and
the plan of an operator with a probe are the same. With no probe, the cost
is one attribute check per call.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, Union, cast

if TYPE_CHECKING:
    from ico.core.async_operator import IcoAsyncOperator
    from ico.core.context_operator import IcoContextOperator
    from ico.core.operator import IcoOperator


class IcoProbe(Protocol):
    """Called instead of an operator's callable when set on the operator."""

    def call(self, fn: Callable[[Any], Any], item: Any) -> Any:
        """Call ``fn(item)`` of an ``IcoOperator`` and return its result."""
        ...

    def call_with_context(
        self, fn: Callable[[Any, Any], Any], item: Any, context: Any
    ) -> Any:
        """Call ``fn(item, context)`` of an ``IcoContextOperator``."""
        ...

    async def call_async(self, fn: Callable[[Any], Awaitable[Any]], item: Any) -> Any:
        """Await ``fn(item)`` of an ``IcoAsyncOperator``."""
        ...


ProbedOperator = Union[
    "IcoOperator[Any, Any]",
    "IcoContextOperator[Any, Any, Any]",
    "IcoAsyncOperator[Any, Any]",
]
"""The operators that call a probe."""


def get_probe(node: object) -> IcoProbe | None:
    """The probe set on a node, if any."""
    return cast(IcoProbe | None, getattr(node, "_probe", None))


def set_probe(node: ProbedOperator, probe: IcoProbe | None) -> None:
    """Set (or with ``None`` remove) the probe of an operator."""
    node._probe = probe  # pyright: ignore[reportPrivateUsage]
