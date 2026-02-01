from __future__ import annotations

from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode


class IcoTool(IcoRuntimeNode):
    """Base class for ICO runtime tools."""

    __slots__ = ()

    def on_forward_event(self, event: IcoRuntimeEvent) -> None:
        """Handle forwarded event from runtime."""
        pass
