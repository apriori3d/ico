from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.tree_walker import create_runtime_tree_walker
from apriori.ico.core.tree_utils import TreePathIndex


@runtime_checkable
class IcoToolRegistrationProtocol(Protocol):
    def register_node(self, node: IcoRuntimeNode, path: TreePathIndex) -> None: ...


class IcoToolBox(IcoRuntimeNode):
    """Container for multiple runtime tools."""

    __slots__ = ("shell", "tools")

    shell: IcoRuntimeNode
    tools: list[IcoRuntimeNode]

    def __init__(
        self,
        shell: IcoRuntimeNode,
        tools: Sequence[IcoRuntimeNode] | None = None,
    ) -> None:
        super().__init__()
        self.shell = shell
        self.tools = list(tools) if tools is not None else []
        self.add_runtime_children(*self.tools)

    def register_tools(self) -> None:
        pending_tools = [
            tool for tool in self.tools if isinstance(tool, IcoToolRegistrationProtocol)
        ]
        if len(pending_tools) == 0:
            return

        runtime_walker = create_runtime_tree_walker(include_agent_worker=True)

        for node_info in runtime_walker.traverse(self.shell):
            for tool in pending_tools:
                tool.register_node(*node_info.node_path)

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        super().on_event(event)

        # Forward event to tools
        for tool in self.tools:
            if not tool.on_event(event):
                return None

        return event
