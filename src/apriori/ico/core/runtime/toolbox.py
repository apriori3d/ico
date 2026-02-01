from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode, create_runtime_walker
from apriori.ico.core.runtime.tool import IcoTool
from apriori.ico.core.tree_utils import TreePathIndex


@runtime_checkable
class IcoToolRegistrationProtocol(Protocol):
    def register_node(self, node: IcoRuntimeNode, path: TreePathIndex) -> None: ...


class IcoToolBox(IcoRuntimeNode):
    """Container for multiple runtime tools."""

    __slots__ = ("shell", "tools")

    shell: IcoRuntimeNode
    tools: list[IcoTool]

    def __init__(
        self,
        runtime: IcoRuntimeNode,
        tools: Sequence[IcoTool] | None = None,
    ) -> None:
        super().__init__()
        self.shell = runtime
        self.tools = []
        if tools is not None:
            self.add_tool(*tools)

    def add_tool(self, *tools: IcoTool) -> None:
        self.tools.extend(tools)
        self.add_runtime_children(*tools)

        for tool in tools:
            if isinstance(tool, IcoToolRegistrationProtocol):
                runtime_walker = create_runtime_walker()

                for node_info in runtime_walker.traverse(self.shell):
                    tool.register_node(*node_info.node_path)

    def remove_tool(self, *tools: IcoTool) -> None:
        for tool in tools:
            self.tools.remove(tool)
        self.remove_runtime_child(*tools)

    def on_forward_event(self, event: IcoRuntimeEvent) -> None:
        # Forward event to tools
        for tool in self.tools:
            tool.on_forward_event(event)
