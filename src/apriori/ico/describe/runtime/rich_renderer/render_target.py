from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeAlias

from rich.text import Text

from apriori.ico.core.runtime.node import (
    HasRemoteRuntime,
    IcoRemotePlaceholderNode,
    IcoRuntimeNode,
    RuntimeTreeWalker,
)
from apriori.ico.core.tree_utils import TraversalInfo, TreeWalker
from apriori.ico.describe.runtime.rich_renderer.row_renderer import RuntimeRowRenderer


class RuntimeRenderTarget(Protocol):
    """Protocol defining runtime renderer interface for tree traversal."""

    def render_node(self, node_info: RuntimeRendererTraversalInfo) -> None:
        """Render individual runtime node during tree traversal."""
        ...

    def render_row(
        self,
        row_renderer: RuntimeRowRenderer,
        node: IcoRuntimeNode,
        indent: Text | None = None,
    ) -> None:
        """Render runtime table row with optional indentation."""
        ...

    def push_group_indent(self, indent: Text) -> None:
        """Add group indentation level for runtime display."""
        ...

    def pop_group_indent(self) -> None: ...


# ──────── Runtime tree walker API ────────

RuntimeRendererWalker: TypeAlias = TreeWalker[IcoRuntimeNode, None]
RuntimeRendererTraversalInfo: TypeAlias = TraversalInfo[IcoRuntimeNode, None]


def create_runtime_renderer_walker(
    expand_remote_runtimes: bool = False,
) -> RuntimeRendererWalker:
    """Get a tree walker for ICO runtime nodes."""

    def _get_children(node: IcoRuntimeNode) -> Sequence[IcoRuntimeNode]:
        children = list(node.runtime_children)

        if expand_remote_runtimes and isinstance(node, HasRemoteRuntime):
            factory = node.get_remote_runtime_factory()
            remote_runtime = factory()
            children.append(remote_runtime)

        return [c for c in children if not isinstance(c, IcoRemotePlaceholderNode)]

    return RuntimeTreeWalker(get_children_fn=_get_children)
