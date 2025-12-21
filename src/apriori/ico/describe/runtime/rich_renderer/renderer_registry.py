from collections.abc import Callable

from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.describe.runtime.rich_renderer.custom_renderer import (
    RuntimeCustomRenderer,
)
from apriori.ico.describe.runtime.rich_renderer.row_renderer import RuntimeRowRenderer

RendererMetaTypes = type[RuntimeRowRenderer | RuntimeCustomRenderer]
RendererRegistry: dict[type[IcoRuntimeNode], RendererMetaTypes] = {}


def register_renderer(
    node_type: type[IcoRuntimeNode],
) -> Callable[[RendererMetaTypes], RendererMetaTypes]:
    """Decorator to register a renderer class for a specific node type."""

    def decorator(renderer_cls: RendererMetaTypes) -> RendererMetaTypes:
        RendererRegistry[node_type] = renderer_cls
        return renderer_cls

    return decorator
