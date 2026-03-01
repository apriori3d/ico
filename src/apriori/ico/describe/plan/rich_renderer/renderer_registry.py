from collections.abc import Callable

from apriori.ico.core.node import IcoNode
from apriori.ico.describe.plan.rich_renderer.custom_renderer import CustomRenderer
from apriori.ico.describe.plan.rich_renderer.group_renderer import GroupRenderer
from apriori.ico.describe.plan.rich_renderer.row_renderer import RowRenderer

RendererMetaTypes = type[RowRenderer | GroupRenderer | CustomRenderer]
RendererRegistry: dict[type[IcoNode], RendererMetaTypes] = {}


def register_renderer(
    node_type: type[IcoNode],
) -> Callable[[RendererMetaTypes], RendererMetaTypes]:
    """Decorator to register renderer class for specific node type."""

    def decorator(renderer_cls: RendererMetaTypes) -> RendererMetaTypes:
        RendererRegistry[node_type] = renderer_cls
        return renderer_cls

    return decorator
