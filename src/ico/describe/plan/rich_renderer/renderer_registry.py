from collections import OrderedDict
from collections.abc import Callable

from ico.core.node import IcoNode
from ico.describe.plan.rich_renderer.custom_renderer import CustomRenderer
from ico.describe.plan.rich_renderer.group_renderer import GroupRenderer
from ico.describe.plan.rich_renderer.row_renderer import RowRenderer

RendererMetaTypes = type[RowRenderer | GroupRenderer | CustomRenderer]
RendererRegistry: dict[type[IcoNode], RendererMetaTypes] = OrderedDict()


def register_renderer(
    node_type: type[IcoNode],
) -> Callable[[RendererMetaTypes], RendererMetaTypes]:
    """Decorator to register renderer class for specific node type."""

    def decorator(renderer_cls: RendererMetaTypes) -> RendererMetaTypes:
        RendererRegistry[node_type] = renderer_cls
        return renderer_cls

    return decorator


def select_renderer(node_type: type[IcoNode]) -> RendererMetaTypes | None:
    """Select renderer class for given node type, with fallback to default."""
    for registered_type, renderer_cls in RendererRegistry.items():
        if issubclass(node_type, registered_type):
            return renderer_cls
    return None
