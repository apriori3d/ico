from collections import OrderedDict
from collections.abc import Callable

from ico.core.node import IcoNodeProtocol
from ico.describe.plan.rich_renderer.custom_renderer import CustomRenderer
from ico.describe.plan.rich_renderer.group_renderer import GroupRenderer
from ico.describe.plan.rich_renderer.row_renderer import RowRenderer

RendererMetaTypes = type[RowRenderer | GroupRenderer | CustomRenderer]
RendererRegistry: dict[type[IcoNodeProtocol], RendererMetaTypes] = OrderedDict()


def register_renderer(
    node_type: type[IcoNodeProtocol],
) -> Callable[[RendererMetaTypes], RendererMetaTypes]:
    """Decorator to register renderer class for specific node type."""

    def decorator(renderer_cls: RendererMetaTypes) -> RendererMetaTypes:
        RendererRegistry[node_type] = renderer_cls
        return renderer_cls

    return decorator


def select_renderer(node_type: type[IcoNodeProtocol]) -> RendererMetaTypes | None:
    """Select renderer class for given node type, with fallback to default."""
    matches = [
        (registered_type, renderer_cls)
        for registered_type, renderer_cls in RendererRegistry.items()
        if issubclass(node_type, registered_type)
    ]
    if len(matches) == 0:
        return None

    # Find the renderer for the most specific class
    while len(matches) > 1:
        (first_op_type, first_renderer_type), (second_op_type, second_renderer_type) = (
            matches[:2]
        )
        # Select more specialized class
        selected = (
            (first_op_type, first_renderer_type)
            if issubclass(second_renderer_type, first_renderer_type)
            else (second_op_type, second_renderer_type)
        )
        matches = matches[2:] + [selected]

    return matches[0][1]
