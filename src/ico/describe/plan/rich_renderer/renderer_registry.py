from collections import OrderedDict
from collections.abc import Callable

from ico.describe.plan.rich_renderer.custom_renderer import CustomRenderer
from ico.describe.plan.rich_renderer.group_renderer import GroupRenderer
from ico.describe.plan.rich_renderer.row_renderer import RowRenderer

RendererMetaTypes = type[RowRenderer | GroupRenderer | CustomRenderer]
RendererRegistry: dict[type[object], RendererMetaTypes] = OrderedDict()


def register_renderer(
    *node_types: type[object],
) -> Callable[[RendererMetaTypes], RendererMetaTypes]:
    """Decorator to register renderer class for specific node type."""

    def decorator(renderer_cls: RendererMetaTypes) -> RendererMetaTypes:
        for node_type in node_types:
            RendererRegistry[node_type] = renderer_cls
        return renderer_cls

    return decorator


def select_renderer(node_type: type[object]) -> RendererMetaTypes | None:
    """Select renderer class for given node type, with fallback to default."""

    sources = [
        registered_type
        for registered_type in RendererRegistry
        if registered_type in node_type.__mro__
    ]
    target_type = select_the_most_specific_type(node_type, sources)

    if not target_type:
        return None

    return RendererRegistry[target_type]


def select_the_most_specific_type(
    target: type[object], sources: list[type[object]]
) -> type[object] | None:
    """Resolve renderer for given node type, with fallback to default."""

    sources = [source_type for source_type in sources if source_type in target.__mro__]
    if len(sources) == 0:
        return None

    # Find the renderer for the most specific class
    while len(sources) > 1:
        first_source_type, second_source_type = sources[:2]

        # Select more specialized class
        selected_type = (
            first_source_type
            if issubclass(first_source_type, second_source_type)
            else second_source_type
        )
        sources = sources[2:] + [selected_type]

    return sources[0]
