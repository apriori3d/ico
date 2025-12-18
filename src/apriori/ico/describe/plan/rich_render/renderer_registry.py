import importlib
import pkgutil
from collections.abc import Callable

from apriori.ico.core.node import IcoNode
from apriori.ico.describe.plan.rich_render.custom_renderer import CustomRenderer
from apriori.ico.describe.plan.rich_render.group_renderer import GroupRenderer
from apriori.ico.describe.plan.rich_render.row_renderer import RowRenderer

RendererTypes = type[RowRenderer | GroupRenderer | CustomRenderer]
RendererRegistry: dict[type[IcoNode], RendererTypes] = {}


def register_renderer(
    node_type: type[IcoNode],
) -> Callable[[RendererTypes], RendererTypes]:
    """Decorator to register a renderer class for a specific node type."""

    def decorator(renderer_cls: RendererTypes) -> RendererTypes:
        RendererRegistry[node_type] = renderer_cls
        return renderer_cls

    return decorator


def import_all_renderers(package_name: str):
    package = importlib.import_module(package_name)
    for _, modname, _ in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        importlib.import_module(modname)
