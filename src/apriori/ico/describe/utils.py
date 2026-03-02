import contextlib
import importlib
import pkgutil

from apriori.ico.core.node import IcoNode
from apriori.ico.core.runtime.node import IcoRuntimeNode


def import_all_renderers(package_name: str) -> None:
    """Import all renderer modules from package for registration."""
    with contextlib.suppress(ImportError):
        package = importlib.import_module(package_name)
        for _, modname, _ in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        ):
            importlib.import_module(modname)


def match_icon(
    node_icons: dict[type[IcoNode | IcoRuntimeNode], str],
    node: IcoNode | IcoRuntimeNode,
) -> str | None:
    """Find matching icon for node type using isinstance hierarchy."""
    for node_type, icon in node_icons.items():
        if isinstance(node, node_type):
            return icon
    return None
