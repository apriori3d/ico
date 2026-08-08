import importlib
import pkgutil

from ico.core.node import IcoNodeProtocol
from ico.core.runtime.node import IcoRuntimeNode


def import_all_renderers(package_name: str) -> None:
    """Import all renderer modules from package for registration."""
    try:
        package = importlib.import_module(package_name)
        for _, modname, _ in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        ):
            importlib.import_module(modname)
    except ImportError as e:
        print(f"Error importing renderers from {package_name}: {e}")


def match_icon(
    node_icons: dict[type[IcoNodeProtocol] | type[IcoRuntimeNode], str],
    node: IcoNodeProtocol | IcoRuntimeNode,
) -> str | None:
    """Find matching icon for node type using isinstance hierarchy."""
    for node_type, icon in node_icons.items():
        if isinstance(node, node_type):
            return icon
    return None
