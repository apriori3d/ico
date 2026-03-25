from dataclasses import dataclass
from typing import Literal

from ico.core.node import IcoNodeProtocol
from ico.core.runtime.node import IcoRuntimeNode

RenderBackend = Literal["RichText",]


@dataclass(slots=True)
class RendererOptions:
    """
    Base configuration for ICO describe renderers.

    Common settings for both plan and runtime renderers including
    icon mappings, renderer paths, and display options.
    """

    node_icons: dict[type[IcoNodeProtocol | IcoRuntimeNode], str]
    """Icon mapping for different node types"""

    renderers_paths: list[str]
    """Module paths for automatic renderer loading"""

    backend: RenderBackend = "RichText"
    """Rendering backend (currently only RichText supported)"""

    dim_ico_nodes: bool = False
    """Dim ICO framework nodes to focus on user code"""

    show_node_icons: bool = True
    """Display emoji icons next to node names"""
