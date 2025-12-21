from dataclasses import dataclass
from typing import Literal

from apriori.ico.core.node import IcoNode
from apriori.ico.core.runtime.node import IcoRuntimeNode

RenderBackend = Literal["RichText",]


@dataclass(slots=True)
class RendererOptions:
    node_icons: dict[type[IcoNode | IcoRuntimeNode], str]
    renderers_paths: list[str]
    backend: RenderBackend = "RichText"
    dim_ico_nodes: bool = False
    expand_subflows: bool = True
    expand_subflow_factories: bool = True
    show_node_icons: bool = True
