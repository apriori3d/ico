from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from apriori.ico.core.chain import IcoChain
from apriori.ico.core.node import IcoNode
from apriori.ico.core.pipeline import IcoPipeline

RenderColumn: TypeAlias = Literal["Flow", "Name", "Type", "Signature", "State"]
CallableFormat: TypeAlias = Literal["__name__", "str()"]


@dataclass(slots=True)
class RenderOptions:
    include_runtime: bool = True
    callable_format: CallableFormat = "__name__"
    dim_ico_nodes: bool = True
    query_iterable_size: bool = True
    show_ico_operator: bool = False
    expand_subflows: bool = True
    expand_subflow_factories: bool = True

    columns: list[RenderColumn] = field(
        default_factory=lambda: ["Flow", "Signature", "State", "Name"]
    )

    flatten_node_type: set[type[IcoNode]] = field(
        default_factory=lambda: {IcoChain, IcoPipeline}
    )
