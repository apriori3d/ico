from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from apriori.ico.core.chain import IcoChain
from apriori.ico.core.node import IcoNode
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.process import IcoProcess
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.source import IcoSource
from apriori.ico.core.stream import IcoStream
from apriori.ico.runtime.agent.mp_process.mp_process import MPProcess
from apriori.ico.utils.data.batcher import IcoBatcher

RenderColumn: TypeAlias = Literal["Flow", "Name", "Type", "Signature", "State"]
CallableFormat: TypeAlias = Literal["__name__", "str()"]
SignatureFormat: TypeAlias = Literal["Full", "Input", "Output"]


@dataclass(slots=True)
class RenderOptions:
    include_runtime: bool = True
    callable_format: CallableFormat = "__name__"
    dim_ico_nodes: bool = False
    query_iterable_size: bool = True
    show_ico_operator: bool = False
    expand_subflows: bool = True
    expand_subflow_factories: bool = True
    signature_format: SignatureFormat = "Full"

    columns: list[RenderColumn] = field(
        default_factory=lambda: ["Flow", "Signature", "State", "Name"]
    )

    flatten_node_type: set[type[IcoNode]] = field(
        default_factory=lambda: {IcoChain, IcoPipeline}
    )

    show_node_icons: bool = True

    node_icons: dict[type[IcoNode], str] = field(
        default_factory=lambda: {
            MPProcess: "👷",
            IcoBatcher: "📦",
            IcoStream: "🎞️ ",
            IcoProcess: "🔁",
            IcoSource: "📚",
            IcoSink: "🏁",
        }
    )
