from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.chain import IcoChain
from apriori.ico.core.context_pipeline import IcoContextPipeline
from apriori.ico.core.epoch import IcoEpoch
from apriori.ico.core.node import IcoNode
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.process import IcoProcess
from apriori.ico.core.runtime.agent import IcoAgent
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.progress import IcoProgress
from apriori.ico.core.runtime.runtime_wrapper import IcoRuntimeWrapper
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.source import IcoSource
from apriori.ico.core.stream import IcoStream
from apriori.ico.describe.options import RendererOptions
from apriori.ico.utils.data.batcher import IcoBatcher

PlanRendererColumn: TypeAlias = Literal["Flow", "Signature", "Name"]
CallableFormat: TypeAlias = Literal["__name__", "str()"]
SignatureFormat: TypeAlias = Literal["Full", "Input", "Output"]


@dataclass(slots=True)
class PlanRendererOptions(RendererOptions):
    columns: list[PlanRendererColumn] = field(
        default_factory=lambda: ["Flow", "Signature", "Name"]
    )
    renderers_paths: list[str] = field(
        default_factory=lambda: ["apriori.ico.describe.plan.rich_renderer.node"]
    )

    callable_format: CallableFormat = "__name__"
    signature_format: SignatureFormat = "Full"

    query_iterable_size: bool = True
    show_remote_flows: bool = True

    dim_ico_nodes: bool = False
    show_ico_operator: bool = False
    show_node_icons: bool = True

    flatten_node_type: set[type[IcoNode]] = field(
        default_factory=lambda: {
            IcoChain,
            IcoPipeline,
            IcoContextPipeline,
            IcoRuntimeWrapper,
        }
    )

    node_icons: dict[type[IcoNode | IcoRuntimeNode], str] = field(
        default_factory=lambda: {
            IcoAgent: "👷",
            IcoBatcher: "📦",
            IcoStream: "🎞️ ",
            IcoAsyncStream: "🚀",
            IcoProcess: "🔁",
            IcoSource: "📚",
            IcoSink: "🏁",
            IcoProgress: "⏳",
            IcoEpoch: "🧠",
        }
    )
