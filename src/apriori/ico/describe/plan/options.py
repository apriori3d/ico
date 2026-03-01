from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.batcher import IcoBatcher
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

PlanRendererColumn: TypeAlias = Literal["Flow", "Signature", "Name"]
CallableFormat: TypeAlias = Literal["__name__", "str()"]
SignatureFormat: TypeAlias = Literal["Full", "Input", "Output"]


@dataclass(slots=True)
class PlanRendererOptions(RendererOptions):
    """
    Configuration options for computation flow plan rendering.

    Extends base RendererOptions with plan-specific settings like
    column layout, signature display format, and node flattening rules.
    """

    columns: list[PlanRendererColumn] = field(
        default_factory=lambda: ["Flow", "Signature", "Name"]
    )
    """Display columns: Flow tree, Type signatures, Node names"""

    renderers_paths: list[str] = field(
        default_factory=lambda: ["apriori.ico.describe.plan.rich_renderer.node"]
    )
    """Paths to renderer modules for auto-import"""

    callable_format: CallableFormat = "__name__"
    """Function display format: '__name__' or 'str()'"""

    signature_format: SignatureFormat = "Full"
    """Type signature display: 'Full', 'Input', or 'Output'"""

    query_iterable_size: bool = True
    """Show collection sizes when available"""

    show_remote_flows: bool = True
    """Display flows inside distributed agents"""

    dim_ico_nodes: bool = False
    """Dim ICO framework nodes to focus on user code"""

    show_ico_operator: bool = False
    """Show IcoOperator wrapper details"""

    show_node_icons: bool = True
    """Display emoji icons for node types"""

    flatten_node_type: set[type[IcoNode]] = field(
        default_factory=lambda: {
            IcoChain,
            IcoPipeline,
            IcoContextPipeline,
            IcoRuntimeWrapper,
        }
    )
    """Node types to flatten/unwrap for cleaner display"""

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
    """Emoji icons for visual node type identification"""
