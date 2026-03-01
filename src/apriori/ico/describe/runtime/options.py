from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from apriori.ico.core.node import IcoNode
from apriori.ico.core.runtime.agent import IcoAgent, IcoAgentWorker
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.printer import IcoPrinter
from apriori.ico.core.runtime.progress import IcoProgress
from apriori.ico.core.runtime.runtime import IcoRuntime
from apriori.ico.describe.options import RendererOptions
from apriori.ico.tools.printer.rich_printer_tool import RichPrinterTool
from apriori.ico.tools.progress.rich_progress_tool import RichProgressTool

RuntimeRendererColumn: TypeAlias = Literal["Tree", "State", "Name"]


@dataclass(slots=True)
class RuntimeRendererOptions(RendererOptions):
    """
    Configuration options for runtime tree rendering.

    Extends base RendererOptions with runtime-specific settings including
    column layout, agent expansion, and predefined node icons.
    """

    columns: list[RuntimeRendererColumn] = field(
        default_factory=lambda: ["Tree", "State", "Name"]
    )
    """Display columns: Tree structure, Runtime state, Node names"""

    renderers_paths: list[str] = field(
        default_factory=lambda: ["apriori.ico.describe.runtime.rich_renderer.node"]
    )
    """Paths to runtime renderer modules"""

    expand_agents: bool = True
    """Show internal agent/worker hierarchies"""

    node_icons: dict[type[IcoNode | IcoRuntimeNode], str] = field(
        default_factory=lambda: OrderedDict(
            {
                IcoRuntime: "🚙",
                RichProgressTool: "⏳",
                IcoProgress: "⏳",
                RichPrinterTool: "🖨️ ",
                IcoPrinter: "🖨️ ",
                IcoAgent: "👷",
                IcoAgentWorker: "🤖",
                IcoRuntimeNode: "🏃",
            }
        )
    )
    """Emoji icons for runtime node visualization"""
