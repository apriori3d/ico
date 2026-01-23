from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from apriori.ico.core.node import IcoNode
from apriori.ico.core.runtime.agent import IcoAgent, IcoAgentWorker
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.progress import IcoProgress
from apriori.ico.core.runtime.runtime import IcoRuntime
from apriori.ico.describe.options import RendererOptions
from apriori.ico.tools.printer.node import IcoPrinter
from apriori.ico.tools.printer.rich.rich_printer_tool import RichPrinterTool
from apriori.ico.tools.progress.rich_progress_tool import RichProgressTool

RuntimeRendererColumn: TypeAlias = Literal["Tree", "State", "Name"]


@dataclass(slots=True)
class RuntimeRendererOptions(RendererOptions):
    columns: list[RuntimeRendererColumn] = field(
        default_factory=lambda: ["Tree", "State", "Name"]
    )
    renderers_paths: list[str] = field(
        default_factory=lambda: ["apriori.ico.describe.runtime.rich_renderer.node"]
    )

    expand_agents: bool = True

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
