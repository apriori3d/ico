from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from ico.core.node import IcoNodeProtocol
from ico.core.runtime.node import IcoRuntimeNode
from ico.core.runtime.printer import IcoPrinter
from ico.core.runtime.progress import IcoProgress
from ico.core.runtime.runtime import IcoRuntime
from ico.core.runtime.worker import IcoRunner, IcoWorker
from ico.describe.options import RendererOptions
from ico.tools.printer.rich_printer_tool import RichPrinterTool
from ico.tools.progress.rich_progress_tool import RichProgressTool

RuntimeRendererColumn: TypeAlias = Literal["Tree", "State", "Name"]


@dataclass(slots=True)
class RuntimeRendererOptions(RendererOptions):
    """
    Configuration options for runtime tree rendering.

    Extends base RendererOptions with runtime-specific settings including
    column layout, worker expansion, and predefined node icons.
    """

    columns: list[RuntimeRendererColumn] = field(
        default_factory=lambda: ["Tree", "State", "Name"]
    )
    """Display columns: Tree structure, Runtime state, Node names"""

    renderers_paths: list[str] = field(
        default_factory=lambda: ["ico.describe.runtime.rich_renderer.node"]
    )
    """Paths to runtime renderer modules"""

    expand_workers: bool = True
    """Show internal worker/runner hierarchies"""

    node_icons: dict[type[IcoNodeProtocol | IcoRuntimeNode], str] = field(
        default_factory=lambda: OrderedDict(
            {
                IcoRuntime: "🚙",
                RichProgressTool: "⏳",
                IcoProgress: "⏳",
                RichPrinterTool: "🖨️ ",
                IcoPrinter: "🖨️ ",
                IcoWorker: "👷",
                IcoRunner: "🤖",
                IcoRuntimeNode: "🏃",
            }
        )
    )
    """Emoji icons for runtime node visualization"""
