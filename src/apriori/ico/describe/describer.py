from __future__ import annotations

from rich.console import Console

from apriori.ico.core.node import IcoNode
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.describe.options import RendererOptions
from apriori.ico.describe.plan.options import PlanRendererOptions
from apriori.ico.describe.runtime.options import RuntimeRendererOptions


def describe(
    node: IcoNode | IcoRuntimeNode,
    *,
    console: Console | None = None,
    options: RendererOptions | None = None,
) -> None:
    if isinstance(node, IcoNode):
        options = options or PlanRendererOptions()

        if not isinstance(options, PlanRendererOptions):
            raise ValueError(
                "Describe of IcoNode requires options to be an instance of PlanRendererOptions class."
            )

        if options.backend == "RichText":
            from apriori.ico.describe.plan.rich_renderer.plan_renderer import (
                PlanRenderer,
            )

            renderer = PlanRenderer(console=console, options=options)
            renderer.render(node)
        else:
            raise ValueError("Backend is not yet supported in API.")

    else:
        options = options or RuntimeRendererOptions()

        if not isinstance(options, RuntimeRendererOptions):
            raise ValueError(
                "Describe of IcoRuntimeNode requires options to be an instance of RuntimeRendererOptions class."
            )

        if options.backend == "RichText":
            from apriori.ico.describe.runtime.rich_renderer.tree_renderer import (
                RuntimeTreeRenderer,
            )

            renderer = RuntimeTreeRenderer(console=console, options=options)
            renderer.render(node)
        else:
            raise ValueError("Backend is not yet supported in API.")
