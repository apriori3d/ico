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
    """
    Display visual representation of ICO nodes and runtime trees.

    Main entry point for rendering ICO computation flows and runtime states
    using Rich console formatting. Automatically selects appropriate renderer
    based on node type.

    Args:
        node: IcoNode (computation flow) or IcoRuntimeNode (runtime tree)
        console: Rich Console instance (creates default if None)
        options: Rendering options (auto-selected by node type if None)

    Usage:
        ```python
        # Describe computation flow
        flow = source | operator | sink
        describe(flow)

        # Describe runtime tree
        runtime = IcoRuntime(flow)
        runtime.activate()
        describe(runtime)

        # Custom console and options
        console = Console(width=120)
        options = PlanRendererOptions(show_signatures=True)
        describe(flow, console=console, options=options)
        ```

    Renderers:
        - IcoNode: Uses PlanRenderer for computation flow visualization
        - IcoRuntimeNode: Uses RuntimeTreeRenderer for runtime state display

    Note: Currently supports only RichText backend
    """
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
