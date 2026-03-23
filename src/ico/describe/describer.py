from __future__ import annotations

from rich.console import Console

from ico.core.node import IcoNode
from ico.core.runtime.node import IcoRuntimeNode
from ico.describe import (
    PlanRendererDefaultOptions,
    RuntimeRendererDefaultOptions,
)
from ico.describe.options import RendererOptions
from ico.describe.plan.options import PlanRendererOptions
from ico.describe.runtime.options import RuntimeRendererOptions


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
        if options and not isinstance(options, PlanRendererOptions):
            raise ValueError(
                "Describe of IcoNode requires options to be an instance of PlanRendererOptions class."
            )
        options = options or PlanRendererDefaultOptions

        if options.backend == "RichText":
            from ico.describe.plan.rich_renderer.plan_renderer import (
                PlanRenderer,
            )

            plan_renderer = PlanRenderer(console=console, options=options)
            plan_renderer.render(node)
        else:
            raise ValueError("Backend is not yet supported in API.")

    else:
        if options and not isinstance(options, RuntimeRendererOptions):
            raise ValueError(
                "Describe of IcoRuntimeNode requires options to be an instance of RuntimeRendererOptions class."
            )

        options = options or RuntimeRendererDefaultOptions

        if options.backend == "RichText":
            from ico.describe.runtime.rich_renderer.tree_renderer import (
                RuntimeTreeRenderer,
            )

            runtime_renderer = RuntimeTreeRenderer(console=console, options=options)
            runtime_renderer.render(node)
        else:
            raise ValueError("Backend is not yet supported in API.")
