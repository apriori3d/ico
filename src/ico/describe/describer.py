from __future__ import annotations

from typing import overload

from rich.console import Console

from ico.core.node import IcoNode, IcoNodeProtocol
from ico.core.runtime.node import IcoRuntimeNode
from ico.describe import (
    PlanRendererDefaultOptions,
    RuntimeRendererDefaultOptions,
)
from ico.describe.plan.options import PlanRendererOptions
from ico.describe.runtime.options import RuntimeRendererOptions


@overload
def describe(
    node: IcoNodeProtocol,
    *,
    console: Console | None = None,
    options: PlanRendererOptions | None = None,
) -> None: ...


@overload
def describe(
    node: IcoRuntimeNode,
    *,
    console: Console | None = None,
    options: RuntimeRendererOptions | None = None,
) -> None: ...


def describe(
    node: IcoNodeProtocol | IcoRuntimeNode,
    *,
    console: Console | None = None,
    options: PlanRendererOptions | RuntimeRendererOptions | None = None,
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
    match node, options:
        case IcoNodeProtocol(), None:
            _render_plan(node, console, PlanRendererDefaultOptions)

        case IcoNodeProtocol(), PlanRendererOptions():
            _render_plan(node, console, options)

        case IcoNode(), _:
            raise ValueError(
                "Describe of IcoNode requires options to be an instance of PlanRendererOptions class."
            )

        case IcoRuntimeNode(), None:
            _render_runtime(node, console, RuntimeRendererDefaultOptions)

        case IcoRuntimeNode(), RuntimeRendererOptions():
            _render_runtime(node, console, options)

        case IcoRuntimeNode(), _:
            raise ValueError(
                "Describe of IcoRuntimeNode requires options to be an instance of RuntimeRendererOptions class."
            )


def _render_plan(
    node: IcoNodeProtocol,
    console: Console | None,
    options: PlanRendererOptions,
) -> None:
    """Render computation flow plan."""
    match options.backend:
        case "RichText":
            from ico.describe.plan.rich_renderer.plan_renderer import (
                PlanRenderer,
            )

            renderer = PlanRenderer(console=console, options=options)
            renderer.render(node)

        case _:
            raise ValueError("Backend is not yet supported in API.")


def _render_runtime(
    node: IcoRuntimeNode,
    console: Console | None,
    options: RuntimeRendererOptions,
) -> None:
    """Render runtime tree."""
    match options.backend:
        case "RichText":
            from ico.describe.runtime.rich_renderer.tree_renderer import (
                RuntimeTreeRenderer,
            )

            renderer = RuntimeTreeRenderer(console=console, options=options)
            renderer.render(node)

        case _:
            raise ValueError("Backend is not yet supported in API.")
