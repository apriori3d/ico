from __future__ import annotations

from dataclasses import replace
from typing import Literal, TypeAlias, overload

from rich.console import Console

from apriori.ico.core.node import IcoNode
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.describe.plan.options import RenderOptions

RenderBackend: TypeAlias = Literal["RichText",]


@overload
def describe(
    node: IcoNode,
    *,
    show_runtime_nodes: bool = False,
    console: Console | None = None,
    backend: RenderBackend = "RichText",
) -> None: ...


@overload
def describe(
    node: IcoRuntimeNode,
    *,
    console: Console | None = None,
    backend: RenderBackend = "RichText",
) -> None: ...


def describe(
    node: IcoNode | IcoRuntimeNode,
    *,
    show_runtime_nodes: bool = False,
    console: Console | None = None,
    backend: RenderBackend = "RichText",
) -> None:
    if isinstance(node, IcoNode):
        if backend == "RichText":
            from apriori.ico.describe.plan.rich_render.plan_renderer import (
                PlanRenderer,
            )

            options = replace(RenderOptions(), show_runtime_nodes=show_runtime_nodes)
            renderer = PlanRenderer(console=console, options=options)
            renderer.render(node)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    else:
        raise NotImplementedError("Runtime node description is not yet implemented.")
