"""Plan renderers for nodes with branches: ``IcoFork`` and ``IcoRoute``.

Each branch gets a row of its own (``├─▸ branch 1``, ``├─▸ case 'a'``) and
its flow is rendered below it, so chains in different branches never run
together::

    ╭── fork into 🔀IcoFork(branches=2)   str → tuple[str, int]
    ├─▸ branch 1                          str → str
    │   strip                             str → str
    │   upper                             str → str
    ├─▸ branch 2                          str → int
    │   length                            str → int
    ╰─▸ collect                           tuple[str, int]
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import replace
from typing import Any, cast

from rich.text import Text

from ico.core.fork import IcoFork
from ico.core.node import IcoNodeProtocol
from ico.core.route import IcoRoute
from ico.describe.plan.options import PlanRendererOptions, SignatureFormat
from ico.describe.plan.rich_renderer.custom_renderer import CustomRenderer
from ico.describe.plan.rich_renderer.render_target import (
    PlanRenderTarget,
    create_plan_walker,
)
from ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from ico.describe.plan.rich_renderer.row_renderer import RowRenderer
from ico.describe.rich_style import DescribeStyle


class BranchingRenderer(CustomRenderer):
    """A header row, one labelled row per branch with the branch flow below
    it, and a footer row."""

    header: str
    """Keyword before the node, e.g. ``"fork into "``."""
    footer: str
    """Keyword of the closing row, e.g. ``"collect"``."""

    def __init__(self, options: PlanRendererOptions) -> None:
        super().__init__(options=options)
        renderer = self

        class HeaderRenderer(RowRenderer):
            def _render_node_args_info(self, node: IcoNodeProtocol) -> Text:
                return Text(renderer.settings(node), style=DescribeStyle.meta.value)

        self._header = HeaderRenderer(
            options=options,
            flow_column_prefix=Text(self.header, style=DescribeStyle.keyword.value),
        )
        self._footer = _part_row(
            options,
            Text("╰─▸ ", style=DescribeStyle.group.value),
            self.footer,
            "Output",
        )

    @abstractmethod
    def settings(self, node: IcoNodeProtocol) -> str:
        """The node's settings, shown in parentheses on the header row."""

    @abstractmethod
    def branches(self, node: IcoNodeProtocol) -> list[tuple[str, IcoNodeProtocol]]:
        """The branches in order, with the label of their row."""

    def render(self, plan: PlanRenderTarget, node: IcoNodeProtocol) -> None:
        plan.render_row(
            self._header, node, indent=Text("╭── ", style=DescribeStyle.group.value)
        )
        walker = create_plan_walker(expand_remote_flows=self.options.show_remote_flows)
        for label, branch in self.branches(node):
            row = _part_row(
                self.options,
                Text("├─▸ ", style=DescribeStyle.group.value),
                label,
                "Full",
            )
            plan.render_row(row, branch)
            plan.push_group_indent(Text("│   ", style=DescribeStyle.group.value))
            walker.walk(branch, visit_fn=plan.render_node, order="pre_post")
            plan.pop_group_indent()
        plan.render_row(self._footer, node)


@register_renderer(IcoFork)
class IcoForkRenderer(BranchingRenderer):
    """Every branch receives the same input; the outputs form a tuple."""

    header = "fork into "
    footer = "collect"

    def settings(self, node: IcoNodeProtocol) -> str:
        return f"branches={len(node.children)}"

    def branches(self, node: IcoNodeProtocol) -> list[tuple[str, IcoNodeProtocol]]:
        return [(f"branch {i}", b) for i, b in enumerate(node.children, start=1)]


@register_renderer(IcoRoute)
class IcoRouteRenderer(BranchingRenderer):
    """The key function selects the one branch an item goes to."""

    header = "route by "
    footer = "merge"

    def settings(self, node: IcoNodeProtocol) -> str:
        route = cast(IcoRoute[Any, Any, Any], node)
        keys = ", ".join(repr(key) for key in route.routes)
        return f"keys={keys}" + (", default" if route.default is not None else "")

    def branches(self, node: IcoNodeProtocol) -> list[tuple[str, IcoNodeProtocol]]:
        route = cast(IcoRoute[Any, Any, Any], node)
        out: list[tuple[str, IcoNodeProtocol]] = [("key", route.key)]
        out.extend((f"case {key!r}", branch) for key, branch in route.routes.items())
        if route.default is not None:
            out.append(("default", route.default))
        return out


def _part_row(
    options: PlanRendererOptions, lead: Text, keyword: str, signature: SignatureFormat
) -> RowRenderer:
    """A row with only a keyword and the node's signature."""
    return RowRenderer(
        flow_column_prefix=lead + Text(keyword, style=DescribeStyle.keyword.value),
        options=replace(options, signature_format=signature),
        show_name_column=False,
        show_type_column=False,
        show_state_column=False,
        flow_includes_node_info=False,
    )
