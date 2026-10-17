"""JSON backend of ``describe`` for flows: a render-neutral plan view.

:func:`plan_view` builds a JSON-compatible tree from the same traversal and
per-node-type renderers as the RichText plan renderer, so a consumer (e.g.
a browser board) can render a flow without knowing node kinds::

    {"label": "IcoAsyncStream", "sig": "Iterator[int] → Iterator[str]",
     "ref": None, "badges": ["pool_size=4"], "layout": "parallel",
     "children": [...]}

- ``label``: the node name, else the function name of a plain operator,
  else the class name.
- ``sig``: the signature as in the Signature column, ``None`` for nodes
  without one.
- ``ref``: ``"name@version"`` for registered operators. They are opaque
  leaves, as in a spec.
- ``badges``: node settings as the console shows them in parentheses
  (``pool_size=4``, ``until=done, max_iter=10`` → two badges).
- ``layout``: how to draw the children: ``sequence`` (one after another),
  ``parallel`` (side by side), ``loop`` (repeated) or ``leaf``.

Structural nodes (chains, pipelines) are flattened into their parent, like
in the console plan.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, TypedDict

from ico.core.async_stream import IcoAsyncStream
from ico.core.context_operator import IcoContextOperator
from ico.core.epoch import IcoEpoch
from ico.core.fork import IcoFork
from ico.core.node import IcoNodeProtocol
from ico.core.operator import IcoOperator
from ico.core.partition import IcoPartition
from ico.core.process import IcoProcess
from ico.core.route import IcoRoute
from ico.describe.plan.options import PlanRendererOptions
from ico.describe.plan.rich_renderer.group_renderer import GroupRenderer
from ico.describe.plan.rich_renderer.render_target import create_plan_walker
from ico.describe.plan.rich_renderer.renderer_registry import select_renderer
from ico.describe.plan.rich_renderer.row_renderer import RowRenderer
from ico.describe.utils import import_all_renderers
from ico.registry.registry import get_meta

Layout = Literal["sequence", "parallel", "loop", "leaf"]


class PlanNodeView(TypedDict):
    """One node of a plan view; JSON-compatible."""

    label: str
    sig: str | None
    ref: str | None
    badges: list[str]
    layout: Layout
    children: list[PlanNodeView]


_PARALLEL: tuple[type[object], ...] = (IcoAsyncStream, IcoFork, IcoRoute, IcoPartition)
_LOOP: tuple[type[object], ...] = (IcoProcess, IcoEpoch)
_PLAIN_OPERATORS: tuple[type[object], ...] = (IcoOperator, IcoContextOperator)


def plan_view(
    node: IcoNodeProtocol, options: PlanRendererOptions | None = None
) -> PlanNodeView:
    """The plan of a flow as a JSON-compatible tree.

    Args:
        node: The flow; the root is never flattened.
        options: Plan options; only ``renderers_paths``, ``show_remote_flows``,
            ``flatten_node_type`` and ``callable_format`` apply. By default
            sources are not called to query their size.

    Example:
        >>> view = plan_view(source | batcher | stream)
        >>> view["layout"], [c["label"] for c in view["children"]]
        ('sequence', ['IcoSource', 'IcoBatcher', 'IcoStream'])
    """
    return _PlanViewBuilder(
        options or PlanRendererOptions(query_iterable_size=False)
    ).view(node)


class _PlanViewBuilder:
    def __init__(self, options: PlanRendererOptions) -> None:
        self.options = replace(options, show_node_icons=False)
        self.walker = create_plan_walker(
            expand_remote_flows=self.options.show_remote_flows
        )
        self._rows: dict[type[object], RowRenderer] = {}
        for path in self.options.renderers_paths:
            import_all_renderers(path)

    def view(self, node: IcoNodeProtocol) -> PlanNodeView:
        meta = get_meta(node)
        children = [] if meta is not None else self._children(node)
        args = self._args(node)
        plain = type(node) in _PLAIN_OPERATORS
        signature = RowRenderer(self.options).render_signature_column(node).plain
        return {
            "label": node.name or (args if plain and args else type(node).__name__),
            "sig": signature or None,
            "ref": meta.ref if meta is not None else None,
            "badges": [] if plain or not args else args.split(", "),
            "layout": _layout(node, has_children=bool(children)),
            "children": [self.view(child) for child in children],
        }

    def _children(self, node: IcoNodeProtocol) -> list[IcoNodeProtocol]:
        """Children with structural nodes replaced by their own children."""
        out: list[IcoNodeProtocol] = []
        for child in self.walker.get_children_fn(node):
            if (
                type(child) in self.options.flatten_node_type
                and get_meta(child) is None
            ):
                out.extend(self._children(child))
            else:
                out.append(child)
        return out

    def _args(self, node: IcoNodeProtocol) -> str:
        """Settings shown in parentheses by the console renderer."""
        row = self._row(type(node))
        text = row._render_node_args_info(node)  # pyright: ignore[reportPrivateUsage]  # the console's own settings text
        return text.plain if text is not None else ""

    def _row(self, node_type: type[object]) -> RowRenderer:
        """The row renderer the console uses for a node's own line."""
        row = self._rows.get(node_type)
        if row is None:
            renderer_class = select_renderer(node_type)
            renderer = renderer_class(self.options) if renderer_class else None
            if isinstance(renderer, RowRenderer):
                row = renderer
            elif isinstance(renderer, GroupRenderer):
                row = renderer.header_renderer
            else:  # default or custom renderers
                row = RowRenderer(self.options)
            self._rows[node_type] = row
        return row


def _layout(node: IcoNodeProtocol, *, has_children: bool) -> Layout:
    if not has_children:
        return "leaf"
    if isinstance(node, _PARALLEL):
        return "parallel"
    if isinstance(node, _LOOP):
        return "loop"
    return "sequence"
