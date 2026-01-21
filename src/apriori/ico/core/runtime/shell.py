from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

from apriori.ico.core.node import IcoNode, create_flow_tree_walker
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import (
    IcoRuntimeNode,
)
from apriori.ico.core.runtime.state import BaseStateModel
from apriori.ico.core.runtime.toolbox import IcoToolBox


class IcoShell(IcoRuntimeNode):
    """
    Runtime contour that encapsulates a complete ICO flow.

    ─── ICO Forms ───
        Contour: () → ()
        Flow:    (None → Iterable[O]) → ... → (Iterable[I] → None)

    ─── Description ───
    The contour wraps a full operator flow (called *Flow*) and provides a
    runtime layer responsible for:
        • broadcasting lifecycle events (prepare → reset → cleanup)
        • binding shared progress relays to operators
        • executing the entire flow (() → ())

    The flow itself must begin with a Source (None→Iterable[O])
    and terminate with a Sink (Iterable[O]→None), forming a closed runtime loop.

    ─── Example ───
        >>> from apriori.ico.core import IcoSource, IcoSink, IcoOperator, IcoRuntimeContour

        # Declarative flow: Source → Operator → Sink
        >>> source = IcoSource[int](lambda: range(3), name="dataset")
        >>> double = IcoOperator[int, int](lambda x: x * 2, name="double")
        >>> sink = IcoSink[int](lambda xs: print(list(xs)), name="printer")

        # Compose into a flow and wrap in contour (() → ())
        >>> flow = source | double.map() | sink
        >>> contour = IcoRuntimeContour(flow)

        >>> contour.ready().run().idle()
        [printer] Sink received: [0, 2, 4]

    """

    closure: IcoOperator[None, None]
    _toolbox: IcoToolBox | None

    def __init__(
        self,
        closure: IcoOperator[None, None],
        *,
        name: str | None = None,
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
        state_model: BaseStateModel | None = None,
        tools: Sequence[IcoRuntimeNode] | None = None,
    ) -> None:
        IcoRuntimeNode.__init__(
            self,
            runtime_name=name,
            runtime_parent=runtime_parent,
            runtime_children=runtime_children,
            state_model=state_model,
        )
        self.closure = closure

        _discover_and_connect_runtime_nodes(self, closure)

        if tools is not None:
            self._toolbox = IcoToolBox(shell=self, tools=tools)
            self.add_runtime_children(self._toolbox)
            self._toolbox.register_tools()
        else:
            self._toolbox = None

    # ────── Tools management ──────

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        super().on_event(event)

        return self._toolbox.on_event(event) if self._toolbox else event

    # # ─── Execution ───

    # def run(self) -> IcoRuntimeContour:
    #     """Execute the entire runtime contour."""
    #     try:
    #         self.state_model.running()
    #         self._contour_fn(None)

    #         self.state_model.ready()
    #         return self

    #     except Exception as e:
    #         self.state_model.fault()
    #         raise e

    # def _contour_fn(self, _: None) -> None:
    #     self._closure(None)


# ────────────────────────────────────────────────
# Runtime Nodes Discovery and Connection Utilities
# ────────────────────────────────────────────────


def _discover_and_connect_runtime_nodes(
    runtime_node: IcoRuntimeNode, flow: IcoNode
) -> None:
    """Discover and connect all runtime hosts within the given flow."""
    for nested_runtime in _discover_runtime_subtrees(flow):
        runtime_node.add_runtime_children(nested_runtime)


def _discover_runtime_subtrees(flow: IcoNode) -> list[IcoRuntimeNode]:
    """Discover all runtime hosts within the given flow."""
    roots = OrderedDict[IcoRuntimeNode, None]()
    walker = create_flow_tree_walker(visit_subflows=False)

    for node in walker.traverse(flow):
        if not isinstance(node, IcoRuntimeNode):
            continue

        if node.runtime_parent is None:
            roots[node] = None
            continue

        all_parents = list(node.iterate_parents())
        if len(all_parents) > 0:
            roots[all_parents[-1]] = None

    return list(roots.keys())
