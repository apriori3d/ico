from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from typing import ClassVar

from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.runtime.command import (
    IcoRunCommand,
    IcoRuntimeCommand,
)
from apriori.ico.core.runtime.node import (
    IcoRuntimeNode,
    iterate_parents,
)


class IcoRuntimeContour(IcoRuntimeNode):
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

    runtime_type_name: ClassVar[str] = "Runtime Contour"

    _closure: IcoOperator[None, None]

    def __init__(
        self,
        closure: IcoOperator[None, None],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(runtime_name=name)
        self._closure = closure

        discover_and_connect_runtime_subtrees(self, closure)

    def on_command(self, command: IcoRuntimeCommand) -> IcoRuntimeCommand:
        if isinstance(command, IcoRunCommand):
            self.run()
        return super().on_command(command)

    # ─── Execution ───

    def run(self) -> IcoRuntimeContour:
        """Execute the entire runtime contour."""
        try:
            self.state_model.running()
            self._contour_fn(None)

            self.state_model.ready()
            return self

        except Exception as e:
            self.state_model.fault()
            raise e

    def _contour_fn(self, _: None) -> None:
        self._closure(None)


# ────────────────────────────────────────────────
# Runtime Discovery and Connection Utilities
# ────────────────────────────────────────────────


def discover_runtime_subtrees(flow: IcoNode) -> list[IcoRuntimeNode]:
    """Discover all runtime hosts within the given closure."""
    all_runtimes = list(_discover_runtime_deep(flow))
    roots = OrderedDict[IcoRuntimeNode, None]()

    for runtime in all_runtimes:
        if runtime.runtime_parent is None:
            roots[runtime] = None
            continue

        all_parents = list(iterate_parents(runtime))
        if len(all_parents) > 0:
            roots[all_parents[-1]] = None

    return list(roots.keys())


def _discover_runtime_deep(
    node: IcoNode,
) -> Iterator[IcoRuntimeNode]:
    """Discover all runtime hosts within the given closure."""

    if isinstance(node, IcoRuntimeNode):
        yield node

    for child in node.children:
        yield from _discover_runtime_deep(child)


def discover_and_connect_runtime_subtrees(
    runtime: IcoRuntimeNode,
    flow: IcoNode,
) -> None:
    """Discover and connect all runtime hosts within the given closure."""
    for nested_runtime in discover_runtime_subtrees(flow):
        runtime.connect_runtime(nested_runtime)
