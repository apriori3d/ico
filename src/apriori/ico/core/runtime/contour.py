from __future__ import annotations

from collections.abc import Sequence
from typing import Generic

from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.runtime.node import (
    IcoRuntimeNode,
)
from apriori.ico.core.runtime.state import BaseStateModel
from apriori.ico.core.runtime.utils import discover_and_connect_runtime_nodes


class IcoRuntimeContour(Generic[I, O], IcoRuntimeNode):
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

    flow: IcoOperator[I, O]

    def __init__(
        self,
        flow: IcoOperator[I, O],
        *,
        name: str | None = None,
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
        state_model: BaseStateModel | None = None,
    ) -> None:
        IcoRuntimeNode.__init__(
            self,
            runtime_name=name,
            runtime_parent=runtime_parent,
            runtime_children=runtime_children,
            state_model=state_model,
        )
        self.flow = flow
        discover_and_connect_runtime_nodes(self, flow)

    # def on_command(self, command: IcoRuntimeCommand) -> IcoRuntimeCommand:
    #     if isinstance(command, IcoRunCommand):
    #         self.run()
    #     return super().on_command(command)

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
