from typing import Any

from typing_extensions import Self

from apriori.ico.core.dsl.tree import iterate_nodes
from apriori.ico.core.meta.ico_form import infer_ico_form
from apriori.ico.core.runtime.discovery import discover_and_connect_runtimes
from apriori.ico.core.runtime.progress.mixin import ProgressMixin
from apriori.ico.core.runtime.progress.types import ProgressProtocol, SupportsProgress
from apriori.ico.core.runtime.runtime_operator import IcoRuntimeOperator
from apriori.ico.core.types import IcoOperatorProtocol


class IcoRuntimeContour(
    IcoRuntimeOperator,
    ProgressMixin,
):
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

    _closure: IcoOperatorProtocol[None, None]

    def __init__(
        self,
        closure: IcoOperatorProtocol[None, None],
        name: str | None = None,
    ) -> None:
        # Contour executes the given closure e.g. flow () → ()
        self._validate_closure(closure)

        super().__init__(
            fn=self._contour_fn,
            name=name or "ico_runtime_contour",
            children=[closure],
        )
        self._closure = closure
        discover_and_connect_runtimes(self, closure)

    # ─── Execution ───

    def _contour_fn(self, _: None) -> None:
        self._closure(None)

    # ─── Progress ───

    def attach_progress(self, progress: ProgressProtocol) -> Self:
        super().attach_progress(progress)

        for node in iterate_nodes(self._closure):
            if isinstance(node, SupportsProgress):
                node.progress = self.progress

        return self

    # ─── Internal utilities ───

    def _validate_closure(self, flow: IcoOperatorProtocol[Any, Any]) -> None:
        """Validate that the flow is a closure: begins and ends with unit types (() → ())."""
        form = infer_ico_form(flow)
        if not (form.i == "()" and form.o == "()"):
            raise ValueError(
                f"Invalid flow form: expected (() → ()), got ({form.i} → {form.o})"
            )
