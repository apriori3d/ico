from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar, get_args, get_origin

from ico.core.node import IcoNodeProtocol
from ico.core.operator import I, IcoOperator, IcoOperatorProtocol, O, wrap_operator
from ico.core.runtime.cancel import check_cancelled
from ico.core.signature import IcoSignature

C = TypeVar("C")  # noqa: E741


class IcoScan(
    Generic[I, O, C],
    IcoOperator[C, Iterator[tuple[O, C]]],
):
    """A scan over the items of a source: ``C → Iterator[tuple[O, C]]``.

    A fold with an output per item. For each item of ``source`` the step
    gives an output and the next state, ``(out, state) = step((item,
    state))``, and the scan yields ``(out, state)``; the state goes on to
    the next item. ``step`` is an ordinary operator over a pair, ``tuple[I,
    C] → tuple[O, C]``. A per-frame fit with a warm start is a scan: the
    state is the model fitted on the previous frame, the output the fit of
    this one. A consumer that needs only the outputs drops the states
    (``out for out, _ in scan(state)``); the last pair carries the final
    state.

    Generic Parameters:
        I: The type of the source's items.
        O: The type of the outputs.
        C: The type of the state.

    ICO signature:
        C → Iterator[tuple[O, C]] (via Iterator[I] and
        tuple[I, C] → tuple[O, C] for each item)

    Example:
        >>> from collections.abc import Iterator
        >>> def numbers() -> Iterator[int]:
        ...     return iter([1, 2, 3])
        >>> def running(pair: tuple[int, int]) -> tuple[str, int]:
        ...     number, total = pair
        ...     return f"+{number}", total + number
        >>> scan = IcoScan(lambda _: numbers(), running)
        >>> list(scan(10))
        [('+1', 11), ('+2', 13), ('+3', 16)]

    Attributes:
        source: The operator that gives the items (``None → Iterator[I]``).
        step: The operator that gives an output and the next state
            (``tuple[I, C] → tuple[O, C]``).
        until: Stop condition on the state (``C → bool``), or ``None``.

    Note:
        The scan is lazy: an item is pulled when its pair is asked for.
        With ``until`` it stops before pulling the next item once the
        condition holds, as ``IcoFold`` does (an initial state that already
        satisfies it yields nothing); the source is closed when the scan
        stops or its consumer closes it. A cancelled run stops between
        items (``IcoCancelled``). A state mutated in place is the same
        object in every pair.
    """

    source: IcoOperatorProtocol[None, Iterator[I]]
    step: IcoOperatorProtocol[tuple[I, C], tuple[O, C]]
    until: IcoOperatorProtocol[C, Any] | None

    def __init__(
        self,
        source: Callable[[None], Iterator[I]],
        step: Callable[[tuple[I, C]], tuple[O, C]],
        *,
        until: Callable[[C], object] | None = None,
        name: str | None = None,
    ) -> None:
        """Args:
        source: Gives the items; called with ``None`` (a source or a flow
            that starts with one).
        step: Gives the output for an item and the next state; a plain
            callable is wrapped into an operator.
        until: Stop before the next item once ``until(state)`` is truthy.
        name: Optional name for plans and progress.
        """
        source_op = wrap_operator(source)
        step_op = wrap_operator(step)
        until_op = wrap_operator(until) if until is not None else None
        children: list[IcoNodeProtocol] = [source_op, step_op]
        if until_op is not None:
            children.append(until_op)
        super().__init__(fn=self._scan, name=name, children=children)
        self.source = source_op
        self.step = step_op
        self.until = until_op

    def _scan(self, state: C) -> Iterator[tuple[O, C]]:
        until = self.until
        if until is not None and until(state):
            return
        items = self.source(None)
        try:
            for item in items:
                check_cancelled()
                out, state = self.step((item, state))
                yield out, state
                if until is not None and until(state):
                    break
        finally:
            close = getattr(items, "close", None)
            if close is not None:
                close()

    @property
    def signature(self) -> IcoSignature:
        """``C → Iterator[tuple[O, C]]``, from the output of ``step``."""
        signature = super().signature
        if signature.infered:
            return signature
        out_state: Any = self.step.signature.o
        state: Any = scan_state(self)
        if state is None:
            return signature
        return IcoSignature(
            i=state,
            o=Iterator[out_state],
            infered=self.step.signature.infered,
        )


def scan_state(scan: IcoScan[Any, Any, Any]) -> Any | None:
    """The state type ``C``: the second element of the step's output
    ``tuple[O, C]``; ``None`` when it is not known."""
    step = scan.step.signature
    origin: object = get_origin(step.o)
    if not step.infered or origin is not tuple:
        return None
    args = get_args(step.o)
    return args[1] if len(args) == 2 else None


def scan_input(scan: IcoScan[Any, Any, Any]) -> Any | None:
    """The input ``step`` should take, ``tuple[I, C]``: ``I`` from the items
    of ``source``, ``C`` from the output of ``step``; ``None`` when either
    is not known."""
    source = scan.source.signature
    state: Any = scan_state(scan)
    if not source.infered or state is None:
        return None
    if get_origin(source.o) is None or not get_args(source.o):
        return None
    item: Any = get_args(source.o)[0]
    return tuple[item, state]
