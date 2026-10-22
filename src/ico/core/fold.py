from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar, get_args, get_origin

from ico.core.operator import I, IcoOperator, IcoOperatorProtocol, wrap_operator
from ico.core.runtime.cancel import check_cancelled
from ico.core.signature import IcoSignature

C = TypeVar("C")  # noqa: E741


class IcoFold(
    Generic[I, C],
    IcoOperator[C, C],
):
    """A fold over the items of a source: ``C → C``.

    For each item of ``source`` the state becomes ``step((item, state))``;
    the fold returns the last state. ``step`` is an ordinary operator over a
    pair, ``tuple[I, C] → C`` (a chain too: a training step followed by
    logging and a checkpoint), so profilers, samplers and plans see it as
    any other operator. A training epoch is a fold: the state is the model
    with its optimizer, the items are batches.

    Generic Parameters:
        I: The type of the source's items.
        C: The type of the state.

    ICO signature:
        C → C (via Iterator[I] and tuple[I, C] → C for each item)

    Example:
        >>> from collections.abc import Iterator
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Stats:
        ...     total: int = 0
        ...     count: int = 0
        >>> def numbers() -> Iterator[int]:
        ...     return iter([1, 2, 3, 4, 5])
        >>> def add(pair: tuple[int, Stats]) -> Stats:
        ...     number, stats = pair
        ...     return Stats(stats.total + number, stats.count + 1)
        >>> fold = IcoFold(lambda _: numbers(), add)
        >>> fold(Stats())
        Stats(total=15, count=5)

    Attributes:
        source: The operator that gives the items (``None → Iterator[I]``).
        step: The operator that folds an item into the state
            (``tuple[I, C] → C``).

    Note:
        A cancelled run stops between items (``IcoCancelled``).
    """

    source: IcoOperatorProtocol[None, Iterator[I]]
    step: IcoOperatorProtocol[tuple[I, C], C]

    def __init__(
        self,
        source: Callable[[None], Iterator[I]],
        step: Callable[[tuple[I, C]], C],
        *,
        name: str | None = None,
    ) -> None:
        """Args:
        source: Gives the items; called with ``None`` (a source or a flow
            that starts with one).
        step: Folds an item into the state; a plain callable is wrapped
            into an operator.
        name: Optional name for plans and progress.
        """
        source_op = wrap_operator(source)
        step_op = wrap_operator(step)
        super().__init__(fn=self._fold, name=name, children=[source_op, step_op])
        self.source = source_op
        self.step = step_op

    def _fold(self, state: C) -> C:
        for item in self.source(None):
            check_cancelled()
            state = self.step((item, state))
        return state

    @property
    def signature(self) -> IcoSignature:
        """``C → C``, with ``C`` from the output of ``step``."""
        signature = super().signature
        if signature.infered:
            return signature
        state: Any = self.step.signature.o
        return IcoSignature(i=state, o=state, infered=self.step.signature.infered)


def fold_input(fold: IcoFold[Any, Any]) -> Any | None:
    """The input ``step`` should take, ``tuple[I, C]``: ``I`` from the items
    of ``source``, ``C`` from the output of ``step``; ``None`` when either
    is not known."""
    source, step = fold.source.signature, fold.step.signature
    if not (source.infered and step.infered):
        return None
    if get_origin(source.o) is None or not get_args(source.o):
        return None
    item: Any = get_args(source.o)[0]
    state: Any = step.o
    return tuple[item, state]
