from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic

from ico.core.operator import IcoOperator, IcoOperatorProtocol, IInv, wrap_operator
from ico.core.signature import IcoSignature


class IcoTap(Generic[IInv], IcoOperator[IInv, IInv]):
    """Run a child for its side effect and pass the input on.

    For steps whose result is on disk (masks, cleaned clips) while the next
    step needs the same input again::

        master → tap(scan) → master → tap(refine) → master → assemble

    The child's output is dropped; an iterator output is consumed to the
    end first, so that its side effects happen.

    Generic Parameters:
        I: The input, passed on unchanged.
    """

    __slots__ = ("child",)

    def __init__(
        self, child: Callable[[IInv], object], *, name: str | None = None
    ) -> None:
        child_op = wrap_operator(child)
        super().__init__(fn=self._tap_fn, name=name, children=[child_op])
        self.child = child_op

    def _tap_fn(self, item: IInv) -> IInv:
        result = self.child(item)
        if isinstance(result, Iterator):
            for _ in result:  # pyright: ignore[reportUnknownVariableType]
                pass
        return item

    @property
    def signature(self) -> IcoSignature:
        """``I → I``, from the child's input."""
        signature = super().signature
        if signature.infered:
            return signature
        child = self.child.signature
        if not child.infered:
            return child
        return IcoSignature(i=child.i, c=None, o=child.i)


def tap(
    child: Callable[[IInv], object], *, name: str | None = None
) -> IcoOperatorProtocol[IInv, IInv]:
    """Run ``child`` for its side effect and pass the input on (see
    :class:`IcoTap`)."""
    return IcoTap[Any](child, name=name)
