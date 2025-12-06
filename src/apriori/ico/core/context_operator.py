from __future__ import annotations

from collections.abc import Callable, Sequence
from types import FunctionType
from typing import Generic, TypeVar, overload

from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import I, O

# ────────────────────────────────────────────────
# Generic type variables for ICO model
# ────────────────────────────────────────────────

C = TypeVar("C")

# ────────────────────────────────────────────────
# Operator Class
# ────────────────────────────────────────────────


class IcoContextOperator(Generic[I, C, O], IcoNode):
    type_name: str = "context_operator"

    fn: Callable[[I, C], O]

    def __init__(
        self,
        fn: Callable[[I, C], O],
        *,
        name: str | None = None,
        ico_form_target: object | None = None,
        parent: IcoNode | None = None,
        children: Sequence[IcoNode] | None = None,
    ):
        if not name:
            cls = getattr(fn, "__class__", None)
            if cls is FunctionType:
                name = getattr(fn, "__name__", None)
            else:
                name = name or getattr(cls, "__name__", None)

        super().__init__(
            name=name,
            parent=parent,
            children=children,
            ico_form_target=ico_form_target or fn,
        )
        self.fn = fn

    def __str__(self) -> str:
        return self.name

    def __call__(self, item: I, context: C) -> O:
        return self.fn(item, context)


# ─────────────────────────────────────────────
# Operator Wrapping Utility
# ─────────────────────────────────────────────


@overload
def wrap_context_operator(
    fn: IcoContextOperator[I, C, O],
) -> IcoContextOperator[I, C, O]: ...


@overload
def wrap_context_operator(fn: Callable[[I, C], O]) -> IcoContextOperator[I, C, O]: ...


def wrap_context_operator(
    fn: Callable[[I, C], O] | IcoContextOperator[I, C, O],
) -> IcoContextOperator[I, C, O]:
    """
    Wrap raw callable into IcoContextOperator only when necessary.
    Ensures type inference for both mypy and pyright.
    """
    if isinstance(fn, IcoContextOperator):
        # Suppress runtime type checker warning,
        # because we know the type is correct here from static analysis.
        return fn  # pyright: ignore[reportUnknownVariableType]
    return IcoContextOperator[I, C, O](fn=fn)
