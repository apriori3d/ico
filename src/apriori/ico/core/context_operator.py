from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import ClassVar, Generic, TypeVar, overload

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
    type_name: ClassVar[str] = "Context Operator"

    fn: Callable[[I, C], O]

    def __init__(
        self,
        fn: Callable[[I, C], O],
        *,
        name: str | None = None,
        parent: IcoNode | None = None,
        children: Sequence[IcoNode] | None = None,
    ):
        super().__init__(name=name, parent=parent, children=children)
        self.fn = fn

    def __call__(self, item: I, context: C) -> O:
        return self.fn(item, context)


# ─────────────────────────────────────────────
# Operator Wrapping Utility
# ─────────────────────────────────────────────


def context_operator() -> Callable[[Callable[[I, C], C]], IcoContextOperator[I, C, C]]:
    def decorator(fn: Callable[[I, C], C]) -> IcoContextOperator[I, C, C]:
        return wrap_context_operator(fn)

    return decorator


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
