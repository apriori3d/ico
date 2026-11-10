"""Fields of an operator class that the plan shows next to its name."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=type)

_SHOWN = "__ico_shown_params__"


def show_params(*names: str) -> Callable[[T], T]:
    """Class decorator: the plan (``describe``, the Live Board) shows these
    fields of an operator next to its class name, values by ``repr``.

    Subclasses inherit the list; a subclass may declare its own.

    Example:
        >>> @show_params("factor")
        ... class Scale(IcoOperator[float, float]):
        ...     def __init__(self, factor: float) -> None:
        ...         self.factor = factor
        ...         super().__init__(fn=self._apply)
        ...     def _apply(self, x: float) -> float:
        ...         return x * self.factor
        >>> # the plan row: Scale(factor=2.0)
    """

    def decorator(cls: T) -> T:
        setattr(cls, _SHOWN, names)
        return cls

    return decorator


def shown_params(node: object) -> tuple[str, ...]:
    """The fields of a node that the plan shows (``show_params``), if any."""
    names: tuple[str, ...] = getattr(type(node), _SHOWN, ())
    return names
