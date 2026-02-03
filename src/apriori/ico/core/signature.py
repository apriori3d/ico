# In this module me infer ICO forms for possibly untyped callables,
# and have to disable categories of errors related to using Any type in inspect api.
# mypy: disable-error-code=misc
from __future__ import annotations

from dataclasses import dataclass

# ────────────────────────────────────────────────
# Signature descriptions
# ────────────────────────────────────────────────


@dataclass(slots=True)
class IcoSignature:
    """
    A representation of the ICO form of an operator in DSL.
    Possible types are type[Any] or typing generics."""

    i: object
    c: object | None
    o: object
    infered: bool = True

    def format(self) -> str:
        from apriori.ico.core.signature_utils import format_ico_type

        if self.c is None or self.c is type(None):
            return f"{format_ico_type(self.i)} → {format_ico_type(self.o)}"
        return f"{format_ico_type(self.i)}, {format_ico_type(self.c)} → {format_ico_type(self.o)}"

    @property
    def name(self) -> str:
        return self.format()

    @property
    def has_input(self) -> bool:
        return self.i is not None and self.i is not type(None)

    @property
    def has_context(self) -> bool:
        return self.c is not None and self.c is not type(None)

    @property
    def has_output(self) -> bool:
        return self.o is not None and self.o is not type(None)
