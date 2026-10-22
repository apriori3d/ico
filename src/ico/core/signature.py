# In this module me infer ICO forms for possibly untyped callables,
# and have to disable categories of errors related to using Any type in inspect api.
# mypy: disable-error-code=misc
from __future__ import annotations

from dataclasses import dataclass
from types import GenericAlias
from typing import Any, Union, get_origin

# ────────────────────────────────────────────────
# Signature descriptions
# ────────────────────────────────────────────────


SignatureParamType = type | GenericAlias


@dataclass(slots=True)
class IcoSignature:
    """
    A representation of the ICO type signature for operators in the DSL.

    Captures the input and output types of an ICO operator to enable type
    inference, validation, and visualization. Supports both concrete types
    and typing generics. State that an operator carries along goes in its
    input and output, e.g. ``tuple[Batch, Model] → Model`` (see ``IcoFold``).

    ICO signature format:
        I → O

    Example:
        >>> sig = IcoSignature(i=int, o=str)
        >>> print(sig.format())  # "int → str"

    Attributes:
        i: Input type of the operator (can be Any or typing generic).
        o: Output type of the operator (can be Any or typing generic).
        infered: Whether this signature was successfully inferred from code.
    """

    i: Any
    o: Any
    infered: bool = True

    def __post_init__(self) -> None:
        """Validate that the types are supported for ICO signatures."""
        if not is_supported_type(self.i):
            raise TypeError(f"Unsupported input type for ICO signature: {self.i}")
        if not is_supported_type(self.o):
            raise TypeError(f"Unsupported output type for ICO signature: {self.o}")

    def format(self) -> str:
        """Format the signature as a human-readable type flow string.

        Returns:
            A string representation showing the type flow, "I → O".

        Example:
            >>> sig = IcoSignature(i=int, o=float)
            >>> print(sig.format())  # "int → float"
        """
        from ico.core.signature_utils import format_ico_type

        return f"{format_ico_type(self.i)} → {format_ico_type(self.o)}"

    @property
    def name(self) -> str:
        """Get the formatted name of this signature.

        Returns:
            The same string as format() - a human-readable type flow.

        Note:
            This property provides convenient access to the formatted signature.
        """
        return self.format()

    @property
    def has_input(self) -> bool:
        """Check if this signature has a defined input type.

        Returns:
            True if input type is specified and not None, False otherwise.

        Note:
            Used for validation and signature analysis.
        """
        return self.i is not type(None)

    @property
    def has_output(self) -> bool:
        """Check if this signature has a defined output type.

        Returns:
            True if output type is specified and not None, False otherwise.

        Note:
            Used for validation and ensuring proper type flow in compositions.
        """
        return self.o is not type(None)

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return self.format()


def is_supported_type(t: Any) -> bool:
    """Check if a type is supported for ICO signature inference."""
    origin: object = get_origin(t)
    return (
        isinstance(t, SignatureParamType) or isinstance(origin, type) or origin is Union
    )
