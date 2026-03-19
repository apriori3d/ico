# In this module me infer ICO forms for possibly untyped callables,
# and have to disable categories of errors related to using Any type in inspect api.
# mypy: disable-error-code=misc
from __future__ import annotations

from dataclasses import dataclass
from types import GenericAlias

# ────────────────────────────────────────────────
# Signature descriptions
# ────────────────────────────────────────────────


SignatureParamType = type | GenericAlias


@dataclass(slots=True)
class IcoSignature:
    """
    A representation of the ICO type signature for operators in the DSL.

    Captures the input, context, and output types of an ICO operator to enable
    type inference, validation, and visualization. Supports both concrete types
    and typing generics.

    ICO signature format:
        I → O (simple transformation)
        I, C → O (transformation with context)

    Example:
        >>> # Simple int to string conversion
        >>> sig = IcoSignature(i=int, c=None, o=str)
        >>> print(sig.format())  # "int → str"

        >>> # Transformation with context
        >>> sig = IcoSignature(i=str, c=dict, o=int)
        >>> print(sig.format())  # "str, dict → int"

    Attributes:
        i: Input type of the operator (can be Any or typing generic).
        c: Optional context type for stateful operations.
        o: Output type of the operator (can be Any or typing generic).
        infered: Whether this signature was successfully inferred from code.
    """

    i: SignatureParamType
    c: SignatureParamType | None
    o: SignatureParamType
    infered: bool = True

    def format(self) -> str:
        """Format the signature as a human-readable type flow string.

        Returns:
            A string representation showing the type flow:
            - "I → O" for simple transformations
            - "I, C → O" for transformations with context

        Example:
            >>> sig = IcoSignature(i=int, c=str, o=float)
            >>> print(sig.format())  # "int, str → float"
        """
        from ico.core.signature_utils import format_ico_type

        if self.c is None:
            return f"{format_ico_type(self.i)} → {format_ico_type(self.o)}"

        return f"{format_ico_type(self.i)}, {format_ico_type(self.c)} → {format_ico_type(self.o)}"

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
    def has_context(self) -> bool:
        """Check if this signature has a defined context type.

        Returns:
            True if context type is specified and not None, False otherwise.

        Note:
            Context is optional - operators can work with or without it.
        """
        return self.c is not None and self.c is not type(None)

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
