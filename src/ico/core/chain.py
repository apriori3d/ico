from __future__ import annotations

from typing import Any, Generic, TypeVar, cast

from ico.core.node import IcoNode
from ico.core.operator import O2, I, IcoOperator, IcoOperatorProtocol, O
from ico.core.signature import IcoSignature

O3 = TypeVar("O3")


class IcoChain(Generic[I, O, O2], IcoOperator[I, O2]):
    """Sequential composition of two ICO operators: (I → O) → (O → O2) = (I → O2).

    IcoChain represents the fundamental composition pattern in ICO framework where
    the output of one operator becomes the input of the next operator. This enables
    building complex data processing pipelines from simple, reusable components.

    Generic Parameters:
        I: Input type - the type accepted by the first operator.
        O: Intermediate type - the output of first operator, input of second operator.
        O2: Output type - the final type produced by the second operator.

    ICO signature:
        I → O2 (composed from I → O and O → O2)

    Example:
        >>> # Simple number processing chain
        >>> def double(x: int) -> int:
        ...     return x * 2

        >>> def add_ten(x: int) -> int:
        ...     return x + 10

        >>> # Chain operators using pipe syntax
        >>> doubler = IcoOperator(double, name="double")
        >>> adder = IcoOperator(add_ten, name="add_ten")
        >>> transform = doubler | adder  # Same as IcoChain(doubler, adder)

        >>> result = transform(5)  # (5 * 2) + 10 = 20
        >>> assert result == 20

        >>> # Multi-step calculation chain
        >>> def square(x: int) -> int:
        ...     return x ** 2

        >>> def to_string(x: int) -> str:
        ...     return f"Result: {x}"

        >>> squarer = IcoOperator(square, name="square")
        >>> formatter = IcoOperator(to_string, name="format")
        >>> calculator = doubler | squarer | formatter

        >>> output = calculator(3)  # (3 * 2) ** 2 = 36 -> "Result: 36"
        >>> assert output == "Result: 36"

    Attributes:
        _left: First operator in the chain (I → O).
        _right: Second operator in the chain (O → O2).

    Note:
        Chain composition is associative: (A | B) | C ≡ A | (B | C).
        The pipe operator | is the recommended syntax for chaining operators.
    """

    _left: IcoOperatorProtocol[I, O]
    _right: IcoOperatorProtocol[O, O2]

    def __init__(
        self,
        left: IcoOperatorProtocol[I, O],
        right: IcoOperatorProtocol[O, O2],
        strict: bool | None = None,
    ):
        """Initialize a chain from two compatible operators.

        Args:
            left: First operator in the chain (I → O). Its output type O must
                 match the input type of the right operator.
            right: Second operator in the chain (O → O2). Its input type O must
                  match the output type of the left operator.
            strict: Check at runtime that left's output type is assignable to
                  right's input type. ``None`` follows the current
                  ``strict_signatures()`` context (off by default).

        Raises:
            IcoSignatureError: In strict mode, if the operator types are
                incompatible. Links with uninferred signatures are skipped.

        Note:
            The chain automatically creates a composite name "left | right"
            and establishes parent-child relationships for proper tree structure.
        """
        from ico.core.signature_check import (
            IcoSignatureError,
            chain_length,
            check_link,
            is_strict,
        )
        from ico.core.tree_utils import TreePathIndex

        if strict if strict is not None else is_strict():
            left_node, right_node = cast(IcoNode, left), cast(IcoNode, right)
            # Relative to this chain: the final flow tree is not known yet.
            step = TreePathIndex.create(chain_length(left_node))
            issue = check_link(left_node, right_node, path=step, chain_relative=True)
            if issue is not None and issue.is_error:
                raise IcoSignatureError(issue)

        super().__init__(
            fn=self._chained_fn,
            name=f"{left} | {right}",
            parent=None,
            children=[cast(IcoNode, left), cast(IcoNode, right)],
        )
        self._left = left
        self._right = right

    def __or__(self, other: IcoOperatorProtocol[O2, O3]) -> IcoOperatorProtocol[I, O3]:
        # mypy cannot always prove that recursive generic protocol conformance
        # holds for IcoChain here, although it is valid at runtime.
        return cast(IcoOperatorProtocol[I, O3], chain(self, other))

    def _chained_fn(self, item: I) -> O2:
        """Internal implementation that executes the chained operators.

        Args:
            item: Input of type I to be processed by the chain.

        Returns:
            Result of type O2 after applying both operators sequentially.

        Note:
            This is the function used by __call__. It first applies the left
            operator to get intermediate result, then applies right operator.
        """
        return self._right(self._left(item))

    @property
    def signature(self) -> IcoSignature:
        """Infer the ICO type signature for this chain.

        Combines signatures from both operators to create the composite signature.
        The input type comes from the left operator, output type from the right.

        Returns:
            IcoSignature with input type I from left operator and output type O2
            from right operator. Context is None since chains don't use context.

        Note:
            If either operator has an uninfered signature, the chain signature
            will also be uninfered with Any types for safety.
        """
        left = self._left.signature
        right = self._right.signature

        if left.infered and right.infered:
            return IcoSignature(
                i=left.i,
                c=None,
                o=right.o,
            )
        # Fallback to Any types
        return IcoSignature(
            i=type[Any],
            c=None,
            o=type[Any],
            infered=False,
        )


def chain(
    left: IcoOperatorProtocol[I, O],
    right: IcoOperatorProtocol[O, O2],
    strict: bool | None = None,
) -> IcoChain[I, O, O2]:
    """Create a chain composition of two ICO operators.

    Convenience function for creating IcoChain instances. This function provides
    a more functional approach to operator composition as an alternative to the
    class constructor or pipe operator syntax.

    Args:
        left: First operator in the chain (I → O).
        right: Second operator in the chain (O → O2).
        strict: Check signature compatibility at runtime; ``None`` follows
            the current ``strict_signatures()`` context.

    Returns:
        New IcoChain operator that represents the composition I → O2.

    Example:
        >>> # These three approaches are equivalent:
        >>> chain1 = IcoChain(op1, op2)           # Class constructor
        >>> chain2 = chain(op1, op2)              # Function call
        >>> chain3 = op1 | op2                    # Pipe operator (recommended)

        >>> # Simple usage example
        >>> def multiply_by_3(x: int) -> int:
        ...     return x * 3
        >>> def subtract_5(x: int) -> int:
        ...     return x - 5
        >>>
        >>> mul_op = IcoOperator(multiply_by_3)
        >>> sub_op = IcoOperator(subtract_5)
        >>> pipeline = chain(mul_op, sub_op)  # or: mul_op | sub_op
        >>>
        >>> result = pipeline(4)  # (4 * 3) - 5 = 7
        >>> assert result == 7

    Note:
        The output type O of left operator must match input type O of right
        operator for type safety. The pipe operator | is generally preferred
        over this function for better readability: op1 | op2.
    """
    return IcoChain(left, right, strict=strict)
