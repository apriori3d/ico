from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Generic, final

from ico.core.context_operator import (
    C,
    IcoContextOperator,
    wrap_context_operator,
)
from ico.core.operator import (
    I,
    IcoOperator,
    O,
    wrap_operator,
)
from ico.core.signature import IcoSignature


@final
class IcoContextPipeline(
    Generic[I, C, O],
    IcoContextOperator[I, C, O],
):
    """Context pipeline that applies a context operator followed by a sequence of operators.

    IcoContextPipeline combines a context-aware operation with a series of transformations.
    First, it applies the context operator to get an intermediate result, then passes this
    result through a pipeline of regular operators. This pattern is useful for initializing
    computation with context and then processing the result through transformations.

    Generic Parameters:
        I: Input item type - the type of data items processed with context.
        C: Context type - the type of context object used in initial operation.
        O: Output type - the final type after context operation and transformations.

    ICO signature:
        (I, C) → O (via context operation followed by transformations)

    Example:
        >>> # Context initialization + transformations
        >>> def apply_context(num: int, multiplier: int) -> int:
        ...     return num * multiplier

        >>> def add_five(x: int) -> int:
        ...     return x + 5

        >>> def double(x: int) -> int:
        ...     return x * 2

        >>> # Pipeline: apply context, then add_five, then double
        >>> pipeline = IcoContextPipeline(apply_context, add_five, double, name="math_pipeline")
        >>> result = pipeline(3, 4)  # (3 * 4) + 5) * 2 = (12 + 5) * 2 = 34
        >>> assert result == 34

        >>> # Single transformation example
        >>> def initialize(value: int, base: int) -> int:
        ...     return value + base

        >>> def square(x: int) -> int:
        ...     return x ** 2

        >>> simple_pipeline = IcoContextPipeline(initialize, square)
        >>> result = simple_pipeline(5, 3)  # (5 + 3) ** 2 = 64
        >>> assert result == 64
        >>> assert len(simple_pipeline) == 1  # One transformation step

    Attributes:
        apply: Context operator that processes (I, C) → O.
        body: Sequence of operators that transform the apply result.

    Note:
        The pipeline body must contain at least one operator. This ensures
        meaningful composition beyond just the context application.
    """

    __slots__ = ("apply", "body")

    apply: IcoContextOperator[I, C, O]
    body: Sequence[IcoOperator[O, O]]

    def __init__(
        self,
        apply: Callable[[I, C], O],
        *body: Callable[[O], O],
        name: str | None = None,
    ):
        """Initialize a context pipeline with context operator and transformation steps.

        Args:
            apply: Context operator function (I, C) → O that processes input with context.
                  Will be wrapped as IcoContextOperator automatically.
            *body: Variable number of transformation functions O → O applied sequentially.
                  Each will be wrapped as IcoOperator automatically.
            name: Optional name for this pipeline (useful for debugging/visualization).

        Raises:
            ValueError: If body is empty (at least one transformation is required).

        Note:
            All functions are automatically wrapped in appropriate operator types
            and stored as children of this pipeline node for tree structure.
        """
        if len(body) == 0:
            raise ValueError("Pipeline body must contain at least one operator.")

        apply_op = wrap_context_operator(apply)
        body_ops = [wrap_operator(step) for step in body]
        children = [apply_op] + body_ops

        super().__init__(
            fn=self._run_pipeline,
            name=name,
            children=children,
        )
        self.apply = apply_op
        self.body = body_ops

    def _run_pipeline(self, item: I, context: C) -> O:
        """Internal implementation that executes the complete pipeline.

        Args:
            item: Input item of type I to be processed.
            context: Context object of type C for the initial operation.

        Returns:
            Final result of type O after applying context operation and all transformations.

        Note:
            This is the function used by __call__. It first applies the context
            operator, then sequentially applies each transformation in the body.
        """
        output = self.apply(item, context)
        for step in self.body:
            output = step(output)
        return output

    def __len__(self) -> int:
        """Return the number of transformation steps in the pipeline body.

        Returns:
            Number of operators in the body (excludes the context apply operator).

        Note:
            This counts only the transformation steps, not the initial context
            application. Useful for pipeline introspection and debugging.
        """
        return len(self.body)

    @property
    def signature(self) -> IcoSignature:
        """Infer the ICO type signature for this context pipeline.

        Derives the pipeline signature from the context operator's signature,
        since the pipeline's input/output types match the context operator types.

        Returns:
            IcoSignature with types (I, C) → O derived from the context operator,
            or the inferred signature if available from the parent class.

        Note:
            The signature represents the overall pipeline transformation, abstracting
            away the intermediate steps in the body transformations.
        """
        signature = super().signature

        return signature if signature.infered else self.apply.signature
