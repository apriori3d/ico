from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Generic, final

from apriori.ico.core.operator import (
    I,
    IcoOperator,
    wrap_operator,
)
from apriori.ico.core.signature import IcoSignature


@final
class IcoPipeline(
    Generic[I],
    IcoOperator[I, I],
):
    """Sequential pipeline of operators that transform the same type.

    IcoPipeline chains multiple operators of type I → I into a single composite
    operator. Each step in the pipeline receives the output of the previous step,
    enabling complex transformations through simple composition.

    Generic Parameters:
        I: Type being transformed - both input and output type for all pipeline steps.

    ICO signature:
        I → I (through multiple I → I steps)

    Example:
        >>> validate = IcoOperator(lambda x: x if x > 0 else 0)
        >>> double = IcoOperator(lambda x: x * 2)
        >>> clamp = IcoOperator(lambda x: min(x, 100))

        >>> pipeline = IcoPipeline(validate, double, clamp)
        >>> result = pipeline(42)  # 42 → 42 → 84 → 84
        >>> assert result == 84

    Note:
        All operators in the pipeline must have the same input/output type.
    """

    __slots__ = "body"

    body: Sequence[IcoOperator[I, I]]

    def __init__(
        self,
        *body: Callable[[I], I],
        name: str | None = None,
    ):
        """Initialize a pipeline with a sequence of operators.

        Args:
            *body: Variable number of callables, each transforming I → I.
                  Will be wrapped as IcoOperators automatically.
            name: Optional name for this pipeline (defaults to "streamline").

        Raises:
            ValueError: If no operators are provided (empty pipeline).

        Note:
            Each callable in body will be wrapped as an IcoOperator and
            stored as children of this pipeline node.
        """
        if len(body) == 0:
            raise ValueError("Pipeline body must contain at least one operator.")

        body_ops = [wrap_operator(step) for step in body]

        super().__init__(
            fn=self._run_streamline,
            name=name or "streamline",
            children=body_ops,
        )
        self.body = body_ops

    def _run_streamline(self, item: I) -> I:
        """Execute the pipeline by applying each operator sequentially.

        Args:
            item: Input value of type I.

        Returns:
            The result after applying all operators in sequence.

        Note:
            This is the internal implementation used by __call__.
        """
        item = item
        for step in self.body:
            item = step(item)
        return item

    def __len__(self) -> int:
        """Return the number of operators in this pipeline.

        Returns:
            The count of operators in the pipeline body.
        """
        return len(self.body)

    @property
    def signature(self) -> IcoSignature:
        """Infer the ICO type signature of this pipeline.

        Attempts to use the parent class signature inference first.
        If that fails, falls back to using the signature of the first
        operator in the pipeline.

        Returns:
            IcoSignature representing the type flow of this pipeline.

        Note:
            Since all operators have type I → I, the pipeline signature
            is also I → I, derived from the first operator if needed.
        """
        signature = super().signature

        return signature if signature.infered else self.body[0].signature
