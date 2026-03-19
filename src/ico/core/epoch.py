from collections.abc import Callable, Iterator
from typing import Any, Generic

from ico.core.context_operator import (
    C,
    I,
    IcoContextOperator,
    wrap_context_operator,
)
from ico.core.operator import IcoOperator, wrap_operator
from ico.core.signature import IcoSignature


class IcoEpoch(
    Generic[I, C],
    IcoOperator[C, C],
):
    """Epoch-based data processing that accumulates context over a data stream.

    IcoEpoch represents a complete processing cycle where all items from a source
    are consumed to progressively update a context object. This pattern is common
    in machine learning training epochs, data aggregation, and stateful processing.

    Generic Parameters:
        I: Input item type - the type of individual items consumed from the source.
        C: Context type - the type of state being accumulated and updated.

    ICO signature:
        C → C (via Iterator[I], C → C for each item)

    Example:
        >>> from collections.abc import Iterator
        >>> from dataclasses import dataclass
        >>> from typing import TypeAlias
        >>> from ico.core.source import source

        >>> # Type definitions for training data
        >>> TrainingBatch: TypeAlias = tuple[float, float]

        >>> @dataclass
        ... class LinearModel:
        ...     weight: float = 0.5
        ...     bias: float = 0.0
        ...     total_loss: float = 0.0
        ...     samples_processed: int = 0

        >>> # Linear regression training epoch example
        >>> @source()
        ... def get_training_data() -> Iterator[TrainingBatch]:
        ...     # Training samples: (x, y) pairs following y = 2*x + 1
        ...     return iter([(1.0, 3.0), (2.0, 5.0), (3.0, 7.0), (4.0, 9.0)])

        >>> def update_model(sample: TrainingBatch, model: LinearModel) -> LinearModel:
        ...     x, y = sample
        ...     # Forward pass: prediction = weight * x + bias
        ...     prediction = model.weight * x + model.bias
        ...     error = prediction - y
        ...
        ...     # Compute gradients
        ...     grad_weight = error * x
        ...     grad_bias = error
        ...
        ...     # Update parameters with learning rate
        ...     learning_rate = 0.01
        ...     model.weight -= learning_rate * grad_weight
        ...     model.bias -= learning_rate * grad_bias
        ...
        ...     # Accumulate loss for monitoring
        ...     model.total_loss += error ** 2
        ...     model.samples_processed += 1
        ...
        ...     return model

        >>> epoch = IcoEpoch(get_training_data, update_model, name="training_epoch")
        >>> model = LinearModel()
        >>> trained_model = epoch(model)
        >>> avg_loss = trained_model.total_loss / trained_model.samples_processed
        >>> print(f"Trained weight: {trained_model.weight:.3f}")
        >>> print(f"Average loss: {avg_loss:.3f}")

        >>> # Data aggregation example
        >>> @dataclass
        ... class AggregateStats:
        ...     sum: float = 0.0
        ...     count: int = 0
        ...     sum_squares: float = 0.0
        ...
        ...     def mean(self) -> float:
        ...         return self.sum / self.count if self.count > 0 else 0.0
        ...
        ...     def variance(self) -> float:
        ...         if self.count == 0:
        ...             return 0.0
        ...         mean_val = self.mean()
        ...         return (self.sum_squares / self.count) - mean_val ** 2

        >>> def get_numbers() -> Iterator[int]:
        ...     return iter([1, 2, 3, 4, 5])

        >>> def accumulate(num: int, stats: AggregateStats) -> AggregateStats:
        ...     stats.sum += num
        ...     stats.count += 1
        ...     stats.sum_squares += num ** 2
        ...     return stats

        >>> aggregator = IcoEpoch(get_numbers, accumulate)
        >>> result = aggregator(AggregateStats())
        >>> print(f"Mean: {result.mean():.1f}, Variance: {result.variance():.1f}")
        >>> assert result.sum == 15 and result.count == 5 and result.sum_squares == 55

    Attributes:
        source: Operator that provides the data stream for this epoch.
        context_operator: Operator that updates context with each data item.

    Note:
        The epoch processes ALL items from the source before returning,
        making it suitable for batch processing and complete data passes.
    """

    source: IcoOperator[None, Iterator[I]]
    context_operator: IcoContextOperator[I, C, C]

    def __init__(
        self,
        source: Callable[[None], Iterator[I]],
        context_operator: Callable[[I, C], C],
        *,
        name: str | None = None,
    ) -> None:
        """Initialize an epoch with a data source and context update operator.

        Args:
            source: Zero-argument callable that returns an Iterator[I] of data items.
                   Will be wrapped as an IcoOperator automatically.
            context_operator: Function that takes (item: I, context: C) and returns
                            updated context C. Will be wrapped as IcoContextOperator.
            name: Optional name for this epoch (useful for debugging/visualization).

        Note:
            Both callables are automatically wrapped in appropriate operator types
            and stored as children of this epoch node.
        """
        source_op = wrap_operator(source)
        context_op = wrap_context_operator(context_operator)

        super().__init__(
            fn=self._process_fn,
            name=name,
            children=[source_op, context_op],
        )
        self.source = source_op
        self.context_operator = context_op

    def _process_fn(self, context: C) -> C:
        """Internal implementation that processes the complete epoch.

        Args:
            context: Initial context state to start the epoch with.

        Returns:
            Updated context after processing all items from the source.

        Note:
            This is the function used by __call__. It iterates through ALL
            items from the source, applying the context operator to each one
            sequentially to build up the final context state.
        """
        for item in self.source(None):
            context = self.context_operator(item, context)
        return context

    @property
    def signature(self) -> IcoSignature:
        """Infer the ICO type signature for this epoch.

        Derives the epoch signature from the context operator's signature,
        since the epoch's input/output types match the context types.

        Returns:
            IcoSignature with context type C as both input and output,
            derived from the context operator's signature.

        Note:
            The signature reflects C → C transformation, even though
            internally the epoch processes Iterator[I] items to update C.
        """
        signature = super().signature

        if not signature.infered:
            signature = self.context_operator.signature

        # Help mypy to understand this is a type, not just a variable
        i_type: Any = signature.i
        c_type: Any = signature.c

        return IcoSignature(
            i=Iterator[i_type],
            c=c_type,
            o=c_type,
        )
