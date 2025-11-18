from collections.abc import Callable, Iterator
from typing import Generic, final

from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.types import IcoNodeType, O


@final
class IcoSource(
    Generic[O],
    IcoOperator[None, Iterator[O]],
):
    """
    A data source operator, produces a data generator without requiring any input.
    ICO Form:
        () → Iterator[O]

    Example:
        >>> dataset = IcoSource(lambda: (1.0, 2.0, 3.0), name="dataset")
        >>> scale = IcoOperator(lambda x: x * 2, name="scale")
        >>> to_sum = IcoOperator(sum, name="sum")

        >>> flow = dataset | IcoStream(body=scale) | to_sum
        >>> result = flow()
        >>> print(result)
        12.0
    """

    def __init__(
        self, generator: Callable[[None], Iterator[O]], name: str | None = None
    ):
        super().__init__(
            fn=generator,
            name=name,
            node_type=IcoNodeType.source,
            children=[],
        )

    def __call__(self, item: None = None) -> Iterator[O]:
        yield from self.fn(item)
