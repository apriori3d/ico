from collections.abc import Callable, Iterator
from typing import ClassVar, Generic, final

from apriori.ico.core.operator import (
    IcoOperator,
    O,
)


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

    type_name: ClassVar[str] = "Source"

    iterator: Callable[[], Iterator[O]]

    def __init__(
        self,
        fn: Callable[[], Iterator[O]],
        *,
        name: str | None = None,
    ):
        super().__init__(
            fn=self._iterator_fn,
            name=name,
            original_fn=fn,
            children=[],
        )
        self.iterator = fn

    def _iterator_fn(self, _: None) -> Iterator[O]:
        yield from self.iterator()


# ─────────────────────────────────────────────
# Decorator
# ─────────────────────────────────────────────


def source() -> Callable[[Callable[[], Iterator[O]]], IcoSource[O]]:
    def decorator(fn: Callable[[], Iterator[O]]) -> IcoSource[O]:
        return IcoSource(fn)

    return decorator
