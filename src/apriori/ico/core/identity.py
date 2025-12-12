from typing import Generic

from apriori.ico.core.operator import I, IcoOperator


class IcoIdentity(Generic[I], IcoOperator[I, I]):
    type_name = "Identity"

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(fn=self._identity_fn, name=name)

    def _identity_fn(self, item: I) -> I:
        return item
