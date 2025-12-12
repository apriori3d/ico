from typing import Generic

from apriori.ico.core.identity import IcoIdentity
from apriori.ico.core.operator import I


class IcoMonitor(Generic[I], IcoIdentity[I]):
    type_name = "Monitor"

    def __init__(self, *, name: str | None = None) -> None:
        IcoIdentity.__init__(self, name=name)  # pyright: ignore[reportUnknownMemberType]

    def __call__(self, item: I) -> I:
        self._before_call(item)

        result = self.fn(item)

        self._after_call(item)
        return result

    def _before_call(self, item: I) -> None:
        pass

    def _after_call(self, item: I) -> None:
        pass
