from typing import Generic

from apriori.ico.core.identity import IcoIdentity
from apriori.ico.core.operator import I
from apriori.ico.core.runtime.node import IcoRuntimeNode


class IcoMonitor(
    Generic[I],
    IcoIdentity[I],
    IcoRuntimeNode,
):
    def __init__(self, *, name: str | None = None) -> None:
        IcoIdentity.__init__(self, name=name)  # pyright: ignore[reportUnknownMemberType]
        IcoRuntimeNode.__init__(self, runtime_name=name)

    def __call__(self, item: I) -> I:
        self._before_call(item)

        result = super.__call__(item)

        self._after_call(item)
        return result

    def _before_call(self, item: I) -> None:
        self.state_model.running()

    def _after_call(self, item: I) -> None:
        self.state_model.ready()
