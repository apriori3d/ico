from typing import Generic

from ico.core.identity import IcoIdentity
from ico.core.operator import IInv
from ico.core.runtime.node import IcoRuntimeNode


class IcoMonitor(
    Generic[IInv],
    IcoIdentity[IInv],
    IcoRuntimeNode,
):
    """An identity node of the flow that is also a node of the runtime tree,
    so it can report on the items that pass it (see ``IcoProgress``).

    Subclasses override ``_before_call`` and ``_after_call``, called around
    each item. The monitor's state follows the lifecycle only (Ready while
    active): items do not change it, as several threads may pass items at
    once.
    """

    def __init__(self, *, name: str | None = None) -> None:
        IcoIdentity.__init__(self, name=name)  # pyright: ignore[reportUnknownMemberType]
        IcoRuntimeNode.__init__(self, runtime_name=name)

    def __call__(self, item: IInv) -> IInv:
        self._before_call(item)
        result = super().__call__(item)
        self._after_call(item)
        return result

    def _before_call(self, item: IInv) -> None:  # noqa: B027
        """Called before ``item`` passes."""

    def _after_call(self, item: IInv) -> None:  # noqa: B027
        """Called after ``item`` passed."""
