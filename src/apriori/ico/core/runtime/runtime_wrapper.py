from collections.abc import Callable, Sequence
from typing import Generic

from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.runtime.command import IcoRuntimeCommand
from apriori.ico.core.runtime.node import IcoRuntimeNode


class IcoRuntimeWrapper(
    Generic[I, O],
    IcoOperator[I, O],
    IcoRuntimeNode,
):
    operator: IcoOperator[I, O]

    def __init__(
        self,
        operator: IcoOperator[I, O],
        *,
        name: str | None = None,
        runtime_name: str | None = None,
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
    ) -> None:
        # Note: pylance cannot infer IcoOperator.__init__ from Generic inheritance, but mypy can.
        IcoOperator.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            fn=self._wrapper_fn,
            name=name,
            children=[operator],
        )

        IcoRuntimeNode.__init__(
            self,
            runtime_name=runtime_name,
            runtime_parent=runtime_parent,
            runtime_children=runtime_children,
        )
        self.operator = operator

    def _wrapper_fn(self, item: I) -> O:
        return self.operator(item)

    def on_command(self, command: IcoRuntimeCommand):
        return super().on_command(command)


# ─────────────────────────────────────────────
# Decorator
# ─────────────────────────────────────────────


def with_runtime(
    *nodes: IcoRuntimeNode,
) -> Callable[[IcoOperator[I, O]], IcoOperator[I, O]]:
    def decorator(operator: IcoOperator[I, O]) -> IcoOperator[I, O]:
        return with_runtime_fn(operator, *nodes)

    return decorator


def with_runtime_fn(
    operator: IcoOperator[I, O],
    *nodes: IcoRuntimeNode,
) -> IcoOperator[I, O]:
    unbind_nodes = [n for n in nodes if n.runtime_parent is None]
    if len(unbind_nodes) == 0:
        return operator

    if isinstance(operator, IcoRuntimeNode):
        # Attach only unbind nodes to runtime tree.
        operator.add_runtime_children(*unbind_nodes)
        return operator

    return IcoRuntimeWrapper[I, O](operator, runtime_children=unbind_nodes)
