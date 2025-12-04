from collections.abc import Sequence
from typing import Generic

from apriori.ico.core.operator import I, IcoOperator, O
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
        runtime_name: str | None = None,
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
    ) -> None:
        name = f"Runtime({operator.name})"

        # Note: pylance cannot infer IcoOperator.__init__ from Generic inheritance, but mypy can.
        IcoOperator.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            fn=self._wrapped_fn,
            name=name,
            ico_form_target=operator,
            children=[operator],
        )

        IcoRuntimeNode.__init__(
            self,
            runtime_name=runtime_name or name,
            runtime_parent=runtime_parent,
            runtime_children=runtime_children,
        )
        self.operator = operator

    def _wrapped_fn(self, item: I) -> O:
        return self.operator(item)
