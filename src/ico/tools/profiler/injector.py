from collections.abc import Callable
from typing import Any, cast, overload

from ico.core.chain import IcoChain
from ico.core.context_operator import IcoContextOperatorProtocol
from ico.core.context_pipeline import IcoContextPipeline
from ico.core.node import HasRemoteFlow, IcoNodeProtocol
from ico.core.operator import IcoOperatorProtocol
from ico.core.pipeline import IcoPipeline
from ico.core.stream import IcoStream
from ico.tools.profiler.nodes import (
    IcoContextOperatorProfiler,
    IcoOperatorProfiler,
    IcoStreamProfiler,
)


class IcoProfiledRemoteFlowFactory:
    original_factory: Callable[[], IcoNodeProtocol]

    def __init__(self, operator: HasRemoteFlow) -> None:
        self.original_factory = operator.get_remote_flow_factory()

    def __call__(self) -> IcoOperatorProtocol[Any, Any]:
        original_flow = self.original_factory()

        if isinstance(original_flow, IcoOperatorProtocol):
            op = cast(IcoOperatorProtocol[Any, Any], original_flow)
            return inject_profiler(op)

        raise ValueError("The remote flow is not an IcoOperator or IcoContextOperator.")


@overload
def inject_profiler(
    op: IcoOperatorProtocol[Any, Any],
) -> IcoOperatorProtocol[Any, Any]: ...


@overload
def inject_profiler(
    op: IcoContextOperatorProtocol[Any, Any, Any],
) -> IcoContextOperatorProtocol[Any, Any, Any]: ...


def inject_profiler(
    op: IcoOperatorProtocol[Any, Any] | IcoContextOperatorProtocol[Any, Any, Any],
) -> IcoOperatorProtocol[Any, Any] | IcoContextOperatorProtocol[Any, Any, Any]:
    if isinstance(op, IcoChain):
        chain = cast(IcoChain[Any, Any, Any], op)
        left_profiler = inject_profiler(chain.left)
        right_profiler = inject_profiler(chain.right)
        return IcoChain[Any, Any, Any](left_profiler, right_profiler)

    if isinstance(op, IcoPipeline):
        profiled_children = [inject_profiler(child) for child in op.body]
        return IcoPipeline[Any](*profiled_children)

    if isinstance(op, IcoContextPipeline):
        apply = inject_profiler(op.apply)
        body = [inject_profiler(child) for child in op.body]
        return IcoContextPipeline[Any, Any, Any](apply, *body)

    if isinstance(op, IcoStream):
        return IcoStreamProfiler(cast(IcoStream[Any, Any], op), name=f"Profiler({op})")

    if isinstance(op, HasRemoteFlow):
        op.set_remote_flow_factory(IcoProfiledRemoteFlowFactory(op))

    if isinstance(op, IcoOperatorProtocol):
        return IcoOperatorProfiler(op, name=f"Profiler({op})")

    if isinstance(op, IcoContextOperatorProtocol):  # pyright: ignore[reportUnnecessaryIsInstance]
        return IcoContextOperatorProfiler(op, name=f"Profiler({op})")

    raise ValueError("The provided operator is not a chain and cannot be injected.")
