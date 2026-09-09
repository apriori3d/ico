from typing import Any, cast

from ico.core.chain import IcoChain
from ico.core.context_operator import IcoContextOperator
from ico.core.context_pipeline import IcoContextPipeline
from ico.core.node import IcoNodeProtocol, create_flow_walker
from ico.core.operator import IcoOperator, IcoOperatorProtocol, operator
from ico.core.pipeline import IcoPipeline
from ico.core.runtime.runtime import IcoRuntime
from ico.runtime.agent.mp.mp_agent import MPAgent
from ico.tools.profiler.injector import inject_profiler
from ico.tools.profiler.nodes import IcoContextOperatorProfiler, IcoOperatorProfiler


def assert_all_profiler_injected(op: IcoNodeProtocol) -> None:
    walker = create_flow_walker(expand_remote_flows=True)
    node_infos = list(walker.traverse(op))

    i = 0
    while i < len(node_infos):
        node_info = node_infos[i]
        node: object = node_info.node

        # skip structural nodes like chains and pipelines, as they are not directly profiled
        if isinstance(node, IcoChain | IcoPipeline | IcoContextPipeline):
            i += 1
            continue

        if isinstance(node, IcoOperator):
            op = cast(IcoOperator[Any, Any], node)
            assert isinstance(
                op, IcoOperatorProfiler
            ), f"Node {op} is not a profiler-injected operator."
            i += 2  # skip the next node since it is the original operator wrapped by the profiler
            continue

        if isinstance(node, IcoContextOperator):
            op = cast(IcoContextOperator[Any, Any, Any], node)
            assert isinstance(
                op, IcoContextOperatorProfiler
            ), f"Node {op} is not a profiler-injected context operator."
            i += 2  # skip the next node since it is the original context operator wrapped by the profiler
            continue

        raise ValueError(
            f"Node {node} is not an IcoOperator and cannot be checked for profiler injection."
        )


def test_chain_injection() -> None:
    op_a = IcoOperator[str, str](lambda s: s + "a")
    op_b = IcoOperator[str, str](lambda s: s + "b")
    flow = op_a | op_b

    result = flow("")
    assert result == "ab"

    flow.name = "OriginalFlow"
    flow.describe()

    profiled_flow = inject_profiler(flow)
    profiled_flow.name = "Profiled Flow"
    profiled_flow.describe()

    assert_all_profiler_injected(profiled_flow)

    profiled_result = profiled_flow("")
    assert profiled_result == "ab"


def _flow_factory() -> IcoOperatorProtocol[str, str]:
    op_a = IcoOperator[str, str](lambda s: s + "a")
    op_b = IcoOperator[str, str](lambda s: s + "b")
    return op_a | op_b


def test_mp_agent_factory_injection() -> None:
    @operator()
    def _test_data_provider(_: None) -> str:
        return "test"

    @operator()
    def _check_result(result: str) -> None:
        assert result == "testab"

    flow = _test_data_provider | MPAgent(_flow_factory) | _check_result
    flow.name = "Original flow"
    flow.describe()

    runtime = IcoRuntime(flow)
    runtime.activate().run().deactivate()

    # a flow can only have single runtime in a lifecycle, so we need to recreate it for the profiled version
    flow = _test_data_provider | MPAgent(_flow_factory) | _check_result
    profiled_flow = inject_profiler(flow)
    profiled_flow.name = "Profiled flow"
    profiled_flow.describe()

    assert_all_profiler_injected(profiled_flow)

    runtime = IcoRuntime(profiled_flow)
    runtime.activate().run().deactivate()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
