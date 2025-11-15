import pytest

from apriori.ico.core import (
    IcoExecutionMixin,
    IcoExecutionState,
    IcoFlowMeta,
    IcoLifecycleMixin,
    IcoLifecycleState,
    IcoOperator,
    IcoPipeline,
    IcoProcess,
    IcoSource,
    IcoStream,
    NodeType,
)


# ─── Operator ───
def test_icoflow_operator_node() -> None:
    op = IcoOperator[int, float](lambda x: float(x), name="to_float")
    flow = IcoFlowMeta.from_operator(op)

    assert flow.node_type == NodeType.operator
    assert flow.ico_form.name == "int → float"
    assert flow.name == "to_float"
    assert not flow.children


# ─── Compose ───
def test_icoflow_compose_node() -> None:
    a = IcoOperator[int, float](float, name="to_float")
    b = IcoOperator[float, str](str, name="to_str")
    composed = a | b

    flow = IcoFlowMeta.from_operator(composed)
    assert flow.node_type == NodeType.chain
    assert flow.ico_form.name == "int → str"
    assert [c.name for c in flow.children] == ["to_float", "to_str"]


# ─── Map ───
def test_icoflow_map_node() -> None:
    base = IcoOperator[int, float](lambda x: x * 0.5, name="scale")
    mapped = base.map()

    flow = IcoFlowMeta.from_operator(mapped)
    assert flow.node_type == NodeType.map
    assert flow.ico_form.name == "Iterable[int] → Iterable[float]"
    assert flow.children and flow.children[0].name == "scale"


# ─── Stream ───
def test_icoflow_stream_node() -> None:
    base = IcoOperator[int, float](lambda x: x * 2, name="scale")
    stream = IcoStream(base, name="stream")

    flow = IcoFlowMeta.from_operator(stream)
    assert flow.node_type == NodeType.stream
    assert flow.ico_form.name == "Iterable[int] → Iterable[float]"
    assert flow.children and flow.children[0].name == "scale"


# ─── Pipeline ───
def test_icoflow_pipeline_node() -> None:
    pipe = IcoPipeline[int, float, str](
        context=IcoOperator[int, float](float, name="to_float"),
        body=[IcoOperator[float, float](lambda x: x * 2, name="scale")],
        output=IcoOperator[float, str](str, name="to_str"),
    )
    flow = IcoFlowMeta.from_operator(pipe)

    assert flow.node_type == NodeType.pipeline
    assert flow.ico_form.name == "int → float → str"
    assert [c.name for c in flow.children] == ["to_float", "scale", "to_str"]


# ─── Process ───
def test_icoflow_process_node() -> None:
    process = IcoProcess[int](lambda c: c + 1, num_iterations=3)
    flow = IcoFlowMeta.from_operator(process)

    assert flow.node_type == NodeType.process
    assert flow.ico_form.name == "int → int"
    assert flow.children and flow.children[0].node_type == NodeType.operator


# ─── Source ───
def test_icoflow_source_node() -> None:
    src = IcoSource[int](lambda: [1, 2, 3], name="data")
    flow = IcoFlowMeta.from_operator(src)

    assert flow.node_type == NodeType.source
    assert flow.ico_form.name == "() → Iterable[int]"
    assert flow.name == "data"
    assert not flow.children


# ─── Lifecycle + Execution ───
def test_icoflow_with_state_tracking() -> None:
    class Stateful(
        IcoOperator[int, int], IcoLifecycleMixin, IcoExecutionMixin[int, int]
    ):
        def __init__(self) -> None:
            IcoOperator.__init__(self, lambda x: x)
            IcoLifecycleMixin.__init__(self)
            IcoExecutionMixin.__init__(self)

        def set_states(self) -> None:
            self._state = IcoLifecycleState.prepared
            self._exec_state = IcoExecutionState.running

    op = Stateful()
    op.set_states()

    flow = IcoFlowMeta.from_operator(op)
    assert flow.state == IcoLifecycleState.prepared
    assert flow.exec_state == IcoExecutionState.running


# ─── Traversal ───
def test_icoflow_traverse_returns_all_nodes() -> None:
    src = IcoSource[int](lambda: [1, 2, 3], name="src")
    op = IcoOperator[int, int](lambda x: x + 1, name="plus")
    pipe = IcoPipeline[int, int, int](
        context=IcoOperator[int, int](lambda x: x, name="ctx"),
        body=[op],
        output=IcoOperator[int, int](lambda x: x, name="out"),
        name="pipe",
    )
    stream = IcoStream(pipe, name="stream")
    flow = src | stream
    flow_desc = IcoFlowMeta.from_operator(flow)
    names = [n.name for n in flow_desc.traverse() if n.name]
    assert {"src", "stream", "pipe", "ctx", "plus", "out"}.issubset(set(names))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
