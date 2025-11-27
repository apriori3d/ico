import pytest

from apriori.ico.core.meta.flow_meta import IcoFlowMeta, IcoNodeType
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.process import IcoProcess
from apriori.ico.core.source import IcoSource
from apriori.ico.core.stream import IcoStream

# ─── Operator ───


def test_icoflow_operator_node() -> None:
    op = IcoOperator[int, float](lambda x: float(x), name="to_float")
    flow = IcoFlowMeta.from_node(op)

    assert flow.node_type == IcoNodeType.operator
    assert flow.ico_form.name == "int → float"
    assert flow.name == "to_float"
    assert not flow.children


# ─── Compose ───
def test_icoflow_compose_node() -> None:
    a = IcoOperator[int, float](float, name="to_float")
    b = IcoOperator[float, str](str, name="to_str")
    composed = a | b

    flow = IcoFlowMeta.from_node(composed)
    assert flow.node_type == IcoNodeType.chain
    assert flow.ico_form.name == "int → str"
    assert [c.name for c in flow.children] == ["to_float", "to_str"]


# ─── Map ───
def test_icoflow_iterate_node() -> None:
    base = IcoOperator[int, float](lambda x: x * 0.5, name="scale")
    iterated = base.iterate()

    flow = IcoFlowMeta.from_node(iterated)
    assert flow.node_type == IcoNodeType.iterate
    assert flow.ico_form.name == "Iterator[int] → Iterator[float]"
    assert flow.children and flow.children[0].name == "scale"


# ─── Stream ───
def test_icoflow_stream_node() -> None:
    base = IcoOperator[int, float](lambda x: x * 2, name="scale")
    stream = IcoStream(base, name="stream")

    flow = IcoFlowMeta.from_node(stream)
    assert flow.node_type == IcoNodeType.stream
    assert flow.ico_form.name == "Iterator[int] → Iterator[float]"
    assert flow.children and flow.children[0].name == "scale"


# ─── Pipeline ───
def test_icoflow_pipeline_node() -> None:
    pipe = IcoPipeline[int, float, str](
        context=IcoOperator[int, float](float, name="to_float"),
        body=[IcoOperator[float, float](lambda x: x * 2, name="scale")],
        output=IcoOperator[float, str](str, name="to_str"),
    )
    flow = IcoFlowMeta.from_node(pipe)

    assert flow.node_type == IcoNodeType.pipeline
    assert flow.ico_form.name == "int → float → str"
    assert [c.name for c in flow.children] == ["to_float", "scale", "to_str"]


# ─── Process ───
def test_icoflow_process_node() -> None:
    increment_op = IcoOperator[int, int](lambda c: c + 1)
    process = IcoProcess[int](increment_op, num_iterations=3)
    flow = IcoFlowMeta.from_node(process)

    assert flow.node_type == IcoNodeType.process
    assert flow.ico_form.name == "int → int"


# ─── Source ───
def test_icoflow_source_node() -> None:
    src = IcoSource[int](lambda: iter([1, 2, 3]), name="data")
    flow = IcoFlowMeta.from_node(src)

    assert flow.ico_form.name == "() → Iterator[int]"
    assert flow.name == "data"
    assert not flow.children


# ─── Traversal ───
def test_icoflow_traverse_returns_all_nodes() -> None:
    src = IcoSource[int](lambda: iter([1, 2, 3]), name="src")
    op = IcoOperator[int, int](lambda x: x + 1, name="plus")
    pipe = IcoPipeline[int, int, int](
        context=IcoOperator[int, int](lambda x: x, name="ctx"),
        body=[op],
        output=IcoOperator[int, int](lambda x: x, name="out"),
        name="pipe",
    )
    stream = IcoStream(pipe, name="stream")
    flow = src | stream
    flow_desc = IcoFlowMeta.from_node(flow)
    names: list[str] = [n.name for n in flow_desc.traverse() if n.name]
    used_names: set[str] = {"src", "stream", "pipe", "ctx", "plus", "out"}
    names_set: set[str] = set(names)
    assert used_names.issubset(names_set)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
