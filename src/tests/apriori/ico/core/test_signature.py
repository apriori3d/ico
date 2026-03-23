from collections.abc import Iterator

from ico.core.batcher import IcoBatcher
from ico.core.context_operator import IcoContextOperator
from ico.core.context_pipeline import IcoContextPipeline
from ico.core.epoch import IcoEpoch
from ico.core.operator import IcoOperator
from ico.core.pipeline import IcoPipeline
from ico.core.process import IcoProcess
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.runtime.agent.mp.mp_agent import MPAgent

# ─── Operator ───


def test_infer_form_operator_generics() -> None:
    op = IcoOperator[int, float](lambda x: float(x))
    form = op.signature
    assert form.name == "int → float"


def test_infer_form_operator_fn() -> None:
    def fn(x: str) -> int:
        return len(x)

    op = IcoOperator(fn)
    assert op.signature.format() == "str → int"


def test_infer_form_operator_no_hints() -> None:
    op = IcoOperator(lambda x: x)  # type: ignore
    assert op.signature.format() == "Any → Any"  # type: ignore


def test_infer_form_operator_with_class_callable() -> None:
    class Adder:
        def __call__(self, x: int) -> int:
            return x + 1

    adder = Adder()
    op = IcoOperator(adder)
    assert op.signature.format() == "int → int"


# ─── Composition ───


def test_infer_form_compose() -> None:
    to_float = IcoOperator[int, float](float)
    to_str = IcoOperator[float, str](str)
    composed = to_float | to_str
    form = composed.signature

    assert form.name == "int → str"


# ─── Stream ───


def test_infer_form_stream() -> None:
    base = IcoOperator[int, float](lambda x: float(x))
    iterate = base.stream()
    streamed = IcoStream(base)

    assert iterate.signature.format() == "Iterator[int] → Iterator[float]"
    assert streamed.signature.format() == "Iterator[int] → Iterator[float]"

    # Test inference with hints
    stream = IcoStream[int, float](lambda x: float(x))

    assert stream.signature.format() == "Iterator[int] → Iterator[float]"


# ─── Process ───


def test_infer_form_process_from_generic() -> None:
    proc = IcoProcess[float](lambda c: c + 1, num_iterations=3)
    assert proc.signature.format() == "float → float"


def test_infer_form_process_from_body() -> None:
    def increment_fn(c: float) -> float:
        return c + 1

    body_op = IcoOperator(increment_fn)

    proc = IcoProcess(body_op, num_iterations=3)
    assert proc.signature.format() == "float → float"


def test_infer_form_process_from_fn() -> None:
    def increment_fn(c: float) -> float:
        return c + 1

    proc = IcoProcess(increment_fn, num_iterations=3)
    assert proc.signature.format() == "float → float"


def test_infer_from_process_fib() -> None:
    def fib_step(state: tuple[int, int]) -> tuple[int, int]:
        return (state[1], state[0] + state[1])

    fib_process = IcoProcess(fib_step, num_iterations=8)
    assert fib_process.signature.format() == "tuple[int, int] → tuple[int, int]"


# ─── Pipeline ───


def test_infer_form_pipeline_generics() -> None:
    p = IcoPipeline[int](
        IcoOperator[int, int](lambda x: x * 2),
        IcoOperator[int, int](lambda x: x + 1),
    )
    assert p.signature.format() == "int → int"


def test_infer_form_pipeline_without_hints() -> None:
    p = IcoPipeline[int](
        lambda x: x * 2,
        lambda x: x + 1,
    )
    assert p.signature.format() == "int → int"


def test_infer_form_pipeline_with_fn() -> None:
    def body_fn(x: int) -> int:
        return x * 2

    p = IcoPipeline(body_fn)
    assert p.signature.format() == "int → int"


# ─── Source ───


def test_infer_form_source_with_generics() -> None:
    src = IcoSource[float](lambda: iter([1.0, 2.0, 3.0]))
    assert src.signature.format() == "() → Iterator[float]"


def test_infer_form_source_with_fn() -> None:
    def source_fn() -> Iterator[float]:
        yield from [1.0, 2.0, 3.0]

    src = IcoSource(source_fn)
    assert src.signature.format() == "() → Iterator[float]"


# ─── Sink ───


def sink_fn(x: float) -> None:
    pass


def test_infer_form_sink_with_generics() -> None:
    sink = IcoSink[float](sink_fn)
    assert sink.signature.format() == "Iterator[float] → ()"


def test_infer_form_sink_with_fn() -> None:
    sink = IcoSink(sink_fn)
    assert sink.signature.format() == "Iterator[float] → ()"


# ─── Batcher ───


def test_infer_form_batcher() -> None:
    batcher = IcoBatcher[int](batch_size=2)
    assert batcher.signature.format() == "Iterator[int] → Iterator[Iterator[int]]"


# ─── Context operator ───


def test_infer_form_context_operator_from_generics() -> None:
    ctx_op = IcoContextOperator[int, str, str](lambda i, c: c + str(i))
    assert ctx_op.signature.format() == "int, str → str"


def test_infer_form_context_operator_from_hints() -> None:
    def ctx_fn(i: int, c: str) -> str:
        return c + str(i)

    ctx_op = IcoContextOperator(ctx_fn)
    assert ctx_op.signature.format() == "int, str → str"


# ─── Context Pipeline ───


def test_infer_form_context_pipeline_from_generics() -> None:
    ctx_pipe = IcoContextPipeline[int, str, str](
        IcoContextOperator[int, str, str](lambda i, c: c + str(i)),
        IcoOperator[str, str](lambda c: c.upper()),
    )
    assert ctx_pipe.signature.format() == "int, str → str"


def test_infer_form_context_pipeline_from_type_hints() -> None:
    def apply_fn(i: int, c: str) -> str:
        return c + str(i)

    def step_fn(c: str) -> str:
        return c.upper()

    ctx_pipe = IcoContextPipeline(
        apply_fn,
        step_fn,
    )
    assert ctx_pipe.signature.format() == "int, str → str"


# ─── Epoch ───


def test_infer_form_epoch_from_generics() -> None:
    source = IcoSource[int](lambda: iter([1, 2, 3]))
    context_op = IcoContextOperator[int, str, str](lambda i, c: c + str(i))

    epoch = IcoEpoch(source, context_op)
    assert epoch.signature.format() == "Iterator[int], str → str"


def test_infer_form_epoch_from_type_hints() -> None:
    def source_fn() -> Iterator[int]:
        yield from [1, 2, 3]

    def context_fn(i: int, c: str) -> str:
        return c + str(i)

    source = IcoSource(source_fn)
    context_op = IcoContextOperator(context_fn)

    epoch = IcoEpoch(source, context_op)
    assert epoch.signature.format() == "Iterator[int], str → str"


def test_infer_form_epoch_from_type_hints_fn_only() -> None:
    def source_fn(_: None) -> Iterator[int]:
        yield from [1, 2, 3]

    def context_fn(i: int, c: str) -> str:
        return c + str(i)

    epoch = IcoEpoch(source_fn, context_fn)
    assert epoch.signature.format() == "Iterator[int], str → str"


def test_infer_form_epoch_from_type_hints_with_source() -> None:
    def source_fn() -> Iterator[int]:
        yield from [1, 2, 3]

    def context_fn(i: int, c: str) -> str:
        return c + str(i)

    source = IcoSource(source_fn)
    epoch = IcoEpoch(source, context_fn)
    assert epoch.signature.format() == "Iterator[int], str → str"


# ─── MPAgent ───


def test_infer_form_mp_agent_from_generics() -> None:
    """Test MPAgent signature inference from explicit Generic parameters."""

    def create_worker_flow() -> IcoOperator[str, int]:
        return IcoOperator[str, int](lambda x: len(x))

    agent = MPAgent[str, int](flow_factory=create_worker_flow)
    signature = agent.signature

    # MPAgent should have signature: str → int (input → output)
    # Context (c) should be None for agents
    assert signature.format() == "str → int"


def test_infer_form_mp_agent_with_type_hints() -> None:
    """Test MPAgent signature inference from factory function type hints."""

    def computation(x: int) -> float:
        return float(x * 2)

    def create_worker_flow() -> IcoOperator[int, float]:
        return IcoOperator(computation)

    agent = MPAgent(flow_factory=create_worker_flow)
    signature = agent.signature
    assert signature.format() == "int → float"


def test_infer_form_mp_agent_complex_computation() -> None:
    """Test MPAgent signature with complex computation flow."""

    def string_processor(text: str) -> list[str]:
        return text.split()

    def create_complex_flow() -> IcoOperator[str, list[str]]:
        return IcoOperator(string_processor)

    agent = MPAgent[str, list[str]](flow_factory=create_complex_flow)
    signature = agent.signature

    # Test complex type handling
    assert signature.format() == "str → list[str]"  # Should be "str → list[str]?"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
