from collections.abc import Iterator

from ico.core.batcher import IcoBatcher
from ico.core.fold import IcoFold
from ico.core.operator import IcoOperator
from ico.core.pipeline import IcoPipeline
from ico.core.process import IcoProcess
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.runtime.worker.mp.mp_worker import MPWorker

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


# ─── Fold ───


def test_infer_form_fold_from_generics() -> None:
    source = IcoSource[int](lambda: iter([1, 2, 3]))
    step = IcoOperator[tuple[int, str], str](lambda pair: pair[1] + str(pair[0]))

    fold = IcoFold(source, step)
    assert fold.signature.format() == "str → str"


def test_infer_form_fold_from_type_hints() -> None:
    def source_fn() -> Iterator[int]:
        yield from [1, 2, 3]

    def step_fn(pair: tuple[int, str]) -> str:
        return pair[1] + str(pair[0])

    fold = IcoFold(IcoSource(source_fn), step_fn)
    assert fold.signature.format() == "str → str"
    assert fold("") == "123"


def test_infer_form_fold_with_a_chain_as_its_step() -> None:
    def source_fn() -> Iterator[int]:
        yield from [1, 2, 3]

    def add(pair: tuple[int, str]) -> str:
        return pair[1] + str(pair[0])

    def shout(state: str) -> str:
        return state + "!"

    fold = IcoFold(IcoSource(source_fn), IcoOperator(add) | IcoOperator(shout))
    assert fold.signature.format() == "str → str"
    assert fold("") == "1!2!3!"


# ─── MPWorker ───


def test_infer_form_mp_worker_from_generics() -> None:
    """Test MPWorker signature inference from explicit Generic parameters."""

    def create_worker_flow() -> IcoOperator[str, int]:
        return IcoOperator[str, int](lambda x: len(x))

    worker = MPWorker[str, int](flow_factory=create_worker_flow)
    signature = worker.signature

    # MPWorker should have signature: str → int (input → output)
    # Context (c) should be None for workers
    assert signature.format() == "str → int"


def test_infer_form_mp_worker_with_type_hints() -> None:
    """Test MPWorker signature inference from factory function type hints."""

    def computation(x: int) -> float:
        return float(x * 2)

    def create_worker_flow() -> IcoOperator[int, float]:
        return IcoOperator(computation)

    worker = MPWorker(flow_factory=create_worker_flow)
    signature = worker.signature
    assert signature.format() == "int → float"


def test_infer_form_mp_worker_complex_computation() -> None:
    """Test MPWorker signature with complex computation flow."""

    def string_processor(text: str) -> list[str]:
        return text.split()

    def create_complex_flow() -> IcoOperator[str, list[str]]:
        return IcoOperator(string_processor)

    worker = MPWorker[str, list[str]](flow_factory=create_complex_flow)
    signature = worker.signature

    # Test complex type handling
    assert signature.format() == "str → list[str]"  # Should be "str → list[str]?"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))


Report = dict[str, str | int | list[int]]


class _Summarize(IcoOperator[Report, str]):
    def __init__(self) -> None:
        super().__init__(fn=self._apply)

    def _apply(self, report: Report) -> str:
        return str(report)


def test_class_operator_with_pep604_unions_in_its_base() -> None:
    signature = _Summarize().signature

    assert signature.infered
    assert signature.format() == "dict[str, str | int | list[int]] → str"
    flow = IcoOperator[str, Report](lambda s: {"title": s}) | _Summarize()
    assert flow("x") == "{'title': 'x'}"
