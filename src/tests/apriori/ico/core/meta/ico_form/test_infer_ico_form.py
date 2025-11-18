from collections.abc import Iterator

from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.dsl.pipeline import IcoPipeline
from apriori.ico.core.dsl.process import IcoProcess
from apriori.ico.core.dsl.sink import IcoSink
from apriori.ico.core.dsl.source import IcoSource
from apriori.ico.core.dsl.stream import IcoStream
from apriori.ico.core.meta.ico_form import IcoForm, infer_ico_form

# ─── Operator ───


def test_infer_form_operator_generics() -> None:
    op = IcoOperator[int, float](lambda x: float(x))
    form = infer_ico_form(op)
    assert isinstance(form, IcoForm)
    assert form.name == "int → float"


def test_infer_form_operator_fn() -> None:
    def fn(x: str) -> int:
        return len(x)

    op = IcoOperator(fn)
    assert infer_ico_form(op).name == "str → int"


def test_infer_form_operator_no_hints() -> None:
    op = IcoOperator(lambda x: x)  # pyright: ignore[reportUnknownVariableType]
    assert infer_ico_form(op).name == "Any → Any"  # pyright: ignore[reportUnknownArgumentType]


# ─── Composition ───


def test_infer_form_compose() -> None:
    to_float = IcoOperator[int, float](float)
    to_str = IcoOperator[float, str](str)
    composed = to_float | to_str
    form = infer_ico_form(composed)

    assert form.name == "int → str"


# ─── Map / Stream ───


def test_infer_form_map_and_stream() -> None:
    base = IcoOperator[int, float](lambda x: float(x))
    mapped = base.map()
    streamed = IcoStream(base)

    assert infer_ico_form(mapped).name == "Iterator[int] → Iterator[float]"
    assert infer_ico_form(streamed).name == "Iterator[int] → Iterator[float]"


# ─── Process ───


def test_infer_form_process_from_generic() -> None:
    proc = IcoProcess[float](lambda c: c + 1, num_iterations=3)
    form = infer_ico_form(proc)
    assert form.name == "float → float"


def test_infer_form_process_from_body() -> None:
    def source_fn(c: float) -> float:
        return c + 1

    body_op = IcoOperator(source_fn)

    proc = IcoProcess(body_op, num_iterations=3)
    form = infer_ico_form(proc)
    assert form.name == "float → float"


def test_infer_form_process_from_fn() -> None:
    def increment_fn(c: float) -> float:
        return c + 1

    proc = IcoProcess(increment_fn, num_iterations=3)
    form = infer_ico_form(proc)
    assert form.name == "float → float"


# ─── Pipeline ───


def test_infer_form_pipeline_generics() -> None:
    p = IcoPipeline[int, float, str](
        context=IcoOperator[int, float](float),
        body=[IcoOperator[float, float](lambda x: x * 2)],
        output=IcoOperator[float, str](str),
    )
    assert infer_ico_form(p).name == "int → float → str"


def test_infer_form_pipeline_without_hints() -> None:
    p = IcoPipeline[int, float, str](
        context=lambda x: float(x),
        body=[lambda x: x * 2],
        output=lambda x: str(x),
    )
    assert infer_ico_form(p).name == "int → float → str"


def test_infer_form_pipeline_with_fn() -> None:
    def context_fn(x: int) -> float:
        return float(x)

    def body_fn(x: float) -> float:
        return x * 2

    def output_fn(x: float) -> str:
        return str(x)

    p = IcoPipeline(
        context=context_fn,
        body=[body_fn],
        output=output_fn,
    )
    assert infer_ico_form(p).name == "int → float → str"


# ─── Source ───


def test_infer_form_source_with_generics() -> None:
    src = IcoSource[float](lambda _: iter([1.0, 2.0, 3.0]))
    form = infer_ico_form(src)
    assert form.name == "() → Iterator[float]"


def test_infer_form_source_with_fn() -> None:
    def source_fn(_: None) -> Iterator[float]:
        yield from [1.0, 2.0, 3.0]

    src = IcoSource(source_fn)
    form = infer_ico_form(src)
    assert form.name == "() → Iterator[float]"


# ─── Sink ───


def sink_fn(x: Iterator[float]) -> None:
    for _ in x:
        pass


def test_infer_form_sink_with_generics() -> None:
    sink = IcoSink[float](sink_fn)
    form = infer_ico_form(sink)
    assert form.name == "Iterator[float] → ()"


def test_infer_form_sink_with_fn() -> None:
    sink = IcoSink(sink_fn)
    form = infer_ico_form(sink)
    assert form.name == "Iterator[float] → ()"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
