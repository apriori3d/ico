from collections.abc import Iterator

from apriori.ico.core.context_operator import IcoContextOperator
from apriori.ico.core.context_pipeline import IcoContextPipeline
from apriori.ico.core.context_stream import IcoEpoch
from apriori.ico.core.meta.ico_form import IcoForm, infer_ico_form
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.process import IcoProcess
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.source import IcoSource
from apriori.ico.core.stream import IcoStream
from apriori.ico.utils.data.batcher import IcoBatcher

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
    op = IcoOperator(lambda x: x)  # type: ignore
    assert infer_ico_form(op).name == "Any → Any"  # type: ignore


def test_infer_form_operator_with_class_callable() -> None:
    class Adder:
        def __call__(self, x: int) -> int:
            return x + 1

    adder = Adder()
    op = IcoOperator(adder)
    assert infer_ico_form(op).name == "int → int"


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
    iterate = base.iterate()
    streamed = IcoStream(base)

    assert infer_ico_form(iterate).name == "Iterator[int] → Iterator[float]"
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
    p = IcoPipeline[int](
        IcoOperator[int, int](lambda x: x * 2),
        IcoOperator[int, int](lambda x: x + 1),
    )
    assert infer_ico_form(p).name == "int → int"


def test_infer_form_pipeline_without_hints() -> None:
    p = IcoPipeline[int](
        lambda x: x * 2,
        lambda x: x + 1,
    )
    assert infer_ico_form(p).name == "int → int"


def test_infer_form_pipeline_with_fn() -> None:
    def body_fn(x: int) -> int:
        return x * 2

    p = IcoPipeline(body_fn)
    assert infer_ico_form(p).name == "int → int"


# ─── Source ───


def test_infer_form_source_with_generics() -> None:
    src = IcoSource[float](lambda: iter([1.0, 2.0, 3.0]))
    form = infer_ico_form(src)
    assert form.name == "() → Iterator[float]"


def test_infer_form_source_with_fn() -> None:
    def source_fn() -> Iterator[float]:
        yield from [1.0, 2.0, 3.0]

    src = IcoSource(source_fn)
    form = infer_ico_form(src)
    assert form.name == "() → Iterator[float]"


# ─── Sink ───


def sink_fn(x: float) -> None:
    pass


def test_infer_form_sink_with_generics() -> None:
    sink = IcoSink[float](sink_fn)
    form = infer_ico_form(sink)
    assert form.name == "Iterator[float] → ()"


def test_infer_form_sink_with_fn() -> None:
    sink = IcoSink(sink_fn)
    form = infer_ico_form(sink)
    assert form.name == "Iterator[float] → ()"


# ─── Batcher ───


def test_infer_form_batcher() -> None:
    batcher = IcoBatcher[int](batch_size=2)
    form = infer_ico_form(batcher)
    assert form.name == "Iterator[int] → Iterator[Iterator[int]]"


# ─── Context operator ───


def test_infer_form_context_operator_from_generics() -> None:
    ctx_op = IcoContextOperator[int, str, str](lambda i, c: c + str(i))
    form = infer_ico_form(ctx_op)
    assert form.name == "int, str → str"


def test_infer_form_context_operator_from_hints() -> None:
    def ctx_fn(i: int, c: str) -> str:
        return c + str(i)

    ctx_op = IcoContextOperator(ctx_fn)
    form = infer_ico_form(ctx_op)
    assert form.name == "int, str → str"


# ─── Context Pipeline ───


def test_infer_form_context_pipeline_from_generics() -> None:
    ctx_pipe = IcoContextPipeline[int, str, str](
        IcoContextOperator[int, str, str](lambda i, c: c + str(i)),
        IcoOperator[str, str](lambda c: c.upper()),
    )
    form = infer_ico_form(ctx_pipe)
    assert form.name == "int, str → str"


def test_infer_form_context_pipeline_from_type_hints() -> None:
    def apply_fn(i: int, c: str) -> str:
        return c + str(i)

    def step_fn(c: str) -> str:
        return c.upper()

    ctx_pipe = IcoContextPipeline(
        apply_fn,
        step_fn,
    )
    form = infer_ico_form(ctx_pipe)
    assert form.name == "int, str → str"


# ─── Epoch ───


def test_infer_form_epoch_from_generics() -> None:
    source = IcoSource[int](lambda: iter([1, 2, 3]))
    context_op = IcoContextOperator[int, str, str](lambda i, c: c + str(i))

    epoch = IcoEpoch(source, context_op)
    form = infer_ico_form(epoch)
    assert form.name == "Iterator[int], str → str"


def test_infer_form_epoch_from_type_hints() -> None:
    def source_fn() -> Iterator[int]:
        yield from [1, 2, 3]

    def context_fn(i: int, c: str) -> str:
        return c + str(i)

    source = IcoSource(source_fn)
    context_op = IcoContextOperator(context_fn)

    epoch = IcoEpoch(source, context_op)
    form = infer_ico_form(epoch)
    assert form.name == "Iterator[int], str → str"


def test_infer_form_epoch_from_type_hints_fn_only() -> None:
    def source_fn(_: None) -> Iterator[int]:
        yield from [1, 2, 3]

    def context_fn(i: int, c: str) -> str:
        return c + str(i)

    epoch = IcoEpoch(source_fn, context_fn)
    form = infer_ico_form(epoch)
    assert form.name == "Iterator[int], str → str"


def test_infer_form_epoch_from_type_hints_with_source() -> None:
    def source_fn() -> Iterator[int]:
        yield from [1, 2, 3]

    def context_fn(i: int, c: str) -> str:
        return c + str(i)

    source = IcoSource(source_fn)
    epoch = IcoEpoch(source, context_fn)
    form = infer_ico_form(epoch)
    assert form.name == "Iterator[int], str → str"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
