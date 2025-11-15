from apriori.ico.core import (
    IcoForm,
    IcoOperator,
    IcoPipeline,
    IcoProcess,
    IcoSource,
    IcoStream,
)
from apriori.ico.core.meta.ico_form import infer_ico_form

# ─── Operator ───


def test_infer_form_operator_generics() -> None:
    op = IcoOperator[int, float](lambda x: float(x))
    form = infer_ico_form(op)
    assert isinstance(form, IcoForm)
    assert form.name == "int → float"


def test_infer_form_operator_annotations() -> None:
    def fn(x: str) -> int:
        return len(x)

    op = IcoOperator(fn)
    assert infer_ico_form(op).name == "str → int"


def test_infer_form_operator_no_hints() -> None:
    op = IcoOperator(lambda x: x)  # type: ignore
    assert infer_ico_form(op).name == "Any → Any"


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

    assert infer_ico_form(mapped).name == "Iterable[int] → Iterable[float]"
    assert infer_ico_form(streamed).name == "Iterable[int] → Iterable[float]"


# ─── Process ───


def test_infer_form_process() -> None:
    proc = IcoProcess[float](lambda c: c + 1, num_iterations=3)
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
    assert infer_ico_form(p.context).name == "int → float"
    assert infer_ico_form(p.body[0]).name == "float → float"
    assert infer_ico_form(p.output).name == "float → str"


def test_infer_form_pipeline_without_hints() -> None:
    p = IcoPipeline[int, float, str](
        context=lambda x: float(x),
        body=[lambda x: x * 2],
        output=lambda x: str(x),
    )
    assert infer_ico_form(p).name == "int → float → str"
    assert infer_ico_form(p.context).name == "Any → Any"
    assert infer_ico_form(p.body[0]).name == "Any → Any"
    assert infer_ico_form(p.output).name == "Any → Any"


# ─── Source ───


def test_infer_form_source_with_generics() -> None:
    src = IcoSource[float](lambda: [1.0, 2.0, 3.0])
    form = infer_ico_form(src)
    assert form.name == "() → Iterable[float]"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
