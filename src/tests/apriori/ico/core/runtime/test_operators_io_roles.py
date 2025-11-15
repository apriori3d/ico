# tests/core/runtime/test_operator_io_roles.py

from apriori.ico.core import IcoOperator, IcoSink, IcoSource
from apriori.ico.core.utils import is_sink, is_source
from apriori.ico.tests.core.runtime.test_runtime_contour import drain_all

# ──── Tests ────


def test_infer_source_operator() -> None:
    src = IcoSource[int](lambda: [1, 2, 3])
    assert is_source(src)
    assert not is_sink(src)


def test_infer_sink_operator() -> None:
    sink = IcoSink[int](drain_all)
    assert is_sink(sink)
    assert not is_source(sink)


def test_infer_regular_operator() -> None:
    op = IcoOperator[int, float](lambda x: float(x))
    assert not is_source(op)
    assert not is_sink(op)


def test_infer_operator_without_annotations() -> None:
    def fn(x: str) -> None:
        pass

    op = IcoOperator(fn)
    assert is_sink(op)


if __name__ == "__main__":
    test_infer_operator_without_annotations()
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
