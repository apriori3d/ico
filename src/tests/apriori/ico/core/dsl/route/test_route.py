from typing import Any, TypeVar

import pytest

from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.core.route import IcoRoute, IcoRouteError, route
from ico.core.signature_check import (
    IcoSignatureError,
    strict_signatures,
    validate_signatures,
)

T = TypeVar("T")


def parity(x: int) -> str:
    return "even" if x % 2 == 0 else "odd"


def half(x: int) -> float:
    return x / 2


def negate(x: int) -> int:
    return -x


def describe_obj(x: object) -> str:
    return type(x).__name__


def str_len(x: str) -> int:
    return len(x)


def identity(x: T) -> T:
    return x


# ─── Semantics ───


def test_route_dispatches_by_key() -> None:
    r = route(parity, {"even": half, "odd": negate})
    assert [r(i) for i in range(4)] == [0.0, -1, 1.0, -3]


def test_route_calls_only_selected_branch() -> None:
    calls: list[str] = []

    def a(x: int) -> int:
        calls.append("a")
        return x

    def b(x: int) -> int:
        calls.append("b")
        return x

    route(parity, {"even": a, "odd": b})(2)
    assert calls == ["a"]


def test_route_default_for_unknown_key() -> None:
    r = route(parity, {"even": half}, default=describe_obj)
    assert r(2) == 1.0
    assert r(3) == "int"


def test_route_default_only() -> None:
    assert route(parity, {}, default=negate)(5) == -5


def test_route_unknown_key_raises() -> None:
    r = route(parity, {"even": half}, name="by_parity")
    with pytest.raises(IcoRouteError) as exc_info:
        r(3)
    assert isinstance(exc_info.value, LookupError)
    assert exc_info.value.key == "odd"
    assert str(exc_info.value) == "route 'by_parity': no branch for key 'odd'"


def test_route_requires_a_branch() -> None:
    with pytest.raises(ValueError, match="at least one branch"):
        IcoRoute(parity, {})


def test_route_structure() -> None:
    r = IcoRoute(parity, {"even": half, "odd": negate}, default=describe_obj)

    assert isinstance(r.key, IcoOperator)
    assert list(r.routes) == ["even", "odd"]
    assert r.default is not None
    assert list(r.children) == [r.key, *r.branches]
    assert len(r.branches) == 3
    assert r.name == "route"


def test_route_in_chain_and_stream() -> None:
    flow = (IcoOperator(negate) | route(parity, {"even": half, "odd": negate})).stream()
    assert list(flow(iter([1, 2]))) == [1, -1.0]


def test_route_static_types() -> None:
    # Checked by pyright and mypy in `make ci`.
    r: IcoOperatorProtocol[int, float] = route(parity, {"even": half, "odd": half})
    assert r(2) == 1.0


# ─── Signature ───


def test_route_signature_is_union_of_outputs() -> None:
    signature = route(parity, {"even": half, "odd": negate}).signature
    assert signature.format() == "int → float | int"


def test_route_signature_single_output_type() -> None:
    signature = route(parity, {"even": negate, "odd": negate}).signature
    assert signature.format() == "int → int"


def test_route_signature_includes_default_and_narrowest_input() -> None:
    signature = route(parity, {"even": half}, default=describe_obj).signature
    assert signature.format() == "int → float | str"


def test_route_signature_uninferred_with_untyped_branch() -> None:
    assert not route(parity, {"even": identity}).signature.infered


# ─── Strict mode and validation ───


def test_strict_route_rejects_conflicting_branch() -> None:
    with strict_signatures(), pytest.raises(IcoSignatureError) as exc_info:
        route(parity, {"even": half, "odd": str_len}, name="bad")  # type: ignore[arg-type,dict-item]

    issue = exc_info.value.issue
    assert issue.kind == "conflict"
    assert (issue.left, issue.right, issue.node) == ("parity", "str_len", "bad")
    assert str(exc_info.value) == (
        "branches of 'bad': inputs int of 'parity' and str of 'str_len' "
        "have no common type"
    )


def test_strict_route_accepts_compatible_branches() -> None:
    with strict_signatures():
        r = route(parity, {"even": half}, default=describe_obj)
    assert r(1) == "int"


def test_strict_flag_on_route_overrides_context() -> None:
    with pytest.raises(IcoSignatureError):
        route(parity, {"odd": str_len}, strict=True)  # type: ignore[arg-type,dict-item]

    with strict_signatures():
        route(parity, {"odd": str_len}, strict=False)  # type: ignore[arg-type,dict-item]


def test_validate_reports_route_conflict() -> None:
    mistyped: IcoOperator[Any, Any] = IcoOperator(str_len)
    flow = IcoOperator(negate) | route(parity, {"odd": mistyped}, name="r")

    errors = [i for i in validate_signatures(flow) if i.is_error]
    assert [(i.kind, i.node) for i in errors] == [("conflict", "r")]


def test_validate_compatible_route() -> None:
    flow = IcoOperator(negate) | route(parity, {"even": half, "odd": negate})
    assert validate_signatures(flow) == []


def test_describe_route(capsys: pytest.CaptureFixture[str]) -> None:
    route(parity, {"even": half, "odd": negate}, name="by_parity").describe()
    out = capsys.readouterr().out
    assert "by_parity" in out
    assert "float | int" in out
