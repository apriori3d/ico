from typing import Any, TypeVar

import pytest

from ico.core.fork import IcoFork, fork
from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.core.signature_check import (
    IcoSignatureError,
    narrowest_type,
    strict_signatures,
    validate_signatures,
)

T = TypeVar("T")


def to_str(x: int) -> str:
    return str(x)


def half(x: int) -> float:
    return x / 2


def double(x: int) -> int:
    return x * 2


def is_even(x: int) -> bool:
    return x % 2 == 0


def str_len(x: str) -> int:
    return len(x)


def describe_obj(x: object) -> str:
    return type(x).__name__


def identity(x: T) -> T:
    return x


def to_pair(x: int) -> tuple[int, int]:
    return (x, x)


def total(xs: tuple[str, float]) -> int:
    return len(xs[0]) + int(xs[1])


# ─── Semantics ───


def test_fork_two_branches() -> None:
    f = fork(to_str, half)
    assert f(3) == ("3", 1.5)


def test_fork_six_branches() -> None:
    f = fork(to_str, half, double, is_even, describe_obj, to_pair)
    assert f(4) == ("4", 2.0, 8, True, "int", (4, 4))


def test_fork_more_than_six_branches() -> None:
    f = fork(double, double, double, double, double, double, double)
    assert f(1) == (2,) * 7


def test_fork_runs_branches_in_order_with_same_input() -> None:
    calls: list[tuple[str, object]] = []

    def first(x: list[int]) -> int:
        calls.append(("first", x))
        return 1

    def second(x: list[int]) -> int:
        calls.append(("second", x))
        return 2

    item = [1, 2]
    assert fork(first, second)(item) == (1, 2)
    assert [name for name, _ in calls] == ["first", "second"]
    assert all(x is item for _, x in calls)


def test_fork_wraps_callables_and_keeps_operators() -> None:
    op = IcoOperator(to_str)
    f = IcoFork[int, tuple[str, float]](op, half)

    assert f.branches[0] is op
    assert isinstance(f.branches[1], IcoOperator)
    assert list(f.children) == list(f.branches)
    assert len(f) == 2


def test_fork_requires_two_branches() -> None:
    with pytest.raises(ValueError, match="at least two branches"):
        IcoFork[int, tuple[str]](to_str)


def test_fork_name() -> None:
    assert IcoFork[int, Any](to_str, half).name == "fork"
    assert IcoFork[int, Any](to_str, half, name="stats").name == "stats"


def test_fork_in_chain() -> None:
    flow = IcoOperator(double) | fork(to_str, half) | IcoOperator(total)
    assert flow(5) == 2 + 5


def test_fork_static_types() -> None:
    # Checked by pyright and mypy in `make ci`.
    two: IcoOperatorProtocol[int, tuple[str, float]] = fork(to_str, half)
    three: IcoOperatorProtocol[int, tuple[str, float, bool]] = fork(
        to_str, half, is_even
    )
    many: IcoOperatorProtocol[int, tuple[Any, ...]] = fork(
        double, double, double, double, double, double, double
    )
    assert two(1) == ("1", 0.5)
    assert three(2) == ("2", 1.0, True)
    assert len(many(0)) == 7


# ─── Signature ───


def test_fork_signature() -> None:
    assert fork(to_str, half, is_even).signature.format() == (
        "int → tuple[str, float, bool]"
    )


def test_fork_signature_uses_narrowest_input() -> None:
    signature = fork(describe_obj, to_str).signature
    assert signature.i is int
    assert signature.infered


def test_fork_signature_uninferred_with_untyped_branch() -> None:
    assert not fork(to_str, identity).signature.infered


def test_fork_signature_uninferred_on_conflict() -> None:
    assert not fork(to_str, str_len).signature.infered  # type: ignore[misc]


def test_narrowest_type() -> None:
    assert narrowest_type([object, int, int]) is int
    assert narrowest_type([bool, int]) is bool
    assert narrowest_type([Any, int]) is int
    assert narrowest_type([Any]) is Any
    assert narrowest_type([str, int]) is None
    assert narrowest_type([]) is None


# ─── Strict mode and validation ───


def test_strict_fork_rejects_conflicting_inputs() -> None:
    with strict_signatures(), pytest.raises(IcoSignatureError) as exc_info:
        fork(to_str, str_len, name="bad")  # type: ignore[misc]

    issue = exc_info.value.issue
    assert issue.kind == "conflict"
    assert issue.is_error
    assert (issue.left, issue.right, issue.node) == ("to_str", "str_len", "bad")
    assert str(exc_info.value) == (
        "branches of 'bad': inputs int of 'to_str' and str of 'str_len' have no common type"
    )


def test_strict_fork_accepts_compatible_inputs() -> None:
    with strict_signatures():
        f = fork(to_str, describe_obj, identity)
    assert f(1) == ("1", "int", 1)


def test_strict_flag_on_fork_overrides_context() -> None:
    with pytest.raises(IcoSignatureError):
        fork(to_str, str_len, strict=True)  # type: ignore[misc]

    with strict_signatures():
        fork(to_str, str_len, strict=False)  # type: ignore[misc]


def test_strict_chain_checks_link_into_fork() -> None:
    with strict_signatures(), pytest.raises(IcoSignatureError) as exc_info:
        _ = IcoOperator(to_str) | fork(double, half)  # type: ignore[arg-type]
    assert exc_info.value.issue.kind == "mismatch"


def test_validate_reports_fork_conflict() -> None:
    mistyped: IcoOperator[Any, Any] = IcoOperator(str_len)
    flow = IcoOperator(double) | fork(to_str, mistyped, name="stats")

    issues = validate_signatures(flow)

    errors = [i for i in issues if i.is_error]
    assert [(i.kind, i.node) for i in errors] == [("conflict", "stats")]
    # The fork signature is uninferred, so the link into it is only a warning.
    assert [i.kind for i in issues if not i.is_error] == ["uninferred"]


def test_validate_compatible_fork() -> None:
    flow = IcoOperator(double) | fork(to_str, half) | IcoOperator(total)
    assert validate_signatures(flow) == []


# ─── Describe ───


def test_describe_fork(capsys: pytest.CaptureFixture[str]) -> None:
    (IcoOperator(double) | fork(to_str, half, name="stats")).describe()
    out = capsys.readouterr().out
    assert "stats" in out
    assert "tuple[str, float]" in out
