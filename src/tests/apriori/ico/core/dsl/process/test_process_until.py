from typing import Any

import pytest

from ico.core.operator import IcoOperator
from ico.core.process import IcoProcess, IcoProcessLimitError
from ico.core.signature_check import (
    IcoSignatureError,
    strict_signatures,
    validate_signatures,
)


def halve(x: float) -> float:
    return x / 2


def small(x: float) -> bool:
    return x < 1


def never(x: float) -> bool:
    return False


def short(x: str) -> bool:
    return len(x) < 3


def increment(x: int) -> int:
    return x + 1


# ─── Conditional mode ───


def test_until_stops_when_condition_holds() -> None:
    process = IcoProcess(halve, until=small, max_iterations=100)
    assert process(10.0) == 0.625  # 10 → 5 → 2.5 → 1.25 → 0.625


def test_until_checked_before_first_iteration() -> None:
    calls: list[float] = []

    def tracked_halve(x: float) -> float:
        calls.append(x)
        return x / 2

    process = IcoProcess(tracked_halve, until=small, max_iterations=100)
    assert process(0.5) == 0.5
    assert calls == []


def test_until_sees_every_state() -> None:
    seen: list[float] = []

    def tracked_small(x: float) -> bool:
        seen.append(x)
        return x < 1

    IcoProcess(halve, until=tracked_small, max_iterations=100)(4.0)
    assert seen == [4.0, 2.0, 1.0, 0.5]


def test_until_uses_truthiness() -> None:
    def remaining(x: int) -> int:
        return 3 - x  # falsy (0) once x reaches 3

    def is_done(x: int) -> bool:
        return not remaining(x)

    assert IcoProcess(increment, until=is_done, max_iterations=10)(0) == 3


def test_limit_returns_last_state_by_default() -> None:
    process = IcoProcess(halve, until=never, max_iterations=3)
    assert process.on_limit == "return"
    assert process(8.0) == 1.0


def test_limit_not_reported_when_last_step_meets_condition() -> None:
    process = IcoProcess(halve, until=small, max_iterations=2, on_limit="raise")
    assert process(2.0) == 0.5


def test_limit_raises_when_requested() -> None:
    process = IcoProcess(
        halve, until=never, max_iterations=3, on_limit="raise", name="shrink"
    )
    with pytest.raises(IcoProcessLimitError) as exc_info:
        process(8.0)

    error = exc_info.value
    assert isinstance(error, RuntimeError)
    assert (error.process_name, error.iterations, error.state) == ("shrink", 3, 1.0)
    assert str(error) == ("process 'shrink': stop condition not met after 3 iterations")


def test_zero_max_iterations() -> None:
    assert IcoProcess(halve, until=never, max_iterations=0)(8.0) == 8.0
    with pytest.raises(IcoProcessLimitError):
        IcoProcess(halve, until=never, max_iterations=0, on_limit="raise")(8.0)


def test_conditional_structure() -> None:
    process = IcoProcess(halve, until=small, max_iterations=7)

    assert isinstance(process.until, IcoOperator)
    assert list(process.children) == [process.body, process.until]
    assert process.max_iterations == 7
    assert process.num_iterations == 7  # backward-compatible alias


# ─── Fixed mode (backward compatibility) ───


def test_fixed_mode_unchanged() -> None:
    process = IcoProcess(increment, num_iterations=3)

    assert process(0) == 3
    assert process.num_iterations == 3
    assert process.max_iterations == 3
    assert process.until is None
    assert list(process.children) == [process.body]


# ─── Argument validation ───


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "Either num_iterations or until"),
        ({"num_iterations": 3, "until": small}, "not both"),
        ({"num_iterations": 3, "max_iterations": 3}, "not both"),
        ({"until": small}, "until requires max_iterations"),
        ({"max_iterations": 3}, "max_iterations requires until"),
        ({"num_iterations": -1}, "non-negative"),
        ({"until": small, "max_iterations": -1}, "non-negative"),
        ({"until": small, "max_iterations": 3, "on_limit": "ignore"}, "on_limit"),
    ],
)
def test_invalid_arguments(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        IcoProcess(halve, **kwargs)


# ─── Signatures ───


def test_signature_unchanged_by_until() -> None:
    process = IcoProcess(halve, until=small, max_iterations=5)
    assert process.signature.format() == "float → float"


def test_strict_rejects_until_with_incompatible_input() -> None:
    with strict_signatures(), pytest.raises(IcoSignatureError) as exc_info:
        IcoProcess(halve, until=short, max_iterations=5)  # type: ignore[arg-type]

    issue = exc_info.value.issue
    assert issue.kind == "mismatch"
    assert (issue.left, issue.right) == ("halve", "short")


def test_strict_flag_on_process_overrides_context() -> None:
    with pytest.raises(IcoSignatureError):
        IcoProcess(halve, until=short, max_iterations=5, strict=True)  # type: ignore[arg-type]

    with strict_signatures():
        IcoProcess(halve, until=short, max_iterations=5, strict=False)  # type: ignore[arg-type]


def test_validate_reports_until_mismatch() -> None:
    mistyped: IcoOperator[Any, Any] = IcoOperator(short)
    flow = IcoOperator(halve) | IcoProcess(halve, until=mistyped, max_iterations=5)

    issues = validate_signatures(flow)
    assert [(i.kind, i.left, i.right) for i in issues] == [
        ("mismatch", "halve", "short")
    ]


def test_validate_compatible_process() -> None:
    flow = IcoOperator(halve) | IcoProcess(halve, until=small, max_iterations=5)
    assert validate_signatures(flow) == []


# ─── Describe ───


def test_describe_conditional_process(capsys: pytest.CaptureFixture[str]) -> None:
    IcoProcess(halve, until=small, max_iterations=100, name="shrink").describe()
    out = capsys.readouterr().out
    assert "until=small, max_iter=100" in out
    assert "small" in out


def test_describe_fixed_process(capsys: pytest.CaptureFixture[str]) -> None:
    IcoProcess(halve, num_iterations=3).describe()
    assert "num_iter=3" in capsys.readouterr().out
