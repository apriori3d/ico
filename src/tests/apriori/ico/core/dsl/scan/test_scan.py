"""IcoScan: a fold with an output per item."""

from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import cast

import pytest

from ico import IcoOperator, IcoSource, IcoStream, plan_view, validate_signatures
from ico.core.runtime.cancel import IcoCancelled, IcoCancelToken, cancellable
from ico.core.scan import IcoScan


@dataclass(frozen=True)
class Pose:
    """The state: a pose fitted on the previous frame (a warm start)."""

    x: float = 0.0


def frames() -> Iterator[float]:
    return iter([1.0, 2.0, 4.0])


def fit(pair: tuple[float, Pose]) -> tuple[str, Pose]:
    target, pose = pair
    fitted = Pose((pose.x + target) / 2)  # halfway from the warm start
    return f"{fitted.x:g}", fitted


def test_a_scan_yields_an_output_and_the_state_per_item() -> None:
    scan = IcoScan(IcoSource(frames), fit)
    assert list(scan(Pose())) == [
        ("0.5", Pose(0.5)),
        ("1.25", Pose(1.25)),
        ("2.625", Pose(2.625)),
    ]
    assert scan.signature.format() == "Pose → Iterator[tuple[str, Pose]]"


def test_a_scan_starts_from_the_output_of_the_previous_node() -> None:
    def start(x: float) -> Pose:
        return Pose(x)

    flow = IcoOperator(start) | IcoScan(IcoSource(frames), fit)
    assert [out for out, _ in flow(4.0)] == ["2.5", "2.25", "3.125"]
    assert validate_signatures(flow) == []


def test_a_scan_is_lazy_and_closes_its_source() -> None:
    pulled: list[float] = []
    closed: list[bool] = []

    def tracked() -> Iterator[float]:
        try:
            for frame in [1.0, 2.0, 4.0]:
                pulled.append(frame)
                yield frame
        finally:
            closed.append(True)

    pairs = cast(Generator[tuple[str, Pose]], IcoScan(IcoSource(tracked), fit)(Pose()))
    assert pulled == []
    assert next(pairs)[0] == "0.5" and pulled == [1.0]
    pairs.close()  # the consumer stops
    assert closed == [True]


def test_outputs_flow_on_to_a_stream() -> None:
    def out_only(pair: tuple[str, Pose]) -> str:
        return pair[0]

    flow = IcoScan(IcoSource(frames), fit) | IcoStream(IcoOperator(out_only))
    assert list(flow(Pose())) == ["0.5", "1.25", "2.625"]
    assert validate_signatures(flow) == []


def test_until_stops_before_the_next_item() -> None:
    def close_to_two(pose: Pose) -> bool:
        return pose.x >= 1.0

    scan = IcoScan(IcoSource(frames), fit, until=close_to_two)
    assert [out for out, _ in scan(Pose())] == ["0.5", "1.25"]
    assert list(scan(Pose(3.0))) == []  # holds at the start: nothing pulled


def test_a_step_or_until_of_the_wrong_type_is_reported() -> None:
    def words() -> Iterator[str]:
        return iter(["a"])

    def positive(x: int) -> bool:
        return x > 0

    wrong_items = IcoScan(IcoSource(words), fit)  # pyright: ignore[reportArgumentType]
    [issue] = validate_signatures(wrong_items)
    assert issue.output == tuple[str, Pose] and issue.input == tuple[float, Pose]

    wrong_until = IcoScan(IcoSource(frames), fit, until=positive)  # pyright: ignore[reportArgumentType]
    [issue] = validate_signatures(wrong_until)
    assert issue.output == Pose and issue.input is int


def test_a_cancelled_scan_stops_between_items() -> None:
    token = IcoCancelToken()

    def cancel_after_first(pair: tuple[float, Pose]) -> tuple[str, Pose]:
        token.cancel("enough")
        return fit(pair)

    pairs = IcoScan(IcoSource(frames), cancel_after_first)
    with cancellable(token), pytest.raises(IcoCancelled, match="enough"):
        list(pairs(Pose()))


def test_the_plan_shows_the_source_the_step_until_and_what_it_yields() -> None:
    def converged(pose: Pose) -> bool:
        return pose.x > 10

    scan = IcoScan(IcoSource(frames), fit, until=converged)
    rows = ["".join(s["text"] for s in row[0]) for row in plan_view(scan)["rows"]]
    assert rows == [
        "🧭IcoScan()",
        "╭── for each in ",
        "│   📚IcoSource(frames)",
        "├─▸ scan",
        "│   fit",
        "├─▸ until",
        "│   converged",
        "╰─▸ yield each",
    ]
