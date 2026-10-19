"""tap: run a child for its side effect and pass the input on."""

from collections.abc import Iterator

from ico import plan_view
from ico.core.operator import IcoOperator
from ico.core.tap import IcoTap, tap
from ico.registry import IcoRegistry, register
from ico.spec import NodeSpec, build, run, to_spec

REGISTRY = IcoRegistry()
written: list[str] = []


@register("tap.report", 1, registry=REGISTRY)
def report(x: int) -> str:
    """Write a report of a number; return its text."""
    written.append(f"report {x}")
    return f"#{x}"


@register("tap.double", 1, registry=REGISTRY)
def double(x: int) -> int:
    """Double."""
    return 2 * x


def test_the_child_runs_and_the_input_goes_on() -> None:
    seen: list[int] = []

    def write(x: int) -> str:
        seen.append(x)
        return "ignored"

    flow = tap(write) | IcoOperator(double)
    assert flow(5) == 10
    assert seen == [5]


def test_an_iterator_child_is_consumed() -> None:
    produced: list[int] = []

    def frames(x: int) -> Iterator[int]:
        for i in range(x):
            produced.append(i)
            yield i

    assert tap(frames)(3) == 3
    assert produced == [0, 1, 2]


def test_the_signature_is_the_child_input() -> None:
    node = IcoTap(IcoOperator(report))
    assert node.signature.format() == "int → int"


def test_spec_round_trip_and_run() -> None:
    written.clear()
    spec = NodeSpec(
        kind="seq",
        children=(
            NodeSpec(kind="tap", children=(NodeSpec(kind="op", op="tap.report@1"),)),
            NodeSpec(kind="op", op="tap.double@1"),
        ),
    )
    flow = build(spec, REGISTRY)
    assert to_spec(flow).root.children[0].kind == "tap"
    assert run(spec, 4, registry=REGISTRY).outputs == (8,)
    assert written == ["report 4"]


def test_the_plan_says_the_input_goes_on() -> None:
    view = plan_view(tap(IcoOperator(report)) | IcoOperator(double))
    text = "\n".join("".join(seg["text"] for seg in row[0]) for row in view["rows"])
    assert "tap " in text and "pass the input on" in text
