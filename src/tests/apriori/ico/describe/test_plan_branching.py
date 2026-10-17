from ico import IcoOperator, plan_view
from ico.core.fork import fork
from ico.core.route import route
from ico.describe.plan.json_renderer import PlanCell, PlanView


def strip(text: str) -> str:
    return text.strip()


def upper(text: str) -> str:
    return text.upper()


def length(text: str) -> int:
    return len(text)


def double(n: int) -> int:
    return n * 2


def first(text: str) -> str:
    return text[:1]


def text(cell: PlanCell) -> str:
    return "".join(segment["text"] for segment in cell)


def rows(view: PlanView) -> list[tuple[str, str]]:
    return [(text(row[0]), text(row[1])) for row in view["rows"]]


def keywords(view: PlanView) -> list[str]:
    return [
        segment["text"].strip()
        for row in view["rows"]
        for segment in row[0]
        if segment["style"] == "keyword"
    ]


def test_fork_branches_are_separated() -> None:
    forked = fork(
        IcoOperator(strip) | IcoOperator(upper),
        IcoOperator(length) | IcoOperator(double),
        IcoOperator(upper),
    )
    view = plan_view(forked)

    assert rows(view) == [
        ("╭── fork into 🔀IcoFork(branches=3)", "str → tuple[str, int, str]"),
        ("├─▸ branch 1", "str → str"),
        ("│   strip", "str → str"),
        ("│   upper", "str → str"),
        ("├─▸ branch 2", "str → int"),
        ("│   length", "str → int"),
        ("│   double", "int → int"),
        ("├─▸ branch 3", "str → str"),
        ("│   upper", "str → str"),
        ("╰─▸ collect", "tuple[str, int, str]"),
    ]
    assert keywords(view) == [
        "fork into 🔀",
        "branch 1",
        "branch 2",
        "branch 3",
        "collect",
    ]
    assert text(view["rows"][0][2]) == "fork"


def test_route_shows_key_cases_and_default() -> None:
    routed = route(
        IcoOperator(first),
        {"a": IcoOperator(strip) | IcoOperator(upper), "b": IcoOperator(strip)},
        default=IcoOperator(upper),
    )
    flow = [line for line, _ in rows(plan_view(routed))]

    assert flow == [
        "╭── route by 🚦IcoRoute(keys='a', 'b', default)",
        "├─▸ key",
        "│   first",
        "├─▸ case 'a'",
        "│   strip",
        "│   upper",
        "├─▸ case 'b'",
        "│   strip",
        "├─▸ default",
        "│   upper",
        "╰─▸ merge",
    ]


def test_route_without_default() -> None:
    routed = route(IcoOperator(first), {"a": IcoOperator(strip)})
    flow = [line for line, _ in rows(plan_view(routed))]

    assert flow[0] == "╭── route by 🚦IcoRoute(keys='a')"
    assert "├─▸ default" not in flow


def test_nested_fork_keeps_its_branches() -> None:
    inner = fork(IcoOperator(strip), IcoOperator(upper))
    outer = fork(IcoOperator(strip) | inner, IcoOperator(length))
    flow = [line for line, _ in rows(plan_view(outer))]

    assert flow == [
        "╭── fork into 🔀IcoFork(branches=2)",
        "├─▸ branch 1",
        "│   strip",
        "│   ╭── fork into 🔀IcoFork(branches=2)",
        "│   ├─▸ branch 1",
        "│   │   strip",
        "│   ├─▸ branch 2",
        "│   │   upper",
        "│   ╰─▸ collect",
        "├─▸ branch 2",
        "│   length",
        "╰─▸ collect",
    ]
