import json
from collections.abc import Iterator
from typing import Any

from rich.console import Console

from ico import IcoOperator, IcoSource, PlanView, plan_view
from ico.core.async_stream import IcoAsyncStream
from ico.core.batcher import IcoBatcher
from ico.core.process import IcoProcess
from ico.core.runtime.progress import IcoProgress
from ico.core.stream import IcoStream
from ico.describe.plan.json_renderer import PlanCell
from ico.describe.plan.options import PlanRendererOptions
from ico.describe.plan.rich_renderer.plan_renderer import PlanRenderer
from ico.describe.rich_style import DescribeStyle

# ─── Fixtures ───


def words() -> Iterator[str]:
    yield from ["a", "b"]


def upper(text: str) -> str:
    return text.upper()


def halve(x: float) -> float:
    return x / 2


def small(x: float) -> bool:
    return x < 1


def text(cell: PlanCell) -> str:
    return "".join(segment["text"] for segment in cell)


def styled(cell: PlanCell, style: str) -> list[str]:
    return [s["text"].strip() for s in cell if s["style"] == style]


def flow_column(view: PlanView) -> list[str]:
    return [text(row[0]) for row in view["rows"]]


def stream_flow() -> Any:
    progress = IcoProgress[str](total=2, name="items")
    flow = IcoSource(words) | IcoStream(progress | IcoOperator(upper))
    pool = IcoAsyncStream(lambda: IcoOperator(upper), pool_size=4, on_error="skip")
    return flow | pool | IcoBatcher[str](batch_size=2)


# ─── Rows ───


def test_rows_are_the_console_plan() -> None:
    flow = stream_flow()
    view = plan_view(flow)

    console = Console(record=True, width=300, color_system=None)
    PlanRenderer(PlanRendererOptions(query_iterable_size=False), console).render(flow)
    printed = console.export_text()

    assert view["columns"] == ["Flow", "Signature", "Name"]
    assert view["title"] == flow.name
    for row in view["rows"]:
        assert all(text(cell) in printed for cell in row)
    assert flow_column(view) == [
        "📚IcoSource(words)",
        "╭── for each in 🎞️ IcoStream()",
        "│   ⏳IcoProgress(items)",
        "│   upper",
        "╰─▸ yield",
        "╭── parallel in 🚀IcoAsyncStream(pool_size=4, on_error=skip)",
        "│   upper",
        "╰─▸ yield",
        "📦IcoBatcher(batch_size=2)",
    ]


def test_segments_carry_describe_roles() -> None:
    view = plan_view(stream_flow())
    rows = view["rows"]
    roles = {member.name for member in DescribeStyle}

    for row in rows:
        for cell in row:
            for segment in cell:
                assert segment["style"] is None or segment["style"] in roles
    header, footer = rows[1][0], rows[4][0]
    assert styled(header, "group") == ["╭──"]
    assert styled(header, "keyword")[0].startswith("for each in")
    assert styled(footer, "keyword") == ["yield"]
    assert styled(rows[0][0], "fn") == ["words"]
    assert styled(rows[5][0], "meta") == ["pool_size=4, on_error=skip"]
    assert styled(rows[2][2], "text") == ["items"]
    assert text(rows[0][1]) == "() → Iterator[str]"
    assert "signature" in {s["style"] for s in rows[0][1]}


def test_common_signature_prefix_is_dimmed_like_the_console() -> None:
    view = plan_view(stream_flow())
    signature = view["rows"][3][1]  # upper: str → str

    assert text(signature) == "str → str"
    assert all(s["dim"] for s in signature if s["style"] == "signature")


def test_process_group_keywords() -> None:
    process = IcoProcess(IcoOperator(halve), until=IcoOperator(small), max_iterations=9)
    view = plan_view(process)

    assert flow_column(view)[0].startswith("╭── iterate in ")
    assert "until=small, max_iter=9" in flow_column(view)[0]
    assert flow_column(view)[-1] == "╰─▸ emit"


def test_view_is_json_and_sources_are_not_called() -> None:
    calls: list[int] = []

    def counted() -> list[int]:
        calls.append(1)
        return [1, 2]

    view = plan_view(IcoSource(counted))

    assert calls == []
    assert json.loads(json.dumps(view)) == view


class Steps(IcoOperator[float, float]):
    """Counts the steps of each call on a progress node it owns."""

    def __init__(self) -> None:
        self.progress: IcoProgress[int] = IcoProgress(total=4, name="denoise")
        super().__init__(fn=halve, name="steps", children=[self.progress])


def test_an_operator_with_its_own_progress_is_a_row_not_a_subflow() -> None:
    view = plan_view(IcoStream(Steps()))
    rows = [[text(cell) for cell in row] for row in view["rows"]]
    keys = view["keys"]

    [operator] = [i for i, row in enumerate(rows) if row[0].endswith("Steps(halve)")]
    progress = operator + 1
    assert rows[operator][1] == "float → float"  # its own signature
    assert rows[progress][0].endswith("╰ reports ⏳IcoProgress(denoise)")
    assert (rows[progress][1], rows[progress][2]) == ("", "denoise")
    assert keys[progress] == f"{keys[operator]}.0"
    assert not any("Subflow" in row[0] for row in rows)
