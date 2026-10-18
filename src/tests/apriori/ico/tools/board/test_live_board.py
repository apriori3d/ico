import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit

import pytest

from ico import IcoOperator, IcoRuntime, LiveBoardTool, PlanView
from ico.core.runtime.progress import IcoProgress
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.tools.board import BoardRuns, LiveBoard

# ─── Helpers ───


def numbers() -> Iterator[int]:
    yield from range(4)


def ignore(item: int) -> None:
    pass


def fail(item: int) -> int:
    raise RuntimeError("boom")


def get(board: LiveBoard, path: str, token: str | None = None) -> Any:
    url = board.url
    assert url is not None
    parts = urlsplit(url)
    sep = "&" if "?" in path else "?"
    full = f"http://{parts.netloc}{path}{sep}token={token or board.token}"
    with urllib.request.urlopen(full, timeout=5) as response:
        body = response.read()
        if response.headers["Content-Type"].startswith("application/json"):
            return json.loads(body)
        return body.decode()


@pytest.fixture
def board() -> Iterator[tuple[BoardRuns, LiveBoard]]:
    runs = BoardRuns(max_finished=2)
    board = LiveBoard(runs)
    board.start()
    yield runs, board
    board.stop()


# ─── Runs ───


def test_runs_keep_progress_in_flow_order_and_drop_old_finished() -> None:
    runs = BoardRuns(max_finished=2)
    first = runs.start("first")
    first.progress({"key": "1", "label": "b", "done": 0, "total": 4})
    first.progress({"key": "0", "label": "a", "done": 1, "total": None, "rate": 2.0})
    first.progress({"key": "1", "label": "b", "done": 3, "total": 4})
    first.finish(summary="2 outputs")
    for title in ("second", "third"):
        runs.start(title).finish("failed here")
    active = runs.start("active")

    views = runs.runs()

    assert [v["title"] for v in views] == ["active", "third", "second"]
    assert views[0]["state"] == "running"
    assert (views[1]["state"], views[1]["error"]) == ("failed", "failed here")
    assert runs.plan(first.id) is None
    assert active.view()["progress"] == []
    assert [p["key"] for p in first.view()["progress"]] == ["1", "0"]
    assert (first.view()["summary"], views[1]["summary"]) == ("2 outputs", None)
    assert first.view()["progress"][0]["done"] == 3
    assert first.view()["progress"][1]["total"] is None


def test_max_finished_must_be_positive() -> None:
    with pytest.raises(ValueError):
        BoardRuns(max_finished=0)


# ─── HTTP ───


def test_board_serves_page_runs_and_plan(board: tuple[BoardRuns, LiveBoard]) -> None:
    runs, live = board
    plan: PlanView = {
        "title": "flow",
        "columns": ["Flow"],
        "rows": [[[{"text": "upper", "style": "fn", "dim": False}]]],
        "keys": ["0"],
    }
    run = runs.start("demo", plan)

    page = get(live, "/")
    listed = get(live, "/api/runs")

    assert live.url is not None and live.url.startswith("http://127.0.0.1:")
    assert "getStatus" in page and "<script src" not in page
    assert [r["id"] for r in listed["runs"]] == [run.id]
    assert listed["runs"][0]["has_plan"] is True
    assert get(live, f"/api/plan?run={run.id}") == plan
    assert get(live, "/api/plan?run=missing") is None


def test_board_needs_the_token(board: tuple[BoardRuns, LiveBoard]) -> None:
    _, live = board
    for path in ("/", "/api/runs"):
        with pytest.raises(urllib.error.HTTPError) as info:
            get(live, path, token="wrong")
        assert info.value.code == 403


def test_stopped_board_has_no_url() -> None:
    live = LiveBoard(BoardRuns())
    live.start()
    live.stop()
    assert live.url is None


# ─── Tool ───


def test_tool_adds_a_run_with_progress_and_plan(
    board: tuple[BoardRuns, LiveBoard],
) -> None:
    runs, live = board
    progress = IcoProgress[int](total=4, name="items")
    flow = IcoSource(numbers) | progress.stream() | IcoSink(ignore)
    flow.name = "numbers flow"

    IcoRuntime(flow, tools=[LiveBoardTool(runs)]).activate().run().deactivate()

    [run] = get(live, "/api/runs")["runs"]
    assert (run["title"], run["state"], run["error"]) == ("numbers flow", "done", None)
    assert [(p["label"], p["done"], p["total"]) for p in run["progress"]] == [
        ("items", 4.0, 4.0)
    ]
    plan = get(live, f"/api/plan?run={run['id']}")
    assert plan["title"] == "numbers flow"
    assert "".join(s["text"] for s in plan["rows"][0][0]) == "📚IcoSource(numbers)"


def test_tool_marks_a_faulted_run_failed() -> None:
    runs = BoardRuns()
    flow = IcoSource(numbers) | IcoOperator(fail).stream() | IcoSink(ignore)
    runtime = IcoRuntime(flow, tools=[LiveBoardTool(runs, title="boom")])
    runtime.activate()
    with pytest.raises(RuntimeError):
        runtime.run()
    runtime.deactivate()

    [run] = runs.runs()
    assert (run["title"], run["state"]) == ("boom", "failed")
