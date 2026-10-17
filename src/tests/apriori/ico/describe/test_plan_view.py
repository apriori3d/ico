import json
from collections.abc import Iterator
from typing import Any, get_args

from ico import IcoOperator, IcoSource, PlanNodeView, plan_view
from ico.core.async_stream import IcoAsyncStream
from ico.core.batcher import IcoBatcher
from ico.core.fork import fork
from ico.core.process import IcoProcess
from ico.core.runtime.progress import IcoProgress
from ico.core.stream import IcoStream
from ico.describe.plan.json_renderer import Layout
from ico.registry import IcoRegistry, register

# ─── Fixtures ───


def words() -> Iterator[str]:
    yield from ["a", "b"]


def upper(text: str) -> str:
    return text.upper()


def lower(text: str) -> str:
    return text.lower()


def halve(x: float) -> float:
    return x / 2


def small(x: float) -> bool:
    return x < 1


def nodes(view: PlanNodeView) -> Iterator[PlanNodeView]:
    yield view
    for child in view["children"]:
        yield from nodes(child)


# ─── Schema ───


def test_every_node_has_the_schema_fields() -> None:
    flow = (
        IcoSource(words)
        | IcoProgress[Iterator[str]](total=2, name="items")
        | IcoStream(upper)
        | IcoBatcher[str](batch_size=2)
    )
    view = plan_view(flow)

    layouts = set(get_args(Layout))
    for node in nodes(view):
        assert set(node) == {"label", "sig", "ref", "badges", "layout", "children"}
        assert isinstance(node["label"], str) and node["label"]
        assert node["sig"] is None or isinstance(node["sig"], str)
        assert node["ref"] is None or isinstance(node["ref"], str)
        assert all(isinstance(badge, str) for badge in node["badges"])
        assert node["layout"] in layouts
    as_json: Any = json.loads(json.dumps(view))
    assert as_json == view


# ─── Layouts ───


def test_chain_is_flattened_into_a_sequence() -> None:
    flow = (
        IcoSource(words)
        | IcoProgress[Iterator[str]](total=2, name="items")
        | IcoStream(upper)
    )
    view = plan_view(flow)

    assert view["layout"] == "sequence"
    assert [c["label"] for c in view["children"]] == ["IcoSource", "items", "IcoStream"]
    source, progress, stream = view["children"]
    assert (source["sig"], source["badges"]) == ("() → Iterator[str]", ["words"])
    assert progress["sig"] == "Iterator[str] → Iterator[str]"
    assert stream["layout"] == "sequence"
    assert [c["label"] for c in stream["children"]] == ["upper"]


def test_async_stream_and_fork_are_parallel() -> None:
    astream = IcoAsyncStream(
        lambda: IcoOperator(upper), pool_size=4, on_error="skip", name="pool"
    )
    forked = fork(IcoOperator(upper), IcoOperator(lower))

    stream_view, fork_view = plan_view(astream), plan_view(forked)

    assert (stream_view["label"], stream_view["layout"]) == ("pool", "parallel")
    assert stream_view["badges"] == ["pool_size=4", "on_error=skip"]
    assert [c["label"] for c in stream_view["children"]] == ["upper"]
    assert fork_view["layout"] == "parallel"
    assert [c["label"] for c in fork_view["children"]] == ["upper", "lower"]
    assert fork_view["sig"] == "str → tuple[str, str]"


def test_process_is_a_loop_with_its_settings() -> None:
    process = IcoProcess(IcoOperator(halve), until=IcoOperator(small), max_iterations=9)
    view = plan_view(process)

    assert view["layout"] == "loop"
    assert view["badges"] == ["until=small", "max_iter=9"]
    assert [c["label"] for c in view["children"]] == ["halve", "small"]


def test_leaf_settings_are_badges() -> None:
    view = plan_view(IcoBatcher[int](batch_size=8))
    assert (view["layout"], view["badges"]) == ("leaf", ["batch_size=8"])


# ─── Registered operators ───


def test_registered_operator_is_an_opaque_leaf_with_its_ref() -> None:
    registry = IcoRegistry()
    shout = register("shout", 1, registry=registry)(
        IcoOperator(upper) | IcoOperator(lower)
    )
    view = plan_view(IcoSource(words) | IcoStream(shout))

    [inner] = view["children"][1]["children"]
    assert (inner["ref"], inner["layout"], inner["children"]) == ("shout@1", "leaf", [])


def test_sources_are_not_called_by_default() -> None:
    calls: list[int] = []

    def counted() -> list[int]:
        calls.append(1)
        return [1, 2]

    view = plan_view(IcoSource(counted))
    assert calls == []
    assert view["badges"] == ["counted"]
