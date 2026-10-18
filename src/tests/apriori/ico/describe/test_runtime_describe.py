"""Describing a runtime subscribes only to that runtime, and only while it
collects the states of its nodes."""

import io
from collections.abc import Iterator
from typing import Any

import pytest
from rich.console import Console

from ico.core.runtime.command import IcoRuntimeCommand
from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.runtime.progress import IcoProgress
from ico.core.runtime.runtime import IcoRuntime
from ico.core.runtime.state import IcoStateRequestCommand
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.describe.runtime.rich_renderer.tree_renderer import RuntimeTreeRenderer


def numbers() -> Iterator[int]:
    yield from range(3)


def ignore(x: Any) -> None:
    pass


def make_runtime(name: str) -> tuple[IcoRuntime, IcoProgress[int]]:
    progress: IcoProgress[int] = IcoProgress(total=3, name=name)
    flow = IcoSource(numbers) | progress.stream() | IcoSink(ignore)
    return IcoRuntime(flow), progress


class Counting(RuntimeTreeRenderer):
    """Counts the events forwarded to it."""

    def __init__(self, console: Console) -> None:
        super().__init__(console=console)
        self.events = 0

    def on_forward_event(self, event: IcoRuntimeEvent) -> None:
        self.events += 1
        super().on_forward_event(event)


def render(runtime: IcoRuntime) -> tuple[RuntimeTreeRenderer, str]:
    out = io.StringIO()
    renderer = RuntimeTreeRenderer(console=Console(file=out, width=160))
    renderer.render(runtime)
    return renderer, out.getvalue()


def test_describe_leaves_no_subscriber() -> None:
    runtime, _ = make_runtime("first")
    runtime.activate()
    tools = list(runtime.toolbox.tools)

    renderer, text = render(runtime)
    render(runtime)  # twice: the renderer was unsubscribed

    assert runtime.toolbox.tools == tools
    assert renderer.runtime_parent is None
    assert "Ready" in text
    runtime.deactivate()


def test_runtimes_do_not_share_subscribers() -> None:
    first, _ = make_runtime("first")
    second, _ = make_runtime("second")
    renderer, _ = render(first)

    second.activate().run().deactivate()

    assert renderer not in second.toolbox.tools
    assert not hasattr(IcoRuntime, "event_listeners")


def test_a_failing_node_does_not_leave_the_subscriber(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, progress = make_runtime("broken")

    def fail(command: IcoRuntimeCommand) -> None:
        if isinstance(command, IcoStateRequestCommand):
            raise RuntimeError("no answer")

    monkeypatch.setattr(progress, "on_command", fail)
    renderer = Counting(console=Console(file=io.StringIO()))
    with pytest.raises(RuntimeError, match="no answer"):
        renderer.render(runtime)

    assert renderer not in runtime.toolbox.tools
    assert renderer.runtime_parent is None
    # Events of later runs do not reach it (a shared listener list did).
    seen = renderer.events
    make_runtime("later")[0].activate().run().deactivate()
    assert renderer.events == seen
