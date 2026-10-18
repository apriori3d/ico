"""Queries on the runtime bus: a command with a request number, answers
collected for the one who asked; describe of a runtime uses one."""

import io
from collections.abc import Iterator
from typing import Any

import pytest
from rich.console import Console

from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.core.runtime.command import IcoRuntimeCommand
from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.runtime.node import IcoRuntimeNode
from ico.core.runtime.progress import IcoProgress
from ico.core.runtime.runtime import IcoRuntime
from ico.core.runtime.state import IcoStateEvent, IcoStateRequestCommand
from ico.core.runtime.tool import IcoRuntimeTool
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.describe.runtime.rich_renderer.tree_renderer import RuntimeTreeRenderer
from ico.runtime.worker.mp.mp_worker import MPWorker

# Module level: spawned runner processes import this module.


def numbers() -> Iterator[int]:
    yield from range(3)


def ignore(x: Any) -> None:
    pass


def double(x: int) -> int:
    return x * 2


def worker_flow() -> IcoOperatorProtocol[int, int]:
    return IcoProgress[int](total=1, name="inner") | IcoOperator(double)


class Spy(IcoRuntimeTool):
    """Records the events forwarded to it."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[IcoRuntimeEvent] = []

    def on_forward_event(self, event: IcoRuntimeEvent) -> None:
        self.events.append(event)


def make_runtime(name: str) -> tuple[IcoRuntime, IcoProgress[int], Spy]:
    progress: IcoProgress[int] = IcoProgress(total=3, name=name)
    spy = Spy()
    flow = IcoSource(numbers) | progress.stream() | IcoSink(ignore)
    return IcoRuntime(flow, tools=[spy]), progress, spy


def test_query_collects_the_answers_of_every_node() -> None:
    runtime, _, spy = make_runtime("first")
    runtime.activate()

    answers = runtime.query(IcoStateRequestCommand.create())

    nodes = list(runtime.iterate_nodes())
    assert len(answers) == len(nodes)
    assert all(isinstance(a, IcoStateEvent) for a in answers)
    assert len({a.reply_to for a in answers}) == 1
    assert not any(isinstance(e, IcoStateEvent) for e in spy.events)  # not tools
    runtime.deactivate()


def test_query_reaches_runners_in_worker_processes() -> None:
    worker = MPWorker(worker_flow)
    runtime = IcoRuntime(IcoSource(numbers) | IcoStream(worker) | IcoSink(ignore))
    runtime.activate()
    try:
        answers = runtime.query(IcoStateRequestCommand.create())
    finally:
        runtime.deactivate()

    local = len(list(runtime.iterate_nodes()))
    assert len(answers) > local  # the runner and its progress node answered
    deepest = max(answers, key=lambda a: len(a.trace.path_index))
    assert len(deepest.trace.path_index) > 2


def test_a_failed_query_leaves_nothing_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, progress, _ = make_runtime("broken")

    def fail(command: IcoRuntimeCommand) -> None:
        if isinstance(command, IcoStateRequestCommand):
            raise RuntimeError("no answer")

    monkeypatch.setattr(progress, "on_command", fail)
    with pytest.raises(RuntimeError, match="no answer"):
        runtime.query(IcoStateRequestCommand.create())
    monkeypatch.undo()

    # The next query works as usual and gets every answer.
    answers = runtime.query(IcoStateRequestCommand.create())
    assert len(answers) == len(list(runtime.iterate_nodes()))


def test_runtimes_do_not_share_answers_or_listeners() -> None:
    first, _, first_spy = make_runtime("first")
    second, _, second_spy = make_runtime("second")

    first.query(IcoStateRequestCommand.create())
    second.activate().run().deactivate()

    assert not any(isinstance(e, IcoStateEvent) for e in second_spy.events)
    assert not any(isinstance(e, IcoStateEvent) for e in first_spy.events)
    assert not hasattr(IcoRuntime, "event_listeners")


def test_describe_asks_for_states_and_is_not_a_tool() -> None:
    runtime, _, _ = make_runtime("described")
    runtime.activate()
    tools = list(runtime.toolbox.tools)
    out = io.StringIO()
    renderer = RuntimeTreeRenderer(console=Console(file=out, width=160))

    renderer.render(runtime)
    renderer.render(runtime)  # twice: nothing is left subscribed

    assert not isinstance(renderer, IcoRuntimeNode)
    assert runtime.toolbox.tools == tools
    assert "Ready" in out.getvalue()
    runtime.deactivate()
