import io
from collections.abc import Iterable, Iterator
from typing import Any

from rich.console import Console

from ico.core.operator import IcoOperator
from ico.core.runtime.progress import IcoProgress
from ico.core.runtime.runtime import IcoRuntime
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.tools.progress.rich_progress_tool import RichProgressTool


def sizes() -> Iterable[int]:
    return [3, 2]


def scenes(n: int) -> Iterator[int]:
    return iter(list(range(n)))


def collect_all(items: Iterator[int]) -> list[int]:
    return list(items)


def ignore(x: Any) -> None:
    pass


def test_rich_progress_takes_the_stream_total() -> None:
    progress: IcoProgress[int] = IcoProgress(name="scenes")
    body: Any = IcoOperator(scenes) | IcoStream(progress) | IcoOperator(collect_all)
    tool = RichProgressTool(console=Console(file=io.StringIO()))
    flow = IcoSource(sizes) | IcoStream(body) | IcoSink(ignore)

    runtime = IcoRuntime(flow, tools=[tool]).activate().run()
    try:
        assert tool.progress is not None
        [task] = tool.progress.tasks
        assert (task.total, task.completed) == (2, 2)  # the last video: 2 scenes
    finally:
        runtime.deactivate()
