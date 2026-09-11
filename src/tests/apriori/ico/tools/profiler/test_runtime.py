from collections.abc import Iterator

from ico.core.operator import operator
from ico.core.runtime.runtime import IcoRuntime
from ico.tools.profiler.injector import inject_profiler


def test_profiler_node_state() -> None:
    @operator()
    def test_data_provider(_: None) -> Iterator[str]:
        yield "test1"
        yield "test2"

    @operator()
    def check_result(result: Iterator[str]) -> None:
        assert list(result) == ["test1a", "test2a"]

    @operator()
    def add_a(s: str) -> str:
        return s + "a"

    flow = test_data_provider | add_a.stream() | check_result
    profiled_flow = inject_profiler(flow)
    profiled_flow.describe()

    runtime = IcoRuntime(profiled_flow)
    runtime.activate()
    runtime.describe()


if __name__ == "__main__":
    test_profiler_node_state()
