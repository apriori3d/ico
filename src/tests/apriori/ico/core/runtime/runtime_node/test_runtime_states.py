import pytest

from ico.core.operator import IcoOperator
from ico.core.runtime.state import (
    FaultState,
    IdleState,
    ReadyState,
    RunningState,
)
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from tests.apriori.ico.core.runtime.runtime_node.test_utils import RecordingRuntime

# ──── Test: normal execution ────


def clousure_fn(x: None) -> None:
    return None


def test_state_transitions_success() -> None:
    clousure = IcoOperator(clousure_fn, name="no_op")
    recording_runtime = RecordingRuntime(clousure)

    # Initially idle
    assert isinstance(recording_runtime.state, IdleState)

    # Call operator
    recording_runtime.activate().run().deactivate()

    assert recording_runtime.recorded_states == [
        IdleState,
        ReadyState,
        RunningState,
        ReadyState,
        IdleState,
    ]


# ──── Test: faulted execution ────


def test_execution_state_transitions_failure() -> None:
    def faulty_fn(item: int) -> int:
        raise RuntimeError("Intentional error for testing.")

    faulty_op = IcoOperator(faulty_fn, name="faulty_op")
    source = IcoSource[int](lambda: iter([1, 2, 3]), name="data")
    sink = IcoSink[int](lambda item: None, name="sink")

    flow = source | faulty_op.stream() | sink

    runtime = RecordingRuntime(flow)

    with pytest.raises(RuntimeError):
        runtime.activate().run()

    runtime.deactivate()

    assert runtime.recorded_states == [
        IdleState,
        ReadyState,
        RunningState,
        FaultState,
        IdleState,
    ]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))


# ──── Test: items do not change states ────


def test_items_leave_monitor_states_alone() -> None:
    from ico.core.runtime.progress import IcoProgress
    from ico.core.runtime.runtime import IcoRuntime

    progress: IcoProgress[int] = IcoProgress(total=3)
    seen: list[type] = []

    def look(item: int) -> int:
        seen.append(type(progress.state))
        return item

    flow = (
        IcoSource[int](lambda: iter([1, 2, 3]))
        | (progress | IcoOperator(look)).stream()
        | IcoSink[int](lambda item: None)
    )
    runtime = IcoRuntime(flow).activate()
    runtime.run()

    assert seen == [ReadyState] * 3  # not Running while an item passes
    after_run = type(progress.state)
    runtime.deactivate()
    assert after_run is ReadyState
    assert isinstance(progress.state, IdleState)
