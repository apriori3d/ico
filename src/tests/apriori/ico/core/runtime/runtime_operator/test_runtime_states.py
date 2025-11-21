import pytest

from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.runtime.types import IcoRuntimeStateType
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.source import IcoSource
from tests.apriori.ico.core.runtime.runtime_operator.test_utils import (
    StateRecordingRuntime,
)

# ──── Test: normal execution ────


def clousure(x: None) -> None:
    return None


def test_state_transitions_success() -> None:
    recording_runtime = StateRecordingRuntime(clousure)
    # Initially idle
    assert recording_runtime.state is IcoRuntimeStateType.inactive

    # Call operator
    recording_runtime.activate().run().pause().resume().deactivate()

    assert recording_runtime.states == [
        IcoRuntimeStateType.inactive,
        IcoRuntimeStateType.ready,
        IcoRuntimeStateType.running,
        IcoRuntimeStateType.ready,
        IcoRuntimeStateType.paused,
        IcoRuntimeStateType.ready,
        IcoRuntimeStateType.inactive,
    ]


# ──── Test: faulted execution ────


def test_execution_state_transitions_failure() -> None:
    def faulty_fn(item: int) -> int:
        raise RuntimeError("Intentional error for testing.")

    faulty_op = IcoOperator(faulty_fn, name="faulty_op")
    source = IcoSource[int](lambda _: iter([1, 2, 3]), name="data")
    sink = IcoSink[int](lambda items: None if list(items) else None, name="sink")

    flow = source | faulty_op.map() | sink

    runtime = StateRecordingRuntime(flow)

    with pytest.raises(RuntimeError):
        runtime.activate().run()

    runtime.deactivate()

    assert runtime.states == [
        IcoRuntimeStateType.inactive,
        IcoRuntimeStateType.ready,
        IcoRuntimeStateType.running,
        IcoRuntimeStateType.error,
        IcoRuntimeStateType.inactive,
    ]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
