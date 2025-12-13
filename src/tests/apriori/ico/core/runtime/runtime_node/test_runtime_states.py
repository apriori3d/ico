import pytest

from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.runtime.node import IcoRuntimeStateOld
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.source import IcoSource
from tests.apriori.ico.core.runtime.runtime_node.test_utils import RecordingContour

# ──── Test: normal execution ────


def clousure_fn(x: None) -> None:
    return None


def test_state_transitions_success() -> None:
    clousure = IcoOperator(clousure_fn, name="no_op")
    recording_runtime = RecordingContour(clousure)

    # Initially idle
    assert recording_runtime.state is IcoRuntimeStateOld.inactive

    # Call operator
    recording_runtime.activate().run().pause().resume().deactivate()

    assert recording_runtime.recorded_states == [
        IcoRuntimeStateOld.inactive,
        IcoRuntimeStateOld.ready,
        IcoRuntimeStateOld.running,
        IcoRuntimeStateOld.ready,
        IcoRuntimeStateOld.paused,
        IcoRuntimeStateOld.ready,
        IcoRuntimeStateOld.inactive,
    ]


# ──── Test: faulted execution ────


def test_execution_state_transitions_failure() -> None:
    def faulty_fn(item: int) -> int:
        raise RuntimeError("Intentional error for testing.")

    faulty_op = IcoOperator(faulty_fn, name="faulty_op")
    source = IcoSource[int](lambda: iter([1, 2, 3]), name="data")
    sink = IcoSink[int](lambda items: None if list(items) else None, name="sink")

    flow = source | faulty_op.stream() | sink

    runtime = RecordingContour(flow)

    with pytest.raises(RuntimeError):
        runtime.activate().run()

    runtime.deactivate()

    assert runtime.recorded_states == [
        IcoRuntimeStateOld.inactive,
        IcoRuntimeStateOld.ready,
        IcoRuntimeStateOld.running,
        IcoRuntimeStateOld.fault,
        IcoRuntimeStateOld.inactive,
    ]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
