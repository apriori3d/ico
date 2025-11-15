import multiprocessing

import pytest

from apriori.ico.core import (
    IcoLifecycleEvent,
)
from apriori.ico.core.agent.process.process_worker import ProcessWorker
from apriori.ico.core.runtime.channels.messages import (
    AcknowledgePayload,
    InputPayload,
    MessageType,
    OutputPayload,
    RuntimeCommandPayload,
    WorkerMessage,
)
from apriori.ico.tests.core.runtime.test_utils import (
    EchoOperator,
    NestedRecordingOperator,
    WorkerQueue,
    shutdown_worker,
)


def test_process_worker_lifecycle_events() -> None:
    in_q: WorkerQueue = WorkerQueue()
    out_q: WorkerQueue = WorkerQueue()

    proc = ProcessWorker.spawn(
        in_queue=in_q,
        out_queue=out_q,
        operator_factory=EchoOperator.create,
        name="TestWorker",
        relay_progress=False,
    )

    # Events to test
    all_events = list(IcoLifecycleEvent)

    # Send events and verify acknowledgments
    for event in all_events:
        in_q.put(WorkerMessage.create(RuntimeCommandPayload(event)))
        msg = out_q.get(timeout=5)

        assert isinstance(msg.payload, AcknowledgePayload)
        assert msg.payload.ack_message_type == MessageType.lifecycle_event

    shutdown_worker(in_q, out_q, proc)


# ──── ProcessWorker Lifecycle Event Deep Forwarding Test ────


def test_process_worker_lifecycle_event_forwarding() -> None:
    in_q: WorkerQueue = WorkerQueue()
    out_q: WorkerQueue = WorkerQueue()

    proc = ProcessWorker.spawn(
        in_queue=in_q,
        out_queue=out_q,
        operator_factory=NestedRecordingOperator.create,
        name="LifecycleRecorder",
        relay_progress=False,
    )

    all_events = list(IcoLifecycleEvent)

    # Events to send (excluding cleanup for now)
    for event in all_events:
        in_q.put(WorkerMessage.create(RuntimeCommandPayload(event)))
        msg = out_q.get(timeout=5)
        assert isinstance(msg.payload, AcknowledgePayload)

    # Collect events received by the operator via output
    in_q.put(WorkerMessage.create(InputPayload(0)))
    while True:
        msg = out_q.get(timeout=5)
        if isinstance(msg.payload, OutputPayload):
            break

    # Verify that all events were forwarded to the operator
    assert msg.payload.output == all_events

    shutdown_worker(in_q, out_q, proc)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
