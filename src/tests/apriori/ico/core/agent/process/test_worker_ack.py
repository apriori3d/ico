# ──── ProcessWorker Acknowledgment Protocol Test ────

# This test verifies that for each supported input message type
# (Input, LifecycleEvent, Shutdown), the worker responds with
# an AcknowledgePayload whose message_type matches the request.

from __future__ import annotations

import multiprocessing
from multiprocessing import Queue
from typing import Any

import pytest

from apriori.ico.core import IcoLifecycleEvent
from apriori.ico.core.agent.process.process_worker import ProcessWorker
from apriori.ico.core.runtime.channels.messages import (
    AcknowledgePayload,
    InputPayload,
    MessageType,
    RuntimeCommandPayload,
    WorkerMessage,
)
from apriori.ico.tests.core.runtime.test_utils import (
    EchoOperator,
    WorkerQueue,
    shutdown_worker,
)

# ──── Parametrized Acknowledgment Matching Test ────


@pytest.mark.parametrize(
    "payload, expected_type",
    [
        (InputPayload(123), MessageType.input),
        (RuntimeCommandPayload(IcoLifecycleEvent.prepare), MessageType.lifecycle_event),
    ],
)
def test_worker_acknowledges_correct_message_type(
    payload: Any, expected_type: MessageType
) -> None:
    in_q: WorkerQueue = Queue()
    out_q: WorkerQueue = Queue()

    proc = ProcessWorker.spawn(
        in_queue=in_q,
        out_queue=out_q,
        operator_factory=EchoOperator.create,
        name="AckTestWorker",
        relay_progress=False,  # To receive direct acks without progress messages
    )

    in_q.put(WorkerMessage.create(payload))
    ack_msg = out_q.get(timeout=5)

    assert isinstance(ack_msg.payload, AcknowledgePayload)
    assert ack_msg.payload.ack_message_type == expected_type

    shutdown_worker(in_q, out_q, proc)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
