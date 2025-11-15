import multiprocessing

import pytest

from apriori.ico.core.agent.process.process_worker import ProcessWorker
from apriori.ico.core.runtime.channels.messages import (
    AcknowledgePayload,
    ExecutionStatePayload,
    InputPayload,
    MessageType,
    OutputPayload,
    WorkerMessage,
)
from apriori.ico.core.runtime.execution import IcoExecutionState
from apriori.ico.tests.core.runtime.test_utils import (
    EchoOperator,
    WorkerQueue,
    shutdown_worker,
)

# ──── ProcessWorker Execution Tests ────


def test_process_worker_execution() -> None:
    in_queue = WorkerQueue()
    out_queue = WorkerQueue()

    proc = ProcessWorker.spawn(
        in_queue=in_queue,
        out_queue=out_queue,
        operator_factory=EchoOperator.create,
        name="TestWorker",
        relay_progress=False,
    )

    in_queue.put(WorkerMessage.create(InputPayload(42)))

    outputs = []
    received = 0

    # Receive two messages: input ack and output
    while received < 4:
        msg = out_queue.get(timeout=10)
        outputs.append(msg.payload)
        received += 1

    assert isinstance(outputs[0], AcknowledgePayload)
    assert isinstance(outputs[1], ExecutionStatePayload)
    assert isinstance(outputs[2], ExecutionStatePayload)
    assert isinstance(outputs[3], OutputPayload)
    assert outputs[0].ack_message_type == MessageType.input
    assert outputs[1].state == IcoExecutionState.running
    assert outputs[2].state == IcoExecutionState.done
    assert outputs[3].output == 42

    shutdown_worker(in_queue, out_queue, proc)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
