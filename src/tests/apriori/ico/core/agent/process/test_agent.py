from __future__ import annotations

import multiprocessing

from apriori.flow.progress.console import ConsoleProgress
from apriori.ico.core import IcoLifecycleEvent
from apriori.ico.core.agent.process.process_agent import IcoProcessAgent
from apriori.ico.tests.core.runtime.test_utils import (
    EchoOperator,
)

# ─── Agent Basic Execution Test ───


def test_agent_run() -> None:
    # Setup agent with simple echo operator
    agent = IcoProcessAgent(
        name="TestAgent",
        operator_factory=EchoOperator.create,
    )
    agent.progress = ConsoleProgress()

    # Trigger lifecycle prepare (starts worker process)
    agent.on_event(IcoLifecycleEvent.prepare)

    # Send input, receive output
    result = agent(123)

    assert result == 123

    # Trigger lifecycle cleanup (stops worker)
    agent.on_event(IcoLifecycleEvent.cleanup)

    assert agent.worker_process and not agent.worker_process.is_alive()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    test_agent_run()

    # import sys

    # import pytest

    # sys.exit(pytest.main([__file__]))
