"""MPAgent Basic Example - Multiprocessing with ICO Framework"""

import time
from collections.abc import Iterator

from rich.console import Console

from apriori.ico.core.operator import IcoOperator, operator
from apriori.ico.core.process import IcoProcess
from apriori.ico.core.runtime.printer import IcoPrinter
from apriori.ico.core.runtime.progress import (
    IcoProgress,
)
from apriori.ico.core.runtime.runtime import IcoRuntime
from apriori.ico.core.runtime.runtime_wrapper import wrap_runtime
from apriori.ico.core.sink import sink
from apriori.ico.core.source import source
from apriori.ico.runtime.agent.mp.mp_agent import MPAgent
from apriori.ico.tools.printer.rich_printer_tool import RichPrinterTool
from apriori.ico.tools.progress.rich_progress_tool import RichProgressTool


class WorkerFlowFactory:
    """
    Factory for creating worker flow that runs inside each MPAgent process.
    Creates IcoProcess with: input → double → shift → progress, repeated num_iterations times.
    Must be at module level for multiprocessing pickling.
    """

    name: str | None = None
    num_iters: int

    def __init__(self, num_iters: int = 10, name: str | None = None):
        self.name = name
        self.num_iters = num_iters

    def __call__(self) -> IcoOperator[int, int]:
        print = IcoPrinter()

        @operator()
        def double(x: int) -> int:
            time.sleep(0.1)
            res = x * 2
            return res

        @operator()
        def shift(x: int) -> int:
            res = x + 1
            return res

        @wrap_runtime(print)
        @operator()
        def report_debug_info(x: int) -> int:
            if x == 0:
                print(f"Worker {self.name} processing: {x}")
            return x

        progress = IcoProgress[int](total=self.num_iters, name=self.name)

        # Final flow with iterations for demonstration purposes
        return IcoProcess[int](
            report_debug_info | double | shift | progress,
            num_iterations=self.num_iters,
        )


if __name__ == "__main__":
    """
    🎓 MPAgent Lifecycle Demonstration

    This example walks through the complete MPAgent lifecycle, showing how worker
    processes are created, managed, and terminated by the ICO runtime system.

    📋 Demonstration Steps:

    1. **Worker Flow Structure**:
       - Display the computation that will run in each worker process
       - Shows how IcoProcess combines operators with progress tracking

    2. **Flow Architecture**:
       - Source generates data items with delays (simulates real data pipeline)
       - MPAgent workers process items in separate processes
       - Progress tracking monitors work across process boundaries

    3. **Runtime Lifecycle Observation**:
       - **Idle** state: `IcoAgentWorker` exists as a logical node but no processes yet
       - **Ready** state: At activation, worker processes are spawned
       - **Running** state: Execute flow with live progress bars showing distributed work
       - **Idle** state: At deactivation, worker processes are cleaned up and shutdown

    🏗️ Flow Architecture:
        numbers (source) → overall_progress → [worker1 | worker2].stream() → sink

    📊 Progress Tracking:
        - "Overall Progress": main flow progress (10 items)
        - "Worker 1": internal iterations in first process
        - "Worker 2": internal iterations in second process

    This demonstrates how ICO seamlessly coordinates distributed processing
    with progress monitoring across process boundaries.
    """

    # ──────── Print worker flow structure ────────

    # Create worker flow for demonstration and display its structure before execution
    worker_flow = WorkerFlowFactory()()
    print("🔧 Worker Flow Structure (runs in each separate process):")
    worker_flow.describe()

    # ──────── Main flow definition ────────

    total = 5

    @source()
    def numbers() -> Iterator[int]:
        for i in range(total):
            time.sleep(1)  # Simulate slow data source (e.g., file reading, API calls)
            yield i

    @sink()
    def save_result(x: int) -> None:
        pass  # Results discarded for demo purposes

    # 🎯 MPAgent Creation: Workers will be spawned on activation
    mp_process1 = MPAgent[int, int](WorkerFlowFactory(name="Worker 1"))
    mp_process2 = MPAgent[int, int](WorkerFlowFactory(name="Worker 2"))

    # Overall progress tracking for main data flow
    progress = IcoProgress[int](total=total, name="Overall Progress")

    # Complete flow: source → progress → workers sequence → sink
    flow = numbers | (progress | mp_process1 | mp_process2).stream() | save_result
    flow.name = "MPAgent Demonstration Flow"

    # Display initial flow structure
    print("\n📋 Initial Flow Structure:")
    flow.describe()

    # Create runtime with progress monitoring
    print("\n🎯 Creating Runtime with MPAgent nodes...")

    console = Console()
    progress_tool = RichProgressTool(console=console)
    printer_tool = RichPrinterTool(console=console)

    runtime = IcoRuntime(flow, tools=[progress_tool, printer_tool])

    print("\n📊 Runtime Tree BEFORE Activation (MPAgent nodes registered):")
    runtime.describe()

    # 🚀 ACTIVATION: This is when worker processes are actually created!
    print("\n🚀 Activating Runtime - Worker Nodes and Processes Being Created...")
    runtime.activate()

    print("\n📊 Runtime Tree AFTER Activation (Worker processes spawned):")
    runtime.describe()

    print("\n▶️  Starting Execution with Live Progress Monitoring...")
    print("Watch: Multiple progress bars tracking distributed work!")

    # Execute flow with live progress visualization
    runtime.run()

    # Cleanup: Worker processes will be terminated
    print("\n🧹 Deactivating Runtime - Shut down Worker Processes...")
    runtime.deactivate()
    runtime.describe()

    print("\n✅ MPAgent Lifecycle Complete!")
    print("🎓 Key Learnings:")
    print("   • MPAgent nodes register during runtime creation")
    print("   • Worker processes spawn only on activation")
    print("   • Progress tracking works across process boundaries")
    print("   • Runtime automatically manages process lifecycle")
    print("   • describe() shows different views before/after activation")
