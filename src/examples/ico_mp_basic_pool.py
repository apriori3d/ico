"""MPAgent Pool Example - Parallel Processing with IcoAsyncStream"""

import time
from collections.abc import Iterator

from rich.console import Console

from examples.ico_mp_basic import WorkerFlowFactory
from ico import IcoRuntime, MPAgent, RichPrinterTool, RichProgressTool, sink, source
from ico.core.async_stream import IcoAsyncStream
from ico.core.runtime.progress import (
    IcoProgress,
)

if __name__ == "__main__":
    """
    MPAgent Pool with IcoAsyncStream Demonstration

    This example shows how to use IcoAsyncStream for parallel processing with multiple
    MPAgent workers, enabling concurrent execution instead of sequential processing.

    Key Features:

    1. **IcoAsyncStream**: Runs multiple MPAgent workers in parallel (not sequential)
    2. **Worker Differentiation**: Different workers have different iteration counts
    3. **Concurrent Processing**: Each input item is processed by ALL workers simultaneously
    4. **Independent Progress**: Each worker shows its own progress tracking

    Async vs Sequential Processing:
    - **Sequential .stream()**: item1→worker1→worker2, then item2→worker1→worker2
    - **IcoAsyncStream**: item1→(worker1 || worker2), item2→(worker1 || worker2)

    Flow Architecture:
        numbers (source) → overall_progress → IcoAsyncStream([worker1, worker2]) → sink

    Progress Tracking:
        - "Overall Progress": main flow progress (10 items)
        - "Worker 1": 5 iterations per item (faster completion)
        - "Worker 2": 10 iterations per item (slower completion)

    Real-world Use Case:
    Processing same data through different models or algorithms concurrently,
    like ensemble ML inference or A/B testing scenarios.
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
    # Different iteration counts to show concurrent vs sequential behavior
    mp_process1 = MPAgent[int, int](WorkerFlowFactory(name="Worker 1", num_iters=5))
    mp_process2 = MPAgent[int, int](WorkerFlowFactory(name="Worker 2", num_iters=10))

    # 🚀 IcoAsyncStream: Runs workers in parallel (concurrently), not sequentially
    # Each input item goes through BOTH workers at the same time
    async_stream = IcoAsyncStream([mp_process1, mp_process2])

    # Overall progress tracking for main data flow
    progress = IcoProgress[int](total=total, name="Overall Progress")

    # Complete flow: source → progress → async workers → sink
    flow = numbers | progress.stream() | async_stream | save_result
    flow.name = "MPAgent Pool with AsyncStream"

    # Display initial flow structure
    print("\n📋 Initial Flow Structure:")
    flow.describe()

    # Create runtime with progress monitoring
    console = Console()  # Share console across tools for unified output
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
    print("Watch: Each item processed by BOTH workers concurrently!")
    print("Notice: Worker 1 completes faster (5 iters) vs Worker 2 (10 iters)")

    # Execute flow with live progress visualization
    runtime.run()

    # Cleanup: Worker processes will be terminated
    print("\n🧹 Deactivating Runtime - Shut down Worker Processes...")
    runtime.deactivate()
    runtime.describe()

    print("\n✅ MPAgent AsyncStream Complete!")
    print("🎓 Key Learnings:")
    print("   • IcoAsyncStream runs workers in parallel, not sequential")
    print("   • Each input item processed by ALL workers concurrently")
    print("   • Different worker speeds visible in progress bars")
    print("   • Async processing enables ensemble/A-B testing patterns")
    print("   • Progress tracking works across async boundaries")

    """
    🚀 What's Next?

    🧠 Machine Learning Pipelines:
    • ml/ico_linear_regression.ipynb - ICO-based ML workflow development
    • ml/cv/cifar/ico_cifar_complete_flow.ipynb - Complete CV pipeline replacing PyTorch DataLoader
    • ml/cv/cifar/ico_cifar_complete_flow_mp.py - Production-ready CV pipeline with multiprocessing workers
    """
