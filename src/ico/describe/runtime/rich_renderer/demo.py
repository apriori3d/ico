from collections.abc import Iterable, Iterator

from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.batcher import IcoBatcher
from apriori.ico.core.operator import IcoOperator, operator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.runtime.progress import IcoProgress
from apriori.ico.core.runtime.runtime import IcoRuntime
from apriori.ico.core.sink import sink
from apriori.ico.core.source import source
from apriori.ico.describe.runtime.rich_renderer.tree_renderer import RuntimeTreeRenderer
from apriori.ico.runtime.agent.mp.mp_agent import MPAgent

"""
Runtime Tree Renderer Visualization Demo

This demo showcases the ICO runtime renderer capabilities by creating
a complex flow and demonstrating real-time runtime state visualization:

• Runtime lifecycle states (Ready → Running → Complete)
• IcoProgress nodes with state tracking
• MP agents with worker state visualization
• Runtime tree hierarchy with expand_agents option
• State-colored display (green/yellow/red indicators)
• Multi-column runtime table (Tree | State | Name)
• Dynamic state updates during execution

Run this script to see how runtime nodes change states during
execution and how different runtime components are visualized.
"""

# ═══ Demo Data Setup ═══
data = [0.5, 1.0, 1.5, 2.0, 0.8, 0.2, 1.0, 1.2, 0.9]

# ═══ 1. Source Components with Runtime Tracking ═══
# These demonstrate source nodes in runtime context


@source()
def indices() -> Iterable[int]:
    """Index generator - shows source node runtime states."""
    return list(range(len(data)))


@operator()
def fetch_data_item(index: int) -> float:
    """Data fetcher - demonstrates operator runtime states."""
    return data[index]


# ═══ 2. Processing with Runtime Progress ═══
# These show runtime state changes and progress tracking


@operator()
def scale(x: float) -> float:
    """Scale operator - shows processing state changes."""
    return x * 1.1


@operator()
def shift(x: float) -> float:
    """Shift operator - demonstrates pipeline runtime flow."""
    return x + 0.1


@operator()
def collate_max(batch: Iterator[float]) -> str:
    """Batch collator - shows Iterator→str runtime signature."""
    return f"max={max(batch)}"


# ═══ 3. Runtime Pipeline with Progress Tracking ═══
# This demonstrates runtime progress nodes and state visualization


def create_augment_flow() -> IcoOperator[Iterator[int], str]:
    """Create flow with IcoProgress for runtime state demonstration."""
    augment = IcoPipeline(
        scale,
        shift,
        IcoProgress[float](name="Processing Progress", total=9),  # Shows progress node
        name="Data Processing Pipeline",
    )

    # Stream with runtime state tracking
    augment_stream = (fetch_data_item | augment).stream() | collate_max
    augment_stream.name = "Runtime Processing Stream"

    return augment_stream


# ═══ 4. Runtime State Visualization Demo ═══
# Run the runtime tree renderer demonstration

if __name__ == "__main__":
    print("\n🎭 ICO Runtime Tree Renderer Demo")
    print("=" * 50)
    print("Building runtime flow with state tracking...\n")

    # ── Batch Processing Layer ──
    # Shows batcher runtime states
    batcher = IcoBatcher[int](batch_size=3)

    # ── MP Agents Layer ──
    # Shows agent worker runtime hierarchy
    agents = [MPAgent(create_augment_flow) for _ in range(2)]
    workers_pool = IcoAsyncStream(
        agents,
        name="MP Workers Pool",  # Named for runtime visualization
    )

    # ── Data Stream Assembly ──
    data_stream = indices | batcher | workers_pool
    data_stream.name = "Runtime Data Stream"

    # ── Training Pipeline with States ──
    # Shows pipeline runtime state progression
    def optimize(values: str) -> str:
        return f"{values} optimized"

    def log_metrics(values: str) -> str:
        return f"logged: {values}"

    training_pipeline = IcoPipeline(optimize, log_metrics)
    training_stream = training_pipeline.stream()
    training_stream.name = "ML Training Runtime Stream"

    # ── Output Sink with Runtime States ──
    @sink()
    def save_results(item: str) -> None:
        """Results saver - demonstrates sink runtime states."""
        print("💾 Saved result:", item)

    # ── Complete Runtime Flow ──
    complete_flow = data_stream | training_stream | save_results
    complete_flow.name = "End-to-End Runtime Demo"

    # ── Initialize Runtime Renderer ──
    renderer = RuntimeTreeRenderer()
    runtime = IcoRuntime(complete_flow, name="Demo Runtime System")
    # ═══ Runtime State Demonstration Sequence ═══

    print("📋 Step 1: Initial Runtime State (Ready)")
    print("-" * 40)
    runtime.activate()
    renderer.render(runtime)

    print("\n⚡ Step 2: During Execution (Running States)")
    print("-" * 40)
    runtime.run()
    renderer.render(runtime)

    print("\n🔄 Step 3: After Agent Deactivation (Mixed States)")
    print("-" * 40)
    agents[1].deactivate()
    renderer.render(runtime)

    print("\n✅ Step 4: Complete Shutdown (Final States)")
    print("-" * 40)
    runtime.deactivate()
    renderer.render(runtime)

    print("\n🎯 Demo Features Shown:")
    print("  • Runtime state progression (Ready → Running → Complete)")
    print("  • IcoProgress nodes with progress tracking")
    print("  • MP agent runtime hierarchies and worker states")
    print("  • State-colored visualization (green/yellow/red)")
    print("  • Runtime tree expansion with agent details")
    print("  • Dynamic state updates during execution")
    print("  • Multi-column runtime format (Tree | State | Name)")
    print("\n✨ Runtime visualization complete!")
