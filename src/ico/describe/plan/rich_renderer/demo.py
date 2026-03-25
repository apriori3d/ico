from collections.abc import Iterable, Iterator

from ico.core.async_stream import IcoAsyncStream
from ico.core.batcher import IcoBatcher
from ico.core.operator import IcoOperatorProtocol, operator
from ico.core.pipeline import IcoPipeline
from ico.core.sink import sink
from ico.core.source import source
from ico.runtime.agent.mp.mp_agent import MPAgent

"""
Plan Renderer Visualization Demo

This demo showcases the ICO plan renderer capabilities by creating
a complex data processing flow with multiple components:

• Source nodes with data size visualization
• Batch processing with batch_size display
• Parallel workers using async streams and MP agents
• Pipeline grouping and nested flow structures
• Named components with rich formatting
• Multi-column table output (Flow, Signature, Name)

Run this script to see how different ICO node types are rendered
with specialized formatting and visual hierarchy.
"""

# ═══ Demo Data Setup ═══
data = [0.5, 1.0, 1.5, 2.0, 0.8, 0.2, 1.0, 1.2, 0.9]

# ═══ 1. Source Components ═══
# These demonstrate source node rendering with data size info


@source()
def indices() -> Iterable[int]:
    """Data indices source - shows Iterable size in renderer."""
    return list(range(len(data)))


@operator()
def fetch_data_item(index: int) -> float:
    """Item fetcher - demonstrates operator node rendering."""
    return data[index]


# ═══ 2. Processing Operators ═══
# These show how simple operators appear in the flow visualization


@operator()
def scale(x: float) -> float:
    """Scale transformation - part of pipeline group."""
    return x * 1.1


@operator()
def shift(x: float) -> float:
    """Shift transformation - part of pipeline group."""
    return x + 0.1


@operator()
def collate_max(batch: Iterator[float]) -> str:
    """Batch aggregation - demonstrates Iterator → str signature."""
    return f"max={max(batch)}"


# ═══ 3. Complex Flow Builder ═══
# This demonstrates pipeline grouping and nested flow structures


def create_augment_flow() -> IcoOperatorProtocol[Iterator[int], str]:
    """Create processing flow showing pipeline and stream grouping."""
    augment = IcoPipeline(
        scale,
        shift,
        name="Data Augment Pipeline",  # Shows named pipeline grouping
    )

    # Stream composition showing complex signature flow
    augment_stream = (fetch_data_item | augment).stream() | collate_max
    augment_stream.name = "Batch Processing Stream"  # Named for clear visualization

    return augment_stream


# ═══ 4. Demo Execution ═══
# Run the visualization demo

if __name__ == "__main__":
    from ico.describe.plan.rich_renderer.plan_renderer import PlanRenderer

    print("\n🎨 ICO Plan Renderer Demo")
    print("=" * 50)
    print("Building complex flow with multiple node types...\n")

    # ── Batch Processing Layer ──
    # Shows batcher node with batch_size parameter
    batcher = IcoBatcher[int](batch_size=3)

    # ── Parallel Processing Layer ──
    # Shows async stream with MP agents (parallel workers)
    workers_pool = IcoAsyncStream(
        lambda: MPAgent(create_augment_flow),
        pool_size=2,
        name="Parallel Workers Pool",  # Named for visualization clarity
    )

    # ── Data Stream Assembly ──
    data_stream = indices | batcher | workers_pool
    data_stream.name = "Data Processing Stream"

    # ── Training Pipeline ──
    # Shows pipeline processing with different signature types
    def optimize(values: str) -> str:
        return f"{values} optimized"

    def log_metrics(values: str) -> str:
        return f"logged: {values}"

    training_pipeline = IcoPipeline(optimize, log_metrics)
    training_stream = training_pipeline.stream()
    training_stream.name = "ML Training Stream"

    # ── Output Sink ──
    # Shows sink node rendering
    @sink()
    def save_results(item: str) -> None:
        """Result persistence - demonstrates sink node visualization."""
        pass

    # ── Complete Flow Assembly ──
    complete_flow = data_stream | training_stream | save_results
    complete_flow.name = "End-to-End ML Pipeline Demo"

    print("📊 Rendering flow visualization...\n")

    # ── Render the Plan ──
    renderer = PlanRenderer()
    renderer.render(complete_flow)

    print("\n✨ Demo Features Shown:")
    print("  • Source nodes with data size info")
    print("  • Batcher with batch_size parameter")
    print("  • Pipeline grouping with named sections")
    print("  • Async streams with parallel workers")
    print("  • MP agent nodes for distributed processing")
    print("  • Complex signature flows (I/C/O types)")
    print("  • Sink nodes for output handling")
    print("  • Multi-column table format (Flow | Signature | Name)")
