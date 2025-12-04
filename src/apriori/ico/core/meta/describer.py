from __future__ import annotations

from collections.abc import Iterator

from rich.text import Text
from rich.tree import Tree

from apriori.ico.core.meta.flow_meta import IcoFlowMeta
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.runtime.contour import IcoRuntimeContour
from apriori.ico.core.runtime.node import IcoRuntimeState
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.source import IcoSource
from apriori.ico.core.stream import IcoStream


def describe(
    flow_meta: IcoFlowMeta,
    *,
    show_ico_form: bool = False,
) -> None:
    """Render an ICO operator graph (flow) as a rich tree."""
    from rich.console import Console

    tree = _build_node(flow_meta, True, show_ico_form)

    console = Console()
    console.rule(f"[bold blue]{flow_meta.name}", style="dim blue")
    console.print(tree)


# ─── Recursive builder ───


def _build_node(
    flow: IcoFlowMeta,
    show_states: bool,
    show_ico_form: bool,
) -> Tree:
    label = _format_label(flow, show_states, show_ico_form)
    node = Tree(label)
    for child in flow.children:
        node.add(_build_node(child, show_states, show_ico_form))
    return node


# ─── Label formatting ───


def _format_label(
    flow_meta: IcoFlowMeta,
    show_states: bool,
    show_ico_form: bool,
) -> Text:
    text = Text(flow_meta.name, style="bold cyan")
    text.append(f" ({flow_meta.node_type.name})", style="dim")

    # Show runtime states if available
    if show_states and flow_meta.runtime_state is not None:
        color = {
            IcoRuntimeState.inactive: "blue",
            IcoRuntimeState.ready: "green",
            IcoRuntimeState.running: "yellow",
            IcoRuntimeState.paused: "grey70",
            IcoRuntimeState.fault: "red",
        }.get(flow_meta.runtime_state, "white")
        text.append(f" [{flow_meta.runtime_state.name}]", style=color)

    # Show ICO form (signature)
    if show_ico_form and flow_meta.ico_form is not None:
        text.append(f"  {flow_meta.ico_form.name}", style="gray70")

    return text


# ──── Example usage ────

Item = float
DataBatch = Iterator[Item]
TrainBatch = float

if __name__ == "__main__":
    # ──── 1. Define a batched data source ────
    def generate_batches() -> Iterator[DataBatch]:
        """Simulate dataset batches: () → Iterable[list[float]]"""
        data = [
            [0.5, 1.0, 1.5],
            [2.0, 0.8, 0.2],
            [1.0, 1.2, 0.9],
        ]
        for batch in data:
            yield iter(batch)

    dataset = IcoSource(generate_batches, name="dataset")

    # ──── 2. Define augmentation & collation pipelines per item ────

    def scale_fn(x: Item) -> Item:
        return x * 1.1

    def shift_fn(x: Item) -> Item:
        return x + 0.1

    augment = IcoPipeline(
        scale_fn,
        shift_fn,
        name="augment_pipeline",
    )

    def collate_max(batch: DataBatch) -> Item:
        return max(batch)

    collate = IcoOperator(collate_max)

    # Augment each item in the batch, then collate to single value (mimic data loader collate)
    batch_flow = augment.iterate() | collate
    aug_stream = IcoStream(batch_flow, name="data_stream")

    """
    ──── 3. Compose into a full data stream ────
    • dataset: () → Iterable[Iterable[float]]
    • stream: Iterable[Iterable[float]] → Iterable[float]
    • augment.map(): Iterable[float] → Iterable[float]
    • augment: float → float,
    • collate: Iterable[float] → float
    • dataflow: () → Iterable[float]
    """
    dataflow = dataset | aug_stream

    # ──── 4. Define training pipeline ────
    # We collate batch into single float via max, so input into training stream is Iterable[float],
    # without batch dimension. In real scenarios, item could be tensors with batch dim.
    # Each batch item passes through a process that transforms floats ≤ 1 via pow(2)
    # train_step: float -> float, ICO Form: I → O
    # train_process: float -> float, ICO Form: C → C
    # I → C → O = Iterable[float] → Iterable[float] → Iterable[float]

    def pow_if_needed(values: TrainBatch) -> TrainBatch:
        return values**2 if values <= 1.0 else values

    train_iter = IcoPipeline(
        pow_if_needed,
        name="train_pipeline",
    )

    train_stream = IcoStream(train_iter, name="train_stream")

    def sink_fn(item: TrainBatch) -> None:
        pass

    sink = IcoSink(sink_fn)

    # ──── 5. Combine all into the full flow ────
    full_flow = dataset | aug_stream | train_stream | sink

    runtime = IcoRuntimeContour(full_flow, name="full_flow_runtime")
    runtime.activate()

    # ──── 6. Visualize ────
    describe(full_flow)
