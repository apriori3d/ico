from __future__ import annotations

from collections.abc import Iterator

from rich.text import Text
from rich.tree import Tree

from apriori.ico.core.meta.flow_meta import IcoFlowMeta
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.runtime.operator import IcoRuntimeOperator
from apriori.ico.core.runtime.types import IcoRuntimeStateType
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.source import IcoSource
from apriori.ico.core.stream import IcoStream


def describe(
    flow: IcoFlowMeta,
    *,
    show_states: bool = True,
    show_ico_form: bool = False,
) -> Tree:
    """Render an ICO operator graph (flow) as a rich tree."""
    return _build_node(flow, show_states, show_ico_form)


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
    text = Text(flow_meta.name or flow_meta.node_type.name, style="bold cyan")
    text.append(f" ({flow_meta.node_type.name})", style="dim")

    # Show runtime states if available
    if show_states and flow_meta.runtime_state is not None:
        color = {
            IcoRuntimeStateType.inactive: "grey50",
            IcoRuntimeStateType.ready: "green",
            IcoRuntimeStateType.running: "yellow",
            IcoRuntimeStateType.paused: "grey70",
            IcoRuntimeStateType.error: "red",
        }.get(flow_meta.runtime_state, "white")
        text.append(f" [{flow_meta.runtime_state.name}]", style=color)

    # Show ICO form (signature)
    if show_ico_form:
        text.append(f"  {flow_meta.ico_form.name}", style="dim blue")

    return text


# ──── Example usage ────

Item = float
DataBatch = Iterator[Item]
TrainBatch = float

if __name__ == "__main__":
    from rich.console import Console

    # ──── 1. Define a batched data source ────
    def generate_batches(_: None) -> Iterator[DataBatch]:
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

    def identity_ctx(x: Item) -> Item:
        return x

    def scale_fn(x: Item) -> Item:
        return x * 1.1

    def shift_fn(x: Item) -> Item:
        return x + 0.1

    augment = IcoPipeline(
        context=IcoOperator(identity_ctx),
        body=[
            IcoOperator(scale_fn),
            IcoOperator(shift_fn),
        ],
        output=IcoOperator(identity_ctx),
        name="augment_pipeline",
    )

    def collate_max(batch: DataBatch) -> Item:
        return max(batch)

    collate = IcoOperator(collate_max)

    # Augment each item in the batch, then collate to single value (mimic data loader collate)
    batch_flow = augment.map() | collate
    aug_stream = IcoStream(batch_flow, name="data_stream")

    """
    ──── 3. Compose into a full data stream ────
    • dataset: () → Iterable[Iterable[float]]
    • stream: Iterable[Iterable[float]] → Iterable[float]
    • augment.map(): Iterable[float] → Iterable[float]
    • augment: float → float → float,
    • collate: Iterable[float] → Iterable[float] → float
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

    def identity_context_train(x: TrainBatch) -> TrainBatch:
        return x

    train_iter = IcoPipeline(
        context=IcoOperator(identity_context_train),
        body=[IcoOperator(pow_if_needed)],
        output=IcoOperator(identity_context_train),
        name="train_pipeline",
    )

    train_stream = IcoStream(train_iter, name="train_stream")

    def sink_fn(stream: Iterator[TrainBatch]) -> None:
        for _ in stream:
            pass

    sink = IcoSink(sink_fn)

    # ──── 5. Combine all into the full flow ────
    full_flow = dataset | aug_stream | train_stream | sink

    runtime = IcoRuntimeOperator(full_flow, name="full_flow_runtime")
    runtime.activate()

    # ──── 6. Visualize ────
    flow_meta = IcoFlowMeta.from_operator(runtime)

    console = Console()
    console.rule("[bold blue]ICO Dataflow: Dataset → Stream → Train → Sink")
    console.print(describe(flow_meta, show_states=True, show_ico_form=True))
