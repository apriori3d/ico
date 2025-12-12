from __future__ import annotations

from collections.abc import Iterator
from typing import overload

from pyparsing import Enum
from rich.console import Console
from rich.text import Text
from rich.tree import Tree

from apriori.ico.core.meta.collector import collect_meta, collect_runtime_meta
from apriori.ico.core.meta.node_meta import IcoNodeMeta, IcoRuntimeNodeMeta
from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.runtime.contour import IcoRuntimeContour
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.state import (
    FaultState,
    IcoRuntimeState,
    ReadyState,
    RunningState,
)
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.source import IcoSource
from apriori.ico.core.stream import IcoStream

# ────────────────────────────────────────────────
# Describer API
# ────────────────────────────────────────────────


@overload
def describe(
    node: IcoNode,
    *,
    include_runtime: bool = False,
    show_ico_form: bool = True,
    console: Console | None = None,
) -> None: ...


@overload
def describe(
    node: IcoRuntimeNode,
    *,
    console: Console | None = None,
) -> None: ...


def describe(
    node: IcoNode | IcoRuntimeNode,
    *,
    show_ico_form: bool = True,
    include_runtime: bool = False,
    console: Console | None = None,
) -> None:
    """Render an ICO operator graph (flow) as a rich tree."""
    if isinstance(node, IcoNode):
        meta = collect_meta(node, include_runtime=include_runtime)
        tree = _build_meta_tree(meta, show_ico_form=show_ico_form)
    else:
        meta = collect_runtime_meta(node)
        tree = _build_runtime_meta_tree(meta)

    console = console or Console()
    console.rule(f"[bold blue]{meta.name}", style="dim blue")
    console.print(tree)


# ────────────────────────────────────────────────
# Recursive tree building
# ────────────────────────────────────────────────


def _build_meta_tree(
    flow: IcoNodeMeta,
    *,
    show_ico_form: bool,
) -> Tree:
    label = _construct_label(flow, show_ico_form)
    node = Tree(label)

    for child in flow.children:
        node.add(_build_meta_tree(child, show_ico_form=show_ico_form))

    if flow.runtime is not None:
        for child in flow.runtime.children:
            node.add(_build_runtime_meta_tree(child))

    return node


def _build_runtime_meta_tree(flow: IcoRuntimeNodeMeta) -> Tree:
    label = _construct_runtime_label(flow)
    node = Tree(label)

    for child in flow.children:
        node.add(_build_runtime_meta_tree(child))

    return node


# ────────────────────────────────────────────────
# Label construction
# ────────────────────────────────────────────────


def _construct_label(node_meta: IcoNodeMeta, show_ico_form: bool) -> Text:
    # Format name with origin-based style
    text = _get_expanded_name_text(node_meta)

    # Show type name
    text.append(f" {node_meta.type_name}", style="dim")

    if node_meta.runtime is not None:
        text.append(f"/{node_meta.runtime.type_name}", style="dim")

    # Show ICO form (signature)
    if show_ico_form and node_meta.ico_form is not None:
        text.append(f"  {node_meta.ico_form.name}", style="cyan")

    return text


def _construct_runtime_label(node_meta: IcoRuntimeNodeMeta) -> Text:
    # Format name with origin-based style
    text = _get_runtime_name_text(node_meta)

    # Show type name
    text.append(f" {node_meta.type_name}", style="dim")

    # Show runtime states if available
    color = _get_state_color(node_meta.state)
    text.append(f" <{node_meta.state}>", style=color)

    return text


# ────────────────────────────────────────────────
# Label Construction/Expanding depending on node type
# ────────────────────────────────────────────────


def _get_expanded_name_text(meta: IcoNodeMeta) -> Text:
    if meta.type_name == "Runtime Wrapper":
        assert meta.runtime is not None

        state_style = _get_state_color(meta.runtime.state)
        name = Text("Runtime(", style=state_style)

        assert len(meta.children) == 1
        # Computation flow perspective of runtime wrapper - we able to use child operator to color name
        wrapped_op_name = _get_expanded_name_text(meta.children[0])
        name.append(wrapped_op_name)

        name.append(")", style=state_style)

        return name

    if meta.type_name == "Iterator":
        name = Text("Iterator(", style=NameStyle.keyword.value)

        assert len(meta.children) == 1
        iterable_name = _get_expanded_name_text(meta.children[0])
        name.append(iterable_name)

        name.append(")", style=NameStyle.keyword.value)
        return name

    if meta.type_name == "Chain":
        assert len(meta.children) == 2
        left_name = _get_expanded_name_text(meta.children[0])
        right_name = _get_expanded_name_text(meta.children[1])

        name = Text("", style=NameStyle.keyword.value)
        name.append(left_name)
        name.append(" | ", style=NameStyle.keyword.value)
        name.append(right_name)
        return name

    return _get_name_text(meta.name, meta.name_origin)


def _get_runtime_name_text(meta: IcoRuntimeNodeMeta) -> Text:
    return _get_name_text(meta.name, meta.name_origin)


def _get_name_text(name: str, name_origin: str) -> Text:
    return Text(name, style=name_origin_scheme.get(name_origin, "black"))


# ────────────────────────────────────────────────
# Label Colorization
# ────────────────────────────────────────────────

state_schemes = [
    (ReadyState, "green"),
    (RunningState, "yellow"),
    (FaultState, "red"),
    (IcoRuntimeState, "gray70"),
]


def _get_state_color(state: IcoRuntimeState) -> str:
    for state_type, color in state_schemes:
        if isinstance(state, state_type):  # type: ignore
            return color
    return "black"


class NameStyle(Enum):
    fn = "#A67F59"
    # class_ = "#9CDCFE"
    type = "#569CD6"
    class_ = "#007ACC"
    keyword = "#E12EE1"
    dimmed = "#C8C8C8"
    meta = "#4FC1FF"


name_origin_scheme = {
    "fn": NameStyle.fn.value,
    "class": NameStyle.class_.value,
    "user": NameStyle.class_.value,
}


# ────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────

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

    train_iter = IcoPipeline(pow_if_needed, name="train_pipeline")

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
