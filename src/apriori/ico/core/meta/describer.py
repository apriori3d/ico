from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, TypeAlias, overload

from pyparsing import Enum
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from apriori.ico.core.meta.inspect.collector import collect_meta, collect_runtime_meta
from apriori.ico.core.meta.meta import IcoNodeMeta, IcoRuntimeNodeMeta
from apriori.ico.core.meta.utils import format_ico_type
from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import operator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.runtime.contour import IcoRuntimeContour
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.state import (
    FaultState,
    IcoRuntimeState,
    ReadyState,
    RunningState,
)
from apriori.ico.core.sink import sink
from apriori.ico.core.source import source

# ────────────────────────────────────────────────
# Describer API
# ────────────────────────────────────────────────

DiscribeFormat: TypeAlias = Literal["Tree", "Plan"]
SignatureFormat: TypeAlias = Literal["Full", "Input", "Output"]


@overload
def describe(
    node: IcoNode,
    *,
    format: DiscribeFormat | None = None,
    signature_format: SignatureFormat | None = None,
    include_runtime: bool = False,
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
    format: DiscribeFormat | None = None,
    signature_format: SignatureFormat | None = None,
    include_runtime: bool = False,
    console: Console | None = None,
) -> None:
    """Render an ICO operator graph (flow) as a rich tree."""

    console = console or Console()

    if isinstance(node, IcoNode):
        meta = collect_meta(node, include_runtime=include_runtime)
        format = format or "Plan"
        signature_format = signature_format or "Full"

        if format == "Tree":
            tree = _build_operators_tree(meta, signature_format=signature_format)

            console.rule(f"[bold blue]Operators tree: {meta.name}", style="dim blue")
            console.print(tree)
        else:
            # Build Flow table
            table = Table(show_lines=False, show_edge=False, box=None)
            table.add_column("Name", no_wrap=True)
            table.add_column("Type", no_wrap=True)
            table.add_column("Signature", no_wrap=True)
            if include_runtime:
                table.add_column("State", no_wrap=True)

            _build_execution_flow(
                meta,
                signature_format=signature_format,
                table=table,
                indent=Text(""),
            )

            console.rule(f"[bold blue]Execution plan: {meta.name}", style="dim blue")
            console.print(table)

    else:
        meta = collect_runtime_meta(node)
        tree = _build_runtime_meta_tree(meta)
        console.rule(f"[bold blue]{meta.name}", style="dim blue")
        console.print(tree)


# ────────────────────────────────────────────────
# Computation Flow, Display format: Tree
# ────────────────────────────────────────────────


def _build_operators_tree(
    flow: IcoNodeMeta,
    *,
    signature_format: SignatureFormat,
) -> Tree:
    label = _construct_tree_label(flow, data_format=signature_format)
    node = Tree(label)

    for child in flow.children:
        node.add(_build_operators_tree(child, signature_format=signature_format))

    if flow.runtime is not None:
        for child in flow.runtime.children:
            node.add(_build_runtime_meta_tree(child))

    return node


def _construct_tree_label(node_meta: IcoNodeMeta, data_format: SignatureFormat) -> Text:
    # Format name with origin-based style
    # text = _get_expanded_name_text(node_meta)
    text = _get_name_label(node_meta)
    text.append(" ")

    # Show type name
    text.append(_get_type_label(node_meta))
    text.append(" ")

    if node_meta.runtime is not None:
        text.append(_get_state_label(node_meta.runtime))
        text.append(" ")

    # Show Signature
    text.append(_get_signature_label(node_meta, data_format))

    return text


# ────────────────────────────────────────────────
# Computation Flow, Display format: Flow
# ────────────────────────────────────────────────

FLATTEN_TYPES = ("Chain", "Pipeline", "Runtime Wrapper")
# "Iterator"


def _build_execution_flow(
    flow: IcoNodeMeta,
    *,
    signature_format: SignatureFormat,
    table: Table,
    indent: Text,
) -> None:
    """Deep-first left-to-right traversal of flow structure.
    Chain nodes are skipped to flatten the flow representation.
    Structural nodes like Iterator produce nested level of sub-flows
    """

    if flow.type_name in FLATTEN_TYPES:
        # Flatten nodes and continue with left-to-right children
        for child in flow.children:
            _build_execution_flow(
                child,
                signature_format=signature_format,
                table=table,
                indent=indent.copy(),
            )
        return

    # Create a new flow node
    if flow.type_name == "Stream":
        _build_stream_execution_flow(
            flow,
            signature_format=signature_format,
            table=table,
            indent=indent.copy(),
        )
        return

    columns = _construct_flow_label(
        flow,
        indent=indent,
        data_format=signature_format,
    )
    table.add_row(*[c for c in columns if c is not None])

    # Structural nodes - produce sub-flows for current node,
    # overwise - just a leaf node
    if len(flow.children) > 0:
        for child in flow.children:
            _build_execution_flow(
                child,
                signature_format=signature_format,
                table=table,
                indent=indent.copy().append(Text("    ")),
            )


def _build_stream_execution_flow(
    stream_meta: IcoNodeMeta,
    *,
    signature_format: SignatureFormat,
    table: Table,
    indent: Text,
) -> None:
    name_col = _get_name_label(stream_meta)
    name_col = indent.copy().append(name_col)
    type_col = _get_type_label(stream_meta)
    signature = _get_signature_label(stream_meta, signature_format)

    table.add_row(name_col, type_col, signature, None)

    itrerate_over_text = Text("iterate", style=NameStyle.keyword.value)
    iterate_name = indent.copy().append(Text("╭─ ")).append(itrerate_over_text)
    iterate_type = _get_signature_label(stream_meta, "Input")

    table.add_row(iterate_name, None, iterate_type, None)

    for child in stream_meta.children:
        _build_execution_flow(
            child,
            signature_format=signature_format,
            table=table,
            indent=indent.copy().append(Text("│   ")),
        )

    yield_text = indent.copy().append(Text("╰─ "))
    yield_text.append(Text("yield", style=NameStyle.keyword.value))
    yield_type = _get_signature_label(stream_meta, "Output")

    table.add_row(yield_text, None, yield_type, None)


def _construct_flow_label(
    node_meta: IcoNodeMeta,
    indent: Text,
    data_format: SignatureFormat,
) -> tuple[Text, Text, Text, Text | None]:
    # Format name with origin-based style
    # text = _get_expanded_name_text(node_meta)

    name_col = _get_name_label(node_meta)
    name_col = indent.append(name_col)

    # Show type name
    type_col = _get_type_label(node_meta)

    # Show Signature
    signature_col = _get_signature_label(node_meta, data_format)

    if node_meta.runtime is not None:
        state_col = _get_state_label(node_meta.runtime)
    else:
        state_col = None

    return name_col, type_col, signature_col, state_col


# ────── Compute Node Label builder ──────


def _get_name_label(node_meta: IcoNodeMeta) -> Text:
    return _get_name_text(f"{node_meta.name} ", node_meta.name_origin)


def _get_type_label(node_meta: IcoNodeMeta) -> Text:
    label = node_meta.type_name
    if node_meta.runtime is not None:
        label = f"{node_meta.name}/{node_meta.runtime.type_name}"
    return Text(label, style="dim")


def _get_state_label(node_meta: IcoRuntimeNodeMeta) -> Text:
    color = _get_state_color(node_meta.state)
    return Text(f"<{node_meta.state}> ", style=color)


def _get_signature_label(
    node_meta: IcoNodeMeta,
    data_format: SignatureFormat,
    add_arrow: bool = True,
) -> Text:
    match data_format:
        case "Full":
            return Text(f"{node_meta.ico_form.format()}", style="cyan")
        case "Input":
            return Text(f"{format_ico_type(node_meta.ico_form.i)}", style="cyan")
        case "Output":
            signature = format_ico_type(node_meta.ico_form.o)
            if add_arrow:
                signature = "→ " + signature
            return Text(f"{signature}", style="cyan")


def _get_name_text(name: str, name_origin: str) -> Text:
    return Text(name, style=name_origin_scheme.get(name_origin, "black"))


# ────────────────────────────────────────────────
# Runtime Flow, Display format: Tree
# ────────────────────────────────────────────────


def _build_runtime_meta_tree(flow: IcoRuntimeNodeMeta) -> Tree:
    label = _construct_runtime_label(flow)
    node = Tree(label)

    for child in flow.children:
        node.add(_build_runtime_meta_tree(child))

    return node


# ────── Runtime Node Label builder ──────


def _construct_runtime_label(node_meta: IcoRuntimeNodeMeta) -> Text:
    # Format name with origin-based style
    text = _get_name_text(node_meta.name, node_meta.name_origin)

    # Show type name
    text.append(f" {node_meta.type_name}", style="dim")

    # Show runtime states if available
    color = _get_state_color(node_meta.state)
    text.append(f" <{node_meta.state}>", style=color)

    return text


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
    class_ = "#0052CC"
    keyword = "#E12EE1"
    dimmed = "#C8C8C8"
    meta = "#4FC1FF"
    text = "#4EB169"


name_origin_scheme = {
    "fn": NameStyle.fn.value,
    "class": NameStyle.class_.value,
    "direct": NameStyle.text.value,
}


# ────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────

DataBatch = Iterator[float]
TrainBatch = int


if __name__ == "__main__":
    # ──── 1. Define a batched data source ────
    data = [0.5, 1.0, 1.5, 2.0, 0.8, 0.2, 1.0, 1.2, 0.9]

    @source()
    def indices() -> Iterator[int]:
        yield from range(len(data))

    @operator()
    def batcher(indices: Iterator[int]) -> Iterator[Iterator[int]]:
        batch: list[int] = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == 3:
                yield iter(batch)
                batch = []

        if len(batch) > 0:
            yield iter(batch)

    @operator()
    def fetch_data_item(index: int) -> float:
        return data[index]

    # ──── 2. Define augmentation & collation pipelines per item ────

    @operator()
    def scale(x: float) -> float:
        return x * 1.1

    @operator()
    def shift(x: float) -> float:
        return x + 0.1

    augment = IcoPipeline(
        scale,
        shift,
        name="Item Augment Pipeline",
    )

    @operator()
    def collate_max(batch: DataBatch) -> str:
        return f"{max(batch)}"

    # Augment each item in the batch, then collate to single value (mimic data loader collate)
    augment_item = (fetch_data_item | augment).stream()
    augment_item.name = "Augment stream"
    stream_batch = (augment_item | collate_max).stream()
    stream_batch.name = "Train data stream"

    """
    ──── 3. Compose into a full data stream ────
    • dataset: () → Iterable[Iterable[float]]
    • stream: Iterable[Iterable[float]] → Iterable[float]
    • augment.map(): Iterable[float] → Iterable[float]
    • augment: float → float,
    • collate: Iterable[float] → float
    • dataflow: () → Iterable[float]
    """
    data_stream = indices | batcher | stream_batch

    # ──── 4. Define training pipeline ────
    # We collate batch into single float via max, so input into training stream is Iterable[float],
    # without batch dimension. In real scenarios, item could be tensors with batch dim.
    # Each batch item passes through a process that transforms floats ≤ 1 via pow(2)
    # train_step: float -> float, ICO Form: I → O
    # train_process: float -> float, ICO Form: C → C
    # I → C → O = Iterable[float] → Iterable[float] → Iterable[float]

    def optimize(values: str) -> str:
        return f"{values} x2"

    def log_metrics(values: str) -> str:
        return values

    train_iter = IcoPipeline(optimize, log_metrics)
    train_stream = train_iter.stream()
    train_stream.name = "Train stream"

    @sink()
    def save_result(item: str) -> None:
        pass

    # ──── 5. Combine all into the full flow ────
    full_flow = data_stream | train_stream | save_result
    full_flow.name = "Example flow"

    runtime = IcoRuntimeContour(full_flow, name="full_flow_runtime")
    runtime.activate()

    # ──── 6. Visualize ────
    describe(full_flow)
