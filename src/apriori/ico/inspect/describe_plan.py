from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import Any, Literal, TypeAlias

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from apriori.ico.core.meta.meta import IcoNodeMeta, IcoRuntimeNodeMeta
from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import IcoOperator, operator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.runtime_wrapper import IcoRuntimeWrapper
from apriori.ico.core.runtime.state import (
    FaultState,
    IcoRuntimeState,
    ReadyState,
    RunningState,
)
from apriori.ico.core.sink import IcoSink, sink
from apriori.ico.core.source import IcoSource, source
from apriori.ico.inspect.describe_utils import NameStyles, format_ico_type

# ────────────────────────────────────────────────
# Plan API
# ────────────────────────────────────────────────

DiscribeFormat: TypeAlias = Literal["Tree", "Plan"]
SignatureFormat: TypeAlias = Literal["Full", "Input", "Output"]


def describe_plan(
    node: IcoNode,
    *,
    format: DiscribeFormat | None = None,
    signature_format: SignatureFormat | None = None,
    include_runtime: bool = False,
    console: Console | None = None,
) -> None:
    """Render an ICO operator graph (flow) as a rich tree."""

    console = console or Console()

    format = format or "Plan"
    signature_format = signature_format or "Full"

    if format == "Tree":
        tree = build_operators_tree(node, signature_format=signature_format)

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

        build_flow_plan(
            meta,
            signature_format=signature_format,
            table=table,
            indent=Text(""),
        )

        console.rule(f"[bold blue]Execution plan: {meta.name}", style="dim blue")
        console.print(table)


def infer_display_name(node: IcoNode) -> Text:
    name = node.name
    name_origin = "direct" if name is not None else None

    if (
        name is None
        and (isinstance(node, IcoOperator) and node.type_name == "Operator")
        or isinstance(node, IcoSource | IcoSink | IcoRuntimeWrapper)
    ):
        name = extract_fn_display_name(node.original_fn)  # type: ignore
        if name is not None:
            name_origin = "fn"

    if name is None:
        name = extract_class_display_name(node)  # type: ignore
        if name is not None:
            name_origin = "class"

    if name is None or name_origin is None:
        name = node.type_name
        name_origin = "type_name"


def get_sink_display_name(source: IcoSource[Any]) -> Text:
    data_source_name = None

    if source.name is not None:
        data_source_name = Text(source.name, style=NameStyles.text.value)

    elif callable(source.provider):
        data_source_name = extract_fn_display_name(source.provider)  # type: ignore
        data_source_name = data_source_name or extract_class_display_name(
            source.provider
        )  # type: ignore
        data_source_name = data_source_name or "Unknown"

    if isinstance(source.provider, Sized):
        data_source_size = Text(
            f", size={len(source.provider)}", style=NameStyles.meta.value
        )
    else:
        data_source_size = None

    assert data_source_name is not None

    display_name = Text("IcoSource(", style=NameStyles.class_.value)
    display_name.append(data_source_name)

    if data_source_size is not None:
        display_name.append(data_source_size)

    display_name.append(Text(")", style=NameStyles.class_.value))

    return display_name


# ────────────────────────────────────────────────
# Computation Flow, Display format: Tree
# ────────────────────────────────────────────────


def build_operators_tree(
    node: IcoNode,
    *,
    signature_format: SignatureFormat,
) -> Tree:
    label = _construct_tree_label(node, data_format=signature_format)
    node = Tree(label)

    for child in node.children:
        node.add(build_operators_tree(child, signature_format=signature_format))

    if node.runtime is not None:
        for child in node.runtime.children:
            node.add(build_runtime_meta_tree(child))

    return node


def _construct_tree_label(node: IcoNode, data_format: SignatureFormat) -> Text:
    # Format name with origin-based style
    # text = _get_expanded_name_text(node_meta)

    display_name = infer_display_name(node)
    display_name.append(" ")

    # Show type name
    display_name.append(_get_type_label(node))
    display_name.append(" ")

    if node.runtime is not None:
        display_name.append(_get_state_label(node.runtime))
        display_name.append(" ")

    # Show Signature
    display_name.append(_get_signature_label(node, data_format))

    return display_name


# ────────────────────────────────────────────────
# Computation Flow, Display format: Flow
# ────────────────────────────────────────────────

FLATTEN_TYPES = ("Chain", "Pipeline", "Runtime Wrapper")
# "Iterator"


def build_flow_plan(
    node: IcoNode,
    *,
    signature_format: SignatureFormat,
    table: Table,
    indent: Text,
) -> None:
    """Deep-first left-to-right traversal of flow structure.
    Chain nodes are skipped to flatten the flow representation.
    Structural nodes like Iterator produce nested level of sub-flows
    """

    if node.type_name in FLATTEN_TYPES:
        # Flatten nodes and continue with left-to-right children
        for child in node.children:
            build_flow_plan(
                child,
                signature_format=signature_format,
                table=table,
                indent=indent.copy(),
            )
        return

    # Create a new flow node
    if node.type_name == "Stream":
        _build_stream_execution_flow(
            node,
            signature_format=signature_format,
            table=table,
            indent=indent.copy(),
        )
        return

    columns = _construct_flow_label(
        node,
        indent=indent,
        data_format=signature_format,
    )
    table.add_row(*[c for c in columns if c is not None])

    # Structural nodes - produce sub-flows for current node,
    # overwise - just a leaf node
    if len(node.children) > 0:
        for child in node.children:
            build_flow_plan(
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

    itrerate_over_text = Text("iterate", style=NameStyles.keyword.value)
    iterate_name = indent.copy().append(Text("╭─ ")).append(itrerate_over_text)
    iterate_type = _get_signature_label(stream_meta, "Input")

    table.add_row(iterate_name, None, iterate_type, None)

    for child in stream_meta.children:
        build_flow_plan(
            child,
            signature_format=signature_format,
            table=table,
            indent=indent.copy().append(Text("│   ")),
        )

    yield_text = indent.copy().append(Text("╰─ "))
    yield_text.append(Text("yield", style=NameStyles.keyword.value))
    yield_type = _get_signature_label(stream_meta, "Output")

    table.add_row(yield_text, None, yield_type, None)


def _construct_flow_label(
    node: IcoNode,
    indent: Text,
    data_format: SignatureFormat,
) -> tuple[Text, Text, Text, Text | None]:
    # Format name with origin-based style
    # text = _get_expanded_name_text(node_meta)

    name_col = _get_name_label(node)
    name_col = indent.append(name_col)

    # Show type name
    type_col = _get_type_label(node)

    # Show Signature
    signature_col = _get_signature_label(node, data_format)

    if node.runtime is not None:
        state_col = _get_state_label(node.runtime)
    else:
        state_col = None

    return name_col, type_col, signature_col, state_col


# ────── Compute Node Label builder ──────


def _get_name_label(node: IcoNode) -> Text:
    return _get_name_text(f"{node.name} ", node.name_origin)


def _get_type_label(node: IcoNode) -> Text:
    if isinstance(node, IcoRuntimeNode):
        operator_base = next(
            p for p in node.__class__.__bases__ if isinstance(p, IcoOperator)
        )
        runtime_base = next(
            p for p in node.__class__.__bases__ if isinstance(p, IcoRuntimeNode)
        )
        label = f"{operator_base.__name__}/{runtime_base.__name__}"
    else:
        label = node.__class__.__name__

    return Text(label, style="gray70")


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


def build_runtime_meta_tree(flow: IcoRuntimeNodeMeta) -> Tree:
    label = _construct_runtime_label(flow)
    node = Tree(label)

    for child in flow.children:
        node.add(build_runtime_meta_tree(child))

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


name_origin_scheme = {
    "fn": NameStyles.fn.value,
    "class": NameStyles.class_.value,
    "direct": NameStyles.text.value,
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

    console = Console()
    console.print(get_source_display_name(indices))

    source2 = IcoSource(data, name="Indices")
    console.print(get_source_display_name(source2))

    print("")

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

    runtime = IcoRuntimeNode(full_flow, name="full_flow_runtime")
    runtime.activate()

    # ──── 6. Visualize ────
    describe_plan(full_flow)
