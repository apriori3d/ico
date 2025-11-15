from __future__ import annotations

from rich.text import Text
from rich.tree import Tree

from apriori.ico.core.meta.flow_meta import IcoFlowMeta
from apriori.ico.core.runtime.execution import IcoExecutionState
from apriori.ico.core.runtime.types import IcoRuntimeCommandType, IcoRuntimeStateType


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

    # Show ICO form (signature)
    if show_ico_form:
        text.append(f"  [{flow_meta.ico_form.name}]", style="magenta")

    # Show runtime states if available
    if show_states:
        if flow_meta.state is not None:
            color = {
                IcoRuntimeStateType.unknown: "grey50",
                IcoRuntimeStateType.running: "yellow",
                IcoRuntimeStateType.running: "green",
                IcoRuntimeStateType.cleaned: "grey70",
            }.get(flow_meta.state, "white")
            text.append(f" [{flow_meta.state.name}]", style=color)

        if flow_meta.exec_state is not None:
            color = {
                IcoExecutionState.idle: "grey50",
                IcoExecutionState.running: "blue",
                IcoExecutionState.done: "green",
                IcoExecutionState.faulted: "red",
            }.get(flow_meta.exec_state, "white")
            text.append(f" <{flow_meta.exec_state.name}>", style=color)

    return text


# ──── Example usage ────

if __name__ == "__main__":
    from collections.abc import Iterable

    from rich.console import Console

    from apriori.ico.core import (
        IcoFlowMeta,
        IcoOperator,
        IcoPipeline,
        IcoSource,
        IcoStream,
    )

    # ──── 1. Define a batched data source ────
    def generate_batches() -> Iterable[list[float]]:
        """Simulate dataset batches: () → Iterable[list[float]]"""
        return [
            [0.5, 1.0, 1.5],
            [2.0, 0.8, 0.2],
            [1.0, 1.2, 0.9],
        ]

    dataset = IcoSource[list[float]](generate_batches, name="dataset")

    # ──── 2. Define augmentation & collation pipelines ────

    augment = IcoPipeline[float, float, float](
        context=IcoOperator[float, float](lambda x: x, name="identity_ctx"),
        body=[
            IcoOperator[float, float](lambda x: x * 1.1, name="scale_up"),
            IcoOperator[float, float](lambda x: x + 0.1, name="shift"),
        ],
        output=IcoOperator[float, float](lambda x: x, name="identity_out"),
        name="augment_pipeline",
    )

    collate = IcoPipeline[Iterable[float], Iterable[float], float](
        context=IcoOperator[Iterable[float], Iterable[float]](list, name="to_list"),
        body=[],
        output=IcoOperator[Iterable[float], float](max, name="max_value"),
        name="collate_pipeline",
    )
    """
    ──── 3. Compose into a full data stream ────
    • dataset: () → Iterable[Iterable[float]]
    • stream: Iterable[Iterable[float]] → Iterable[float]
    • augment.map(): Iterable[float] → Iterable[float]
    • augment: float → float → float,
    • collate: Iterable[float] → Iterable[float] → float
    • dataflow: () → Iterable[float]
    """

    dataflow = dataset | IcoStream(augment.map() | collate, name="data_stream")

    # ──── 4. Define training pipeline ────
    # We collate batch into single float via max, so input into training stream is Iterable[float],
    # without batch dimension. In real scenarios, item could be tensors with batch dim.
    # Each batch item passes through a process that transforms floats ≤ 1 via pow(2)
    # train_step: float -> float, ICO Form: I → O
    # train_process: float -> float, ICO Form: C → C
    # I → C → O = Iterable[float] → Iterable[float] → Iterable[float]

    def pow_if_needed(values: float) -> float:
        return values**2 if values <= 1.0 else values

    train_step = IcoOperator[float, float](pow_if_needed, name="train_step")

    train_pipeline = IcoPipeline[float, float, float](
        context=IcoOperator[float, float](lambda xs: xs, name="identity_ctx"),
        body=[train_step],
        output=IcoOperator[float, float](lambda xs: xs, name="identity_out"),
        name="train_pipeline",
    )

    train_stream = IcoStream(train_pipeline, name="train_stream")

    # ──── 5. Combine all into the full flow ────
    full_flow = dataflow | train_stream

    full_flow.broadcast_event(IcoRuntimeCommandType.activate)
    full_flow.broadcast_event(IcoRuntimeCommandType.reset)

    # ──── 6. Visualize ────
    flow_meta = IcoFlowMeta.from_operator(full_flow)

    console = Console()
    console.rule("[bold blue]ICO Dataflow: Dataset → Stream → Train")
    console.print(describe(flow_meta, show_states=True))
