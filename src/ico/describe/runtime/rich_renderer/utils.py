from __future__ import annotations

from apriori.ico.core.runtime.state import (
    FaultState,
    IcoRuntimeState,
    ReadyState,
    RunningState,
)

state_styles = [
    (ReadyState, "green"),
    (RunningState, "yellow"),
    (FaultState, "red"),
    (IcoRuntimeState, "gray70"),
]


def get_state_color(state: IcoRuntimeState) -> str:
    """Get Rich console color for runtime state visualization."""
    for state_type, color in state_styles:
        if isinstance(state, state_type):  # type: ignore
            return color
    return "black"
