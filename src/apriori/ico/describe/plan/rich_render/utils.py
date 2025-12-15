from __future__ import annotations

from enum import Enum

from apriori.ico.core.runtime.state import (
    FaultState,
    IcoRuntimeState,
    ReadyState,
    RunningState,
)


class PlanStyle(Enum):
    fn = "#A67F59"
    # class_ = "#9CDCFE"
    type = "#569CD6"
    class_ = "#0052CC"
    keyword = "#E12EE1"
    dimmed = "gray70"
    meta = "#4FC1FF"
    text = "#4EB169"
    string = "#DD1616"
    signature = "cyan"


state_styles = [
    (ReadyState, "green"),
    (RunningState, "yellow"),
    (FaultState, "red"),
    (IcoRuntimeState, "gray70"),
]


def get_state_color(state: IcoRuntimeState) -> str:
    for state_type, color in state_styles:
        if isinstance(state, state_type):  # type: ignore
            return color
    return "black"
