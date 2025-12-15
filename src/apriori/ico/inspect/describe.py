from __future__ import annotations

from typing import Literal, TypeAlias, overload

from rich.console import Console

from apriori.ico.core.node import IcoNode
from apriori.ico.core.runtime.node import IcoRuntimeNode

# ────────────────────────────────────────────────
# Plan API
# ────────────────────────────────────────────────

DiscribeFormat: TypeAlias = Literal["Tree", "Plan"]
SignatureFormat: TypeAlias = Literal["Full", "Input", "Output"]


@overload
def describe(
    node: IcoNode,
    *,
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
    include_runtime: bool = False,
    console: Console | None = None,
) -> None:
    pass
