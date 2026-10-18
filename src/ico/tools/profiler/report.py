"""The profile on the plan: cells per plan key, and a Rich report.

:func:`profile_cells` turns a :class:`ProfileSnapshot` into formatted cells
per plan key, for every consumer that shows the plan (the Rich report, the
Live Board). Rows of the plan join them by ``plan_view(...)["keys"]``.
"""

from __future__ import annotations

from typing import TypedDict

from rich.console import Console
from rich.table import Table
from rich.text import Text

from ico.core.node import IcoNodeProtocol
from ico.describe.plan.options import PlanRendererOptions
from ico.describe.plan.rich_renderer.plan_renderer import PlanRenderer
from ico.tools.profiler.table import NodeProfile
from ico.tools.profiler.tool import ProfileSnapshot

PROFILE_COLUMNS: tuple[str, ...] = (
    "Self",
    "Total",
    "Share of work",
    "CPU",
    "Count",
    "p95",
    "Info",
)
"""Columns of :class:`ProfileCells`, in display order."""


STARTUP_MIN = 1e-3
"""Self startup below this (seconds) is not shown."""
MAX_MIN = 1e-3
MAX_FACTOR = 10
"""The max of a node is shown when it is at least ``MAX_MIN`` seconds and
``MAX_FACTOR`` times its p95: a rare slow call or item."""


class ProfileCells(TypedDict):
    """Formatted profile of one plan row; empty strings for a node that was
    not measured."""

    self: str
    """Wall time of the node itself."""
    total: str
    """Wall self time of the node and the nodes below it in the plan."""
    share: str
    """``self`` as a share of all work measured: the sum of the self times
    of all nodes, in all threads and processes (more than the wall time of a
    run with parallel pools)."""
    cpu: str
    """CPU self time."""
    count: str
    """Items for a node returning iterators, else calls."""
    p95: str
    """95th percentile of the wall time of a call or an item."""
    info: str
    """Pool utilization, worker transfer, startup (first call of a value
    node, first ``next()`` of an iterator), a max far above p95, errors."""
    share_value: float
    """``share`` as a number (0..1), e.g. for a bar."""
    hot: bool
    """The node with the largest self time."""


def profile_cells(snapshot: ProfileSnapshot) -> dict[str, ProfileCells]:
    """Formatted cells of every node of a profile, by plan key."""
    nodes = snapshot["nodes"]
    selfs = {n["key"]: _wall(n, "self") for n in nodes}
    whole = sum(selfs.values())
    hot = max(selfs, key=lambda k: selfs[k], default=None)
    out: dict[str, ProfileCells] = {}
    for node in nodes:
        key = node["key"]
        measured = node["calls"] > 0 or node["items"] > 0
        own = selfs[key]
        subtree = sum(
            v for k, v in selfs.items() if k == key or k.startswith(key + ".")
        )
        share = own / whole if whole > 0 else 0.0
        cpu = node["metrics"].get("cpu")
        out[key] = {
            "self": duration(own) if measured else "",
            "total": duration(subtree) if subtree > 0 else "",
            "share": f"{share:.0%}" if measured and whole > 0 else "",
            "cpu": duration(cpu["self"]) if cpu is not None and measured else "",
            "count": str(node["items"] or node["calls"]) if measured else "",
            "p95": _p95(node),
            "info": _info(node),
            "share_value": share,
            "hot": key == hot and own > 0,
        }
    return out


def duration(seconds: float) -> str:
    """``12µs``, ``3.4ms``, ``1.25s``."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f}µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f}ms"
    return f"{seconds:.2f}s"


def print_profile(
    flow: IcoNodeProtocol,
    snapshot: ProfileSnapshot,
    *,
    console: Console | None = None,
    options: PlanRendererOptions | None = None,
) -> None:
    """Print the plan of ``flow`` with the profile columns of ``snapshot``
    (instrumented from that flow) after the flow column; the hottest node is
    highlighted. ``options`` selects the plan columns (by default Flow,
    Signature, Name)."""
    options = options or PlanRendererOptions(query_iterable_size=False)
    _ProfileRenderer(snapshot, options, console).print(flow, snapshot)


class _ProfileRenderer(PlanRenderer):
    def __init__(
        self,
        snapshot: ProfileSnapshot,
        options: PlanRendererOptions,
        console: Console | None,
    ) -> None:
        super().__init__(options=options, console=console)
        self._cells = profile_cells(snapshot)

    def print(self, root: IcoNodeProtocol, snapshot: ProfileSnapshot) -> None:
        self._table = self._create_table()
        self.walk(root)
        elapsed, overhead = snapshot["elapsed"], snapshot["overhead"]
        title = root.name or type(root).__name__
        self.console.rule(
            f"[bold blue]Profile: {title}[/] · {duration(elapsed)}"
            f" · probes ≈{duration(overhead)}",
            style="dim blue",
        )
        self.console.print(self._table)

    def _create_table(self) -> Table:
        # The profile follows the flow column: Flow, profile, Signature, Name.
        table = Table(show_lines=False, show_edge=False, box=None)
        first, *rest = self.options.columns
        table.add_column(first, no_wrap=True)
        for column in PROFILE_COLUMNS:
            table.add_column(column, no_wrap=True, justify="right")
        for column in rest:
            table.add_column(column, no_wrap=True)
        return table

    def add_row(self, columns: list[Text], key: str | None) -> None:
        assert self._table is not None
        cells = self._cells.get(key) if key is not None else None
        first, *rest = columns
        if cells is None:
            self._table.add_row(first, *[""] * len(PROFILE_COLUMNS), *rest)
            return
        values = [cells["total"], cells["share"], cells["cpu"]]
        values += [cells["count"], cells["p95"], cells["info"]]
        self._table.add_row(
            first,
            Text(cells["self"], style="bold red" if cells["hot"] else ""),
            *(Text(v, style="dim" if i == 0 else "") for i, v in enumerate(values)),
            *rest,
        )


def _wall(node: NodeProfile, field: str) -> float:
    wall = node["metrics"].get("wall")
    if wall is None:
        return 0.0
    return float(wall["self"] if field == "self" else wall["total"])


def _p95(node: NodeProfile) -> str:
    """Only for a distribution: not for a node called once (a sink)."""
    wall = node["metrics"].get("wall")
    if wall is None or wall["count"] < 2 or wall["p95"] is None:
        return ""
    return duration(wall["p95"])


def _info(node: NodeProfile) -> str:
    parts: list[str] = []
    if node["utilization"] is not None:
        parts.append(f"util {node['utilization']:.0%}")
    if node["transfer"] is not None:
        parts.append(f"transfer {duration(node['transfer'])}")
    startup = node["startup"]
    wall = startup.get("wall") if startup is not None else None
    # Its own setup only: not the startup of the upstream it pulled; not for
    # a node with a single event (a sink), which would be all startup.
    single = node["calls"] + node["items"] <= 1
    if wall is not None and wall["self"] >= STARTUP_MIN and not single:
        parts.append(f"startup {duration(wall['self'])}")
    events = node["metrics"].get("wall")
    if (
        events is not None
        and events["count"] >= 2
        and events["max"] is not None
        and events["p95"] is not None
        and events["max"] >= MAX_MIN
        and events["max"] >= MAX_FACTOR * events["p95"]
    ):
        parts.append(f"max {duration(events['max'])}")
    if node["errors"]:
        parts.append(f"{node['errors']} error(s)")
    return " · ".join(parts)
