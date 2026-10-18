"""JSON backend of ``describe`` for flows: the console plan as data.

:func:`plan_view` runs the RichText plan renderer, with all its per-node
renderers, group keywords (``for each in`` … ``yield``, ``parallel in``,
``iterate in`` … ``emit``, ``send to`` … ``receive``) and tree lines, but
collects the rows as data instead of printing a table. Each cell is a list
of styled segments; ``style`` names a :class:`DescribeStyle` role, so a
consumer (e.g. a browser board) colors the plan like the console without
knowing node kinds::

    {"title": "Training", "columns": ["Flow", "Signature", "Name"],
     "rows": [[[{"text": "╭── ", "style": "group", "dim": false},
                {"text": "for each in ", "style": "keyword", "dim": false}, ...],
               [{"text": "Iterator[int]", "style": "signature", "dim": false}],
               []], ...]}
"""

from __future__ import annotations

from typing import TypedDict

from rich.style import Style
from rich.text import Text

from ico.core.node import IcoNodeProtocol
from ico.describe.plan.options import PlanRendererOptions
from ico.describe.plan.rich_renderer.plan_renderer import PlanRenderer
from ico.describe.rich_style import DescribeStyle


class PlanSegment(TypedDict):
    """A run of text with one style; JSON-compatible."""

    text: str
    style: str | None
    """A ``DescribeStyle`` member name (``keyword``, ``class_``, ``fn``,
    ``signature``, ``group``, ...), ``None`` for plain text."""
    dim: bool


PlanCell = list[PlanSegment]


class PlanView(TypedDict):
    """The console plan of a flow: a title, column names and rows of cells."""

    title: str
    columns: list[str]
    rows: list[list[PlanCell]]
    keys: list[str | None]
    """The plan key of the node of each row (``0.1.0``, see
    :mod:`ico.core.plan_keys`); ``None`` for closing rows (``yield``,
    ``collect``, ...) and rows of nodes already keyed by another row."""


def plan_view(
    node: IcoNodeProtocol, options: PlanRendererOptions | None = None
) -> PlanView:
    """The console plan of a flow as JSON-compatible rows of styled segments.

    Args:
        node: The flow.
        options: Plan options as for ``describe``; by default sources are not
            called to query their size.

    Example:
        >>> view = plan_view(source | batcher | stream)
        >>> "".join(s["text"] for s in view["rows"][0][0])
        '📚IcoSource(numbers)'
    """
    options = options or PlanRendererOptions(query_iterable_size=False)
    return _RowCollector(options).collect(node)


class _RowCollector(PlanRenderer):
    """The plan renderer, keeping rows instead of printing a table."""

    def __init__(self, options: PlanRendererOptions) -> None:
        super().__init__(options=options)
        self._rows: list[list[PlanCell]] = []
        self._keys: list[str | None] = []

    def collect(self, root: IcoNodeProtocol) -> PlanView:
        self._rows, self._keys = [], []
        self.walk(root)
        return {
            "title": root.name or type(root).__name__,
            "columns": list(self.options.columns),
            "rows": self._rows,
            "keys": self._keys,
        }

    def add_row(self, columns: list[Text], key: str | None) -> None:
        self._rows.append([_segments(text) for text in columns])
        self._keys.append(key)


def _style_key(style: str | Style) -> str:
    return str(Style.parse(style) if isinstance(style, str) else style)


# Style → role name; the first member wins for equal styles (group, tree).
_ROLES: dict[str, str] = {}
for _member in DescribeStyle:
    _ROLES.setdefault(_style_key(_member.value), _member.name)


def _role(style: str | Style) -> tuple[str | None, bool]:
    """The ``DescribeStyle`` role of a style and whether it dims the text."""
    parsed = Style.parse(style) if isinstance(style, str) else style
    if parsed.dim and parsed.color is None:
        return None, True  # e.g. ``stylize("dim", ...)``: dims, keeps the color
    return _ROLES.get(_style_key(parsed)), bool(parsed.dim)


def _segments(text: Text) -> PlanCell:
    """Split a Rich text into runs of one role, as the console colors it.

    Later spans win over earlier ones and over the base style; a plain
    ``dim`` span only dims.
    """
    plain = text.plain
    bounds = sorted({0, len(plain), *(b for s in text.spans for b in (s.start, s.end))})
    base_role, base_dim = _role(text.style)
    out: PlanCell = []
    for start, end in zip(bounds, bounds[1:], strict=False):
        if start >= len(plain):
            break
        role, dim = base_role, base_dim
        for span in text.spans:
            if span.start <= start and end <= span.end:
                span_role, span_dim = _role(span.style)
                role = span_role or role
                dim = dim or span_dim
        piece = plain[start:end]
        if out and out[-1]["style"] == role and out[-1]["dim"] == dim:
            out[-1]["text"] += piece
        else:
            out.append({"text": piece, "style": role, "dim": dim})
    return out
