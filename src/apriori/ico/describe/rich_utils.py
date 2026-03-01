from __future__ import annotations

from types import FunctionType

from rich.text import Text

from apriori.ico.describe.options import RendererOptions
from apriori.ico.describe.plan.options import PlanRendererOptions
from apriori.ico.describe.rich_style import DescribeStyle


def render_node_class(
    node: object,
    options: RendererOptions,
    args_info: Text | None = None,
) -> Text:
    """Render node class name with Rich styling and optional args."""
    display_name = Text(f"{type(node).__name__}(", style=DescribeStyle.class_.value)

    if options.dim_ico_nodes:
        display_name.stylize("dim", 0, len(display_name))

    if args_info is not None:
        display_name += args_info

    display_name += Text(")", style=DescribeStyle.class_.value)

    if options.dim_ico_nodes:
        display_name.stylize("dim", len(display_name) - 1, len(display_name))

    return display_name


def render_callable(obj: object, options: PlanRendererOptions) -> Text:
    """Render callable object with appropriate Rich styling."""
    name = extract_fn_display_name(obj)  # type: ignore
    if name:
        return Text(name, style=DescribeStyle.fn.value)

    if options.callable_format == "str()":
        return Text(str(obj), style=DescribeStyle.meta.value)

    name = extract_class_display_name(obj)  # type: ignore
    if name:
        return Text(f"{name}()", style=DescribeStyle.class_.value)

    return Text("Unknown", style=DescribeStyle.dimmed.value)


# ──────────── Helpers  ────────────


def extract_fn_display_name(fn: object) -> str | None:
    cls = getattr(fn, "__class__", None)

    if cls is FunctionType:
        return getattr(fn, "__name__", None)

    return None


def extract_class_display_name(obj: object) -> str | None:
    cls = getattr(obj, "__class__", None)

    if cls is None:
        return None

    return getattr(cls, "__name__", None)
