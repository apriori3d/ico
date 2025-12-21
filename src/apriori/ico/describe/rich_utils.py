from __future__ import annotations

from rich.text import Text

from apriori.ico.describe.options import RendererOptions
from apriori.ico.describe.plan.options import PlanRendererOptions
from apriori.ico.describe.rich_style import DescribeStyle
from apriori.ico.inspect.utils import (
    extract_class_display_name,
    extract_fn_display_name,
)


def render_node_class(
    node: object,
    options: RendererOptions,
    args_info: Text | None = None,
) -> Text:
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
    name = extract_fn_display_name(obj)  # type: ignore
    if name:
        return Text(name, style=DescribeStyle.fn.value)

    if options.callable_format == "str()":
        return Text(str(obj), style=DescribeStyle.meta.value)

    name = extract_class_display_name(obj)  # type: ignore
    if name:
        return Text(f"{name}()", style=DescribeStyle.class_.value)

    return Text("Unknown", style=DescribeStyle.dimmed.value)
