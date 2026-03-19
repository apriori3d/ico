from dataclasses import replace

from rich.text import Text

from ico.describe.plan.options import PlanRendererOptions
from ico.describe.plan.rich_renderer.row_renderer import RowRenderer
from ico.describe.rich_style import DescribeStyle


class GroupRenderer:
    """
    Renders subflows as expandable groups with entry/exit rows.

    Provides visual hierarchy using indentation and branch symbols
    for nodes containing child flows.
    """

    node_renderer: RowRenderer | None
    header_renderer: RowRenderer
    footer_renderer: RowRenderer

    def __init__(
        self,
        options: PlanRendererOptions,
        node_renderer: RowRenderer | None = None,
        header_renderer: RowRenderer | None = None,
        footer_renderer: RowRenderer | None = None,
    ) -> None:
        """Initialize group renderer with configurable row renderers."""
        self.node_renderer = node_renderer

        self.header_renderer = header_renderer or RowRenderer(
            flow_column_prefix=Text("Subflow Entry ", style=DescribeStyle.dimmed.value),
            options=replace(options, signature_format="Input"),
        )
        self.footer_renderer = footer_renderer or RowRenderer(
            flow_column_prefix=Text("Subflow Exit", style=DescribeStyle.dimmed.value),
            options=replace(options, signature_format="Output"),
            show_name_column=False,
            show_type_column=False,
            show_state_column=False,
            flow_includes_node_info=False,
        )
