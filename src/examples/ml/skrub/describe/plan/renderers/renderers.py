from __future__ import annotations

from typing import Any, cast

from rich.text import Text

from examples.ml.skrub.base import SKOperator, SKOperatorProtocol
from examples.ml.skrub.data import XSource, XYSource
from examples.ml.skrub.describe.plan.utils import SKRendererPerOperatorOptions
from ico.core.node import IcoNodeProtocol
from ico.describe.plan.rich_renderer.renderer_registry import (
    register_renderer,
    select_the_most_specific_type,
)
from ico.describe.plan.rich_renderer.row_renderer import (
    RowRenderer,
)
from ico.describe.rich_style import DescribeStyle
from ico.describe.rich_utils import (
    render_callable,
    render_node_class,
)
from ico.describe.utils import match_icon

AnySKOperator = SKOperator[Any, Any]


@register_renderer(SKOperator)
class BaseRender(RowRenderer):
    def render_flow_column(self, node: IcoNodeProtocol) -> Text:
        """Render Flow column: icons, class name, and arguments."""

        if not isinstance(node, SKOperatorProtocol):
            return super().render_flow_column(node)

        any_sk_operator = cast(AnySKOperator, node)

        # Predefined subclasses of SKBaseTransformer will be rendered using the class name
        # of their estimator, for better readability
        text = self.flow_column_prefix or Text("")

        if not self.flow_includes_node_info:
            return text

        if self.options.show_node_icons:
            icon = match_icon(self.options.node_icons, any_sk_operator)
            if icon:
                text += Text(icon)

        from examples.ml.skrub.transformer import SKTransformer

        if (
            isinstance(any_sk_operator, SKTransformer)
            and len(SKRendererPerOperatorOptions) > 0
        ):
            # Select a transformer target class to show if available

            target_type = select_the_most_specific_type(
                type(any_sk_operator), list(SKRendererPerOperatorOptions.keys())
            )
            assert target_type is not None
            options = SKRendererPerOperatorOptions.get(target_type, None)
            target_for_class = (
                any_sk_operator.transformer
                if options and options.show_estimator_class
                else any_sk_operator
            )
        else:
            target_for_class = any_sk_operator

        # Render args
        args_info = self._render_node_args_info(any_sk_operator)

        # Render class name
        text += render_node_class(
            target_for_class, options=self.options, args_info=args_info
        )

        if self.flow_column_postfix:
            text += self.flow_column_postfix

        return text

    def _render_node_args_info(self, node: IcoNodeProtocol) -> Text:
        if not isinstance(node, SKOperatorProtocol):
            return Text()

        any_estimator = cast(AnySKOperator, node)
        options = SKRendererPerOperatorOptions.get(type(any_estimator), None)
        if not options:
            return Text()

        if options.show_args_named is None or len(options.show_args_named) == 0:
            return Text()

        from examples.ml.skrub.transformer import SKTransformer

        args: list[str] = []

        if isinstance(any_estimator, SKTransformer):
            estimator_target = any_estimator.transformer
        else:
            estimator_target = None

        for name in options.show_args_named:
            arg_value = (
                getattr(any_estimator, name)
                if hasattr(any_estimator, name) or not estimator_target
                else getattr(estimator_target, name, "")
            )
            args.append(f"{name}={arg_value}")

        return Text(", ".join(args), style=DescribeStyle.meta.value)


@register_renderer(XSource, XYSource)
class SKSourceRender(RowRenderer):
    """Specialized renderer for SKSource nodes with data size information."""

    def _render_node_args_info(self, node: IcoNodeProtocol) -> Text:
        """Render source provider info with optional size details."""
        assert isinstance(node, XSource | XYSource)
        source = cast(XSource[Any, Any], node)

        return render_callable(source.provider, options=self.options)


# @register_renderer(SKApplyToCols)
# class ApplyToColsRenderer(GroupRenderer):
#     def __init__(self, options: PlanRendererOptions) -> None:
#         super().__init__(
#             options=options,
#             header_renderer=RowRenderer(
#                 flow_column_prefix=Text(
#                     "apply to columns", style=DescribeStyle.keyword.value
#                 ),
#                 flow_includes_node_info=False,
#                 options=options,
#             ),
#             footer_renderer=RowRenderer(
#                 flow_column_prefix=Text(
#                     "concatinate", style=DescribeStyle.keyword.value
#                 ),
#                 options=replace(options, signature_format="Output"),
#                 show_name_column=False,
#                 show_type_column=False,
#                 show_state_column=False,
#                 flow_includes_node_info=False,
#             ),
#         )
