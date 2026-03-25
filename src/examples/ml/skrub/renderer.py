from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from rich.text import Text

from examples.ml.skrub.base import SKBaseEstimator
from examples.ml.skrub.transformer import SKTransformer
from ico.core.node import IcoNode
from ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from ico.describe.plan.rich_renderer.row_renderer import (
    RowRenderer,
)
from ico.describe.rich_style import DescribeStyle
from ico.describe.rich_utils import (
    render_node_class,
)
from ico.describe.utils import match_icon


@dataclass
class RendererOperatorOptions:
    show_estimator_class: bool = False
    show_args_named: list[str] | None = None


AnyBaseEstimator = SKBaseEstimator[Any, Any]
TEstimator = TypeVar("TEstimator", bound=SKBaseEstimator[Any, Any])

SKRendererPerOperatorOptions = dict[type[AnyBaseEstimator], RendererOperatorOptions]()


def setup_renderer(
    options: RendererOperatorOptions,
) -> Callable[[type[TEstimator]], type[TEstimator]]:
    def decorator(
        operator_cls: type[TEstimator],
    ) -> type[TEstimator]:
        SKRendererPerOperatorOptions[operator_cls] = options
        return operator_cls

    return decorator


def setup_renderer_show_estimator() -> Callable[[type[TEstimator]], type[TEstimator]]:
    def decorator(
        operator_cls: type[TEstimator],
    ) -> type[TEstimator]:
        options = RendererOperatorOptions(show_estimator_class=True)
        SKRendererPerOperatorOptions[operator_cls] = options
        return operator_cls

    return decorator


def setup_renderer_show_args(
    *show_args_named: str,
) -> Callable[[type[TEstimator]], type[TEstimator]]:
    def decorator(
        operator_cls: type[TEstimator],
    ) -> type[TEstimator]:
        options = RendererOperatorOptions(show_args_named=list(show_args_named))
        SKRendererPerOperatorOptions[operator_cls] = options
        return operator_cls

    return decorator


@register_renderer(SKBaseEstimator)
class BaseRender(RowRenderer):
    def render_flow_column(self, node: IcoNode) -> Text:
        """Render Flow column: icons, class name, and arguments."""

        if not isinstance(node, SKBaseEstimator):
            return super().render_flow_column(node)

        any_estimator = cast(AnyBaseEstimator, node)

        # Predefined subclasses of SKBaseTransformer will be rendered using the class name
        # of their estimator, for better readability
        text = self.flow_column_prefix or Text("")

        if not self.flow_includes_node_info:
            return text

        if self.options.show_node_icons:
            icon = match_icon(self.options.node_icons, any_estimator)
            if icon:
                text += Text(icon)

        if isinstance(any_estimator, SKTransformer):
            options = SKRendererPerOperatorOptions.get(type(any_estimator), None)
            target_for_class = (
                any_estimator.transformer
                if options and options.show_estimator_class
                else any_estimator
            )
        else:
            target_for_class = any_estimator

        # Render args
        args_info = self._render_node_args_info(any_estimator)

        # Render class name
        text += render_node_class(
            target_for_class, options=self.options, args_info=args_info
        )

        if self.flow_column_postfix:
            text += self.flow_column_postfix

        return text

    def _render_node_args_info(self, node: IcoNode) -> Text:
        if not isinstance(node, SKBaseEstimator):
            return Text()

        any_estimator = cast(AnyBaseEstimator, node)
        options = SKRendererPerOperatorOptions.get(type(any_estimator), None)
        if not options:
            return Text()

        if options.show_args_named is None or len(options.show_args_named) == 0:
            return Text()

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
