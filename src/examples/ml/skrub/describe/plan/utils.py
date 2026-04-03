from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from examples.ml.skrub.base import SKOperator


@dataclass
class RendererOperatorOptions:
    show_estimator_class: bool = False
    show_args_named: list[str] | None = None


TEstimator = TypeVar("TEstimator", bound=SKOperator[Any, Any])

SKRendererPerOperatorOptions = dict[type[object], RendererOperatorOptions]()


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
