from __future__ import annotations

import abc
from typing import Any, Generic, TypeVar, cast

import sklearn  # type: ignore[import-untyped]

from examples.ml.skrub.base import (
    SKBaseEstimator,
)
from examples.ml.skrub.data import (
    SKData,
    SKResultTypes,
    XDataFrame,
    XSeries,
    XyDataFrame,
    XySeries,
    wrap_result_data,
)
from examples.ml.skrub.describe.plan.utils import (
    setup_renderer_show_estimator,
)

# from ico.core.operator import I, O

I = TypeVar("I", bound=SKData)  # noqa: E741
O = TypeVar("O", bound=SKData)  # noqa: E741


class SKBaseTransformer(Generic[I, O], SKBaseEstimator[I, O], abc.ABC):
    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name=name)

    def _estimator_fn(self, input: I) -> O:
        match self.mode:
            case "fit":
                x1 = self._fit_transform(input)

            case "predict":
                x1 = self._transform(input)

        return self._wrap_result(input, x1)

    @abc.abstractmethod
    def _fit_transform(self, input: I) -> SKResultTypes: ...

    @abc.abstractmethod
    def _transform(self, input: I) -> SKResultTypes: ...

    def _wrap_result(self, input: I, x1: SKResultTypes) -> O:
        return cast(O, wrap_result_data(cast(SKData, input), x1))


@setup_renderer_show_estimator()
class SKTransformer(Generic[I, O], SKBaseTransformer[I, O]):
    transformer: sklearn.base.BaseEstimator

    def __init__(
        self,
        transformer: sklearn.base.BaseEstimator,
        *,
        name: str | None = None,
    ) -> None:
        if not hasattr(transformer, "fit_transform"):
            raise ValueError(
                f"{transformer} does not have a fit_transform method for fit mode"
            )

        if not hasattr(transformer, "transform"):
            raise ValueError(
                f"{transformer} does not have a transform method for predict mode"
            )

        super().__init__(name=name)

        self.transformer = transformer

    def _get_fit_args(self, input: I) -> dict[str, Any]:
        return {}

    def _get_transform_args(self, input: I) -> dict[str, Any]:
        return {}

    def _fit_transform(self, input: I) -> SKResultTypes:
        args = self._get_fit_args(input)

        return self.transformer.fit_transform(input.X, **args)  # type: ignore[misc]

    def _transform(self, input: I) -> SKResultTypes:
        args = self._get_transform_args(input)

        return self.transformer.transform(input.X, **args)  # type: ignore[misc]


class SKSupervisedTransformer(Generic[I, O], SKTransformer[I, O]):
    def _get_fit_args(self, input: I) -> dict[str, Any]:
        assert isinstance(input, (XyDataFrame | XySeries))
        return {"y": input.y}

    def _get_transform_args(self, input: I) -> dict[str, Any]:
        assert isinstance(input, (XyDataFrame | XySeries))
        return {"y": input.y}


class XyDataFrameTransformer(SKSupervisedTransformer[XyDataFrame, XyDataFrame]):
    pass


class XDataFrameTransformer(SKTransformer[XDataFrame, XDataFrame]):
    pass


class XySeriesTransformer(SKSupervisedTransformer[XySeries, XySeries]):
    pass


class XSeriesTransformer(SKTransformer[XSeries, XSeries]):
    pass


class XySeriesToDataFrameTransformer(SKSupervisedTransformer[XySeries, XyDataFrame]):
    pass


class XSeriesToDataFrameTransformer(SKTransformer[XSeries, XDataFrame]):
    pass
