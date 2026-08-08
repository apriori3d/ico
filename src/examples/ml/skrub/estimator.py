from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any, Generic, cast

import sklearn  # type: ignore[import-untyped]

from examples.ml.skrub.base import (
    SKOperator,
)
from examples.ml.skrub.data import (
    PandaXyDataFrame,
    wrap_result_dataframe_xy,
)
from examples.ml.skrub.describe.plan.utils import (
    setup_renderer_show_estimator,
)
from ico.core.node import IcoNodeProtocol
from ico.core.operator import I, O


class SKBaseEstimator(
    Generic[I, O],
    SKOperator[I, O],
    abc.ABC,
):
    def __init__(
        self,
        *,
        name: str | None = None,
        children: Sequence[IcoNodeProtocol] | None = None,
    ) -> None:
        super().__init__(name=name, children=children)

    def _estimator_fn(self, input: I) -> O:
        match self.mode:
            case "fit":
                return self._fit(input)

            case "predict":
                return self._predict(input)

    @abc.abstractmethod
    def _fit(self, input: I) -> O: ...

    @abc.abstractmethod
    def _predict(self, input: I) -> O: ...


@setup_renderer_show_estimator()
class SKEstimator(Generic[I, O], SKBaseEstimator[I, O]):
    estimator: sklearn.base.BaseEstimator
    fit_args: dict[str, Any]
    predict_args: dict[str, Any]

    def __init__(
        self,
        estimator: sklearn.base.BaseEstimator,
        *,
        fit_args: dict[str, Any] | None = None,
        predict_args: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        if not hasattr(estimator, "fit"):
            raise ValueError(f"{estimator} does not have a fit method for fit mode")

        if not hasattr(estimator, "predict"):
            raise ValueError(
                f"{estimator} does not have a predict method for predict mode"
            )

        super().__init__(name=name)

        self.estimator = estimator
        self.fit_args = fit_args or {}
        self.predict_args = predict_args or {}


@setup_renderer_show_estimator()
class DataFrameEstimator(
    SKEstimator[PandaXyDataFrame, PandaXyDataFrame],
):
    def _fit(self, input: PandaXyDataFrame) -> PandaXyDataFrame:
        result = cast(Any, self.estimator.fit(input.X, y=input.y, **self.fit_args))  # type: ignore[misc]
        return cast(PandaXyDataFrame, wrap_result_dataframe_xy(input, result))

    def _predict(self, input: PandaXyDataFrame) -> PandaXyDataFrame:
        result = cast(Any, self.estimator.predict(input.X, **self.predict_args))  # type: ignore[misc]
        return cast(PandaXyDataFrame, wrap_result_dataframe_xy(input, result))
