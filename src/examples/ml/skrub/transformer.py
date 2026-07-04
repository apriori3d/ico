from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any, Generic, cast

import sklearn  # type: ignore[import-untyped]

from examples.ml.skrub.base import (
    SKOperator,
)
from examples.ml.skrub.data import (
    PandaXDataFrame,
    PandaXSeries,
    PandaXyDataFrame,
    PandaXySeries,
    TDataFrame,
    TSeries,
    XyDataFrame,
    XySeries,
    select_column_x,
    select_columns_x,
    wrap_result_dataframe_x,
    wrap_result_dataframe_xy,
    wrap_result_series_x,
)
from examples.ml.skrub.describe.plan.utils import (
    setup_renderer_show_args,
    setup_renderer_show_estimator,
)
from ico.core.node import IcoNodeProtocol
from ico.core.operator import I, O


class SKBaseTransformer(
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
                return self._fit_transform(input)

            case "predict":
                return self._transform(input)

    @abc.abstractmethod
    def _fit_transform(self, input: I) -> O: ...

    @abc.abstractmethod
    def _transform(self, input: I) -> O: ...


@setup_renderer_show_estimator()
class SKTransformer(
    Generic[I, O],
    SKBaseTransformer[I, O],
):
    transformer: sklearn.base.BaseEstimator
    fit_args: dict[str, Any]
    transform_args: dict[str, Any]

    def __init__(
        self,
        transformer: sklearn.base.BaseEstimator,
        *,
        fit_args: dict[str, Any] | None = None,
        transform_args: dict[str, Any] | None = None,
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
        self.fit_args = fit_args or {}
        self.transform_args = transform_args or {}


@setup_renderer_show_estimator()
class DataFrameTransformer(
    Generic[TDataFrame],
    SKTransformer[TDataFrame, TDataFrame],
):
    def _fit_transform(self, input: TDataFrame) -> TDataFrame:
        if isinstance(input, XyDataFrame):
            x1 = self.transformer.fit_transform(input.X, y=input.y, **self.fit_args)  # type: ignore[misc]
            return cast(
                TDataFrame, wrap_result_dataframe_xy(cast(PandaXyDataFrame, input), x1)
            )

        x1 = self.transformer.fit_transform(input.X, **self.fit_args)  # type: ignore[misc]
        return cast(TDataFrame, wrap_result_dataframe_x(input, x1))

    def _transform(self, input: TDataFrame) -> TDataFrame:
        if isinstance(input, XyDataFrame):
            x1 = self.transformer.transform(input.X, y=input.y, **self.transform_args)  # type: ignore[misc]
            return cast(
                TDataFrame, wrap_result_dataframe_xy(cast(PandaXyDataFrame, input), x1)
            )

        x1 = self.transformer.transform(input.X, **self.transform_args)  # type: ignore[misc]
        return cast(TDataFrame, wrap_result_dataframe_x(input, x1))


XDataFrameTransformer = DataFrameTransformer[PandaXDataFrame]
XyDataFrameTransformer = DataFrameTransformer[PandaXyDataFrame]


@setup_renderer_show_estimator()
class SeriesTransformer(
    Generic[TSeries],
    SKTransformer[TSeries, TSeries],
):
    def _fit_transform(self, input: TSeries) -> TSeries:
        if isinstance(input, XySeries):
            x1 = self.transformer.fit_transform(input.X, y=input.y, **self.fit_args)  # type: ignore[misc]
            return cast(TSeries, wrap_result_series_x(cast(PandaXySeries, input), x1))

        x1 = self.transformer.fit_transform(input.X, **self.fit_args)  # type: ignore[misc]
        return cast(TSeries, wrap_result_series_x(input, x1))

    def _transform(self, input: TSeries) -> TSeries:
        if isinstance(input, XySeries):
            x1 = self.transformer.transform(input.X, y=input.y, **self.transform_args)  # type: ignore[misc]
            return cast(TSeries, wrap_result_series_x(cast(PandaXySeries, input), x1))

        x1 = self.transformer.transform(input.X, **self.transform_args)  # type: ignore[misc]
        return cast(TSeries, wrap_result_series_x(input, x1))


XSeriesTransformer = SeriesTransformer[PandaXSeries]
XySeriesTransformer = SeriesTransformer[PandaXySeries]


@setup_renderer_show_estimator()
class SeriesToDataFrameTransformer(
    Generic[TSeries, TDataFrame],
    SKTransformer[TSeries, TDataFrame],
):
    def _fit_transform(self, input: TSeries) -> TDataFrame:
        if isinstance(input, XySeries):
            x1 = self.transformer.fit_transform(input.X, y=input.y, **self.fit_args)  # type: ignore[misc]
            return cast(
                TDataFrame, wrap_result_series_x(cast(PandaXySeries, input), x1)
            )

        x1 = self.transformer.fit_transform(input.X, **self.fit_args)  # type: ignore[misc]
        return cast(TDataFrame, wrap_result_series_x(input, x1))

    def _transform(self, input: TSeries) -> TDataFrame:
        if isinstance(input, XySeries):
            x1 = self.transformer.transform(input.X, y=input.y, **self.transform_args)  # type: ignore[misc]
            return cast(
                TDataFrame, wrap_result_series_x(cast(PandaXySeries, input), x1)
            )

        x1 = self.transformer.transform(input.X, **self.transform_args)  # type: ignore[misc]
        return cast(TDataFrame, wrap_result_series_x(input, x1))


XSeriesToDataFrameTransformer = SeriesToDataFrameTransformer[
    PandaXSeries, PandaXDataFrame
]


@setup_renderer_show_args("column")
class ColumnExtractor(
    Generic[TDataFrame, TSeries],
    SKBaseTransformer[TDataFrame, TSeries],
):
    column: str

    def __init__(self, column: str, name: str | None = None) -> None:
        super().__init__(name=name)
        self.column = column

    def _fit_transform(self, input: TDataFrame) -> TSeries:
        return cast(TSeries, select_column_x(input, self.column))

    def _transform(self, input: TDataFrame) -> TSeries:
        return self._fit_transform(input)


XColumnExtractor = ColumnExtractor[PandaXDataFrame, PandaXSeries]
XyColumnExtractor = ColumnExtractor[PandaXyDataFrame, PandaXySeries]


@setup_renderer_show_args("column")
class ColumnsExtractor(
    SKBaseTransformer[PandaXDataFrame, PandaXDataFrame],
):
    columns_pattern: Any

    def __init__(self, columns_pattern: Any, name: str | None = None) -> None:
        super().__init__(name=name)
        self.columns_pattern = columns_pattern

    def _fit_transform(self, input: PandaXDataFrame) -> PandaXDataFrame:
        return select_columns_x(input, self.columns_pattern)

    def _transform(self, input: PandaXDataFrame) -> PandaXDataFrame:
        return self._fit_transform(input)
