from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any, Generic, cast, overload

import pandas as pd  # type: ignore[import-untyped]
import scipy.sparse as sp  # type: ignore[import-untyped]
import sklearn  # type: ignore[import-untyped]

from examples.ml.skrub.base import (
    SKOperator,
)
from examples.ml.skrub.data import (
    AnyDataFrame,
    AnySeries,
    AnyXDataFrame,
    AnyXSeries,
    AnyXyDataFrame,
    AnyXySeries,
    XDataFrame,
    XyDataFrame,
    select_column_x,
    wrap_result_dataframe_x,
    wrap_result_series_x,
)
from examples.ml.skrub.describe.plan.utils import (
    setup_renderer_show_args,
    setup_renderer_show_estimator,
)
from ico.core.node import IcoNodeProtocol
from ico.core.operator import I, O
from ico.core.signature import IcoSignature


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
class SKTransformer(Generic[I, O], SKBaseTransformer[I, O]):
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
class DataFrameTransformer(SKTransformer[AnyDataFrame, AnyDataFrame]):
    @overload
    def _fit_transform(self, input: AnyXyDataFrame) -> AnyXyDataFrame: ...

    @overload
    def _fit_transform(self, input: AnyXDataFrame) -> AnyXDataFrame: ...

    def _fit_transform(self, input: AnyXDataFrame) -> AnyXDataFrame:
        result = cast(Any, self.transformer.fit_transform(input.X, **self.fit_args))  # type: ignore[misc]
        return wrap_result_dataframe_x(input, result)

    @overload
    def _transform(self, input: AnyXyDataFrame) -> AnyXyDataFrame: ...

    @overload
    def _transform(self, input: AnyXDataFrame) -> AnyXDataFrame: ...

    def _transform(self, input: AnyXDataFrame) -> AnyXDataFrame:
        result = cast(Any, self.transformer.transform(input.X, **self.transform_args))  # type: ignore[misc]
        return wrap_result_dataframe_x(input, result)

    @property
    def signature(self) -> IcoSignature:
        return IcoSignature(i=AnyXDataFrame, c=None, o=AnyXDataFrame)


@setup_renderer_show_estimator()
class SeriesTransformer(SKTransformer[AnySeries, AnySeries]):
    @overload
    def _fit_transform(self, input: AnyXySeries) -> AnyXySeries: ...

    @overload
    def _fit_transform(self, input: AnyXSeries) -> AnyXSeries: ...

    def _fit_transform(self, input: AnyXSeries) -> AnyXSeries:
        result = cast(Any, self.transformer.fit_transform(input.X, **self.fit_args))  # type: ignore[misc]
        return wrap_result_series_x(input, result)

    @overload
    def _transform(self, input: AnyXySeries) -> AnyXySeries: ...

    @overload
    def _transform(self, input: AnyXSeries) -> AnyXSeries: ...

    def _transform(self, input: AnyXSeries) -> AnyXSeries:
        result = cast(Any, self.transformer.transform(input.X, **self.transform_args))  # type: ignore[misc]
        return wrap_result_series_x(input, result)

    @property
    def signature(self) -> IcoSignature:
        return IcoSignature(i=AnyXSeries, c=None, o=AnyXSeries)


@setup_renderer_show_estimator()
class SeriesToDataFrameTransformer(SKTransformer[AnySeries, AnyDataFrame]):
    @overload
    def _fit_transform(self, input: AnyXySeries) -> AnyXyDataFrame: ...

    @overload
    def _fit_transform(self, input: AnyXSeries) -> AnyXDataFrame: ...

    def _fit_transform(self, input: AnyXSeries) -> AnyXDataFrame:
        result = cast(Any, self.transformer.fit_transform(input.X, **self.fit_args))  # type: ignore[misc]
        return wrap_result_dataframe_x(input, result)

    @overload
    def _transform(self, input: AnyXySeries) -> AnyXyDataFrame: ...

    @overload
    def _transform(self, input: AnyXSeries) -> AnyXDataFrame: ...

    def _transform(self, input: AnyXSeries) -> AnyXDataFrame:
        result = cast(Any, self.transformer.transform(input.X, **self.transform_args))  # type: ignore[misc]
        return wrap_result_dataframe_x(input, result)

    @property
    def signature(self) -> IcoSignature:
        return IcoSignature(i=AnyXSeries, c=None, o=AnyXDataFrame)


@setup_renderer_show_estimator()
class SeriesToDataFrameSparseTransformer(
    SKTransformer[
        AnySeries,
        XDataFrame[sp.spmatrix, sp.spmatrix]
        | XyDataFrame[sp.spmatrix, sp.spmatrix, pd.Series],
    ]
):
    @overload
    def _fit_transform(
        self, input: AnyXySeries
    ) -> XyDataFrame[sp.spmatrix, sp.spmatrix, pd.Series]: ...

    @overload
    def _fit_transform(
        self, input: AnyXSeries
    ) -> XDataFrame[sp.spmatrix, sp.spmatrix]: ...

    def _fit_transform(self, input: AnyXSeries) -> XDataFrame[sp.spmatrix, sp.spmatrix]:
        x1 = cast(Any, self.transformer.fit_transform(input.X, **self.fit_args))  # type: ignore[misc]
        output = wrap_result_dataframe_x(input, x1)
        return output

    @overload
    def _transform(
        self, input: AnyXySeries
    ) -> XyDataFrame[sp.spmatrix, sp.spmatrix, pd.Series]: ...

    @overload
    def _transform(self, input: AnyXSeries) -> XDataFrame[sp.spmatrix, sp.spmatrix]: ...

    def _transform(self, input: AnyXSeries) -> XDataFrame[sp.spmatrix, sp.spmatrix]:
        result = cast(Any, self.transformer.transform(input.X, **self.transform_args))  # type: ignore[misc]
        return wrap_result_dataframe_x(input, result)

    @property
    def signature(self) -> IcoSignature:
        return IcoSignature(
            i=AnyXSeries, c=None, o=XDataFrame[sp.spmatrix, sp.spmatrix]
        )


@setup_renderer_show_args("column")
class ColumnExtractor(
    SKBaseTransformer[
        AnyXDataFrame | AnyXyDataFrame,
        AnyXSeries | AnyXySeries,
    ],
):
    column: str

    def __init__(self, column: str, name: str | None = None) -> None:
        super().__init__(name=name)
        self.column = column

    @overload
    def _fit_transform(self, input: AnyXyDataFrame) -> AnyXySeries: ...

    @overload
    def _fit_transform(self, input: AnyXDataFrame) -> AnyXSeries: ...

    def _fit_transform(self, input: AnyXDataFrame) -> AnyXSeries:
        return select_column_x(input, self.column)

    @overload
    def _transform(self, input: AnyXyDataFrame) -> AnyXySeries: ...

    @overload
    def _transform(self, input: AnyXDataFrame) -> AnyXSeries: ...

    def _transform(self, input: AnyXDataFrame) -> AnyXSeries:
        return self._fit_transform(input)

    @property
    def signature(self) -> IcoSignature:
        return IcoSignature(i=AnyXDataFrame, c=None, o=AnyXSeries)
