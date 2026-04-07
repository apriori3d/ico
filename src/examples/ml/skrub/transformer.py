from __future__ import annotations

import abc
from typing import Any, Generic, cast, overload

import pandas as pd  # type: ignore[import-untyped]
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
    XYSource,
    select_column_x,
    wrap_result_dataframe_x,
    wrap_result_dataframe_xy,
    wrap_result_series_x,
    wrap_result_series_xy,
)
from examples.ml.skrub.describe.plan.utils import (
    setup_renderer_show_estimator,
)
from ico.core.operator import I, O
from ico.core.signature import IcoSignature


class SKBaseTransformer(Generic[I, O], SKOperator[I, O], abc.ABC):
    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name=name)

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


class XDataFrameTransformer(SKTransformer[AnyDataFrame, AnyDataFrame]):
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


class XyDataFrameTransformer(
    SKTransformer[AnyXyDataFrame, AnyXyDataFrame],
):
    def _fit_transform(self, input: AnyXyDataFrame) -> AnyXyDataFrame:
        result = cast(
            Any,
            self.transformer.fit_transform(input.X, y=input.y, **self.fit_args),  # type: ignore[misc]
        )
        return wrap_result_dataframe_xy(input, result)

    def _transform(self, input: AnyXyDataFrame) -> AnyXyDataFrame:
        result = cast(
            Any,
            self.transformer.transform(input.X, y=input.y, **self.transform_args),  # type: ignore[misc]
        )
        return wrap_result_dataframe_xy(input, result)

    @property
    def signature(self) -> IcoSignature:
        return IcoSignature(i=AnyXyDataFrame, c=None, o=AnyXyDataFrame)


class SKColumnExtractor(
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


class XSeriesTransformer(SKTransformer[AnySeries, AnySeries]):
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


class XySeriesTransformer(SKTransformer[AnyXySeries, AnyXySeries]):
    def _fit_transform(self, input: AnyXySeries) -> AnyXySeries:
        result = cast(Any, self.transformer.fit_transform(input.X, **self.fit_args))  # type: ignore[misc]
        return wrap_result_series_xy(input, result)

    def _transform(self, input: AnyXySeries) -> AnyXySeries:
        result = cast(Any, self.transformer.transform(input.X, **self.transform_args))  # type: ignore[misc]
        return wrap_result_series_xy(input, result)

    @property
    def signature(self) -> IcoSignature:
        return IcoSignature(i=AnyXySeries, c=None, o=AnyXySeries)


def load_orders_xy() -> XyDataFrame[pd.DataFrame, pd.Series, pd.Series]:
    orders = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "customer_id": [101, 102, 103, 104, 105],
            "amount": [250.0, 150.0, 300.0, 200.0, 100.0],
            "is_valid": [True, True, False, True, False],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1], name="target")  # Dummy target variable

    return XyDataFrame(X=orders, y=y)


def load_orders_x() -> XDataFrame[pd.DataFrame, pd.Series]:
    orders = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "customer_id": [101, 102, 103, 104, 105],
            "amount": [250.0, 150.0, 300.0, 200.0, 100.0],
            "is_valid": [True, True, False, True, False],
        }
    )
    return XDataFrame(X=orders)


if __name__ == "__main__":
    from examples.ml.skrub.data import XDataFrame
    from skrub._to_str import (  # type: ignore[import-untyped]
        ToStr,
    )

    source = XYSource(load_orders_xy)
    select_column = SKColumnExtractor("is_valid")
    to_str = XSeriesTransformer(ToStr())

    pipeline = source | select_column | to_str
    pipeline.describe()

    pipeline.fit_mode()
    resut = pipeline(None)
    print(resut)
