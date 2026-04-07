from __future__ import annotations

import abc
from typing import Any, Generic, cast, overload

import pandas as pd  # type: ignore[import-untyped]
import sklearn  # type: ignore[import-untyped]

from examples.ml.skrub.base import (
    SKOperator,
)
from examples.ml.skrub.data import (
    XDataFrame,
    XSeries,
    XyDataFrame,
    XySeries,
    XYSource,
    select_column_x,
    wrap_result_dataframe_x,
    wrap_result_dataframe_xy,
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


class XDataFrameTransformer(
    SKTransformer[
        XDataFrame[Any, Any] | XyDataFrame[Any, Any, Any],
        XDataFrame[Any, Any] | XyDataFrame[Any, Any, Any],
    ],
):
    @overload
    def _fit_transform(
        self, input: XyDataFrame[Any, Any, Any]
    ) -> XyDataFrame[Any, Any, Any]: ...

    @overload
    def _fit_transform(self, input: XDataFrame[Any, Any]) -> XDataFrame[Any, Any]: ...

    def _fit_transform(self, input: XDataFrame[Any, Any]) -> XDataFrame[Any, Any]:
        result = cast(Any, self.transformer.fit_transform(input.X, **self.fit_args))  # type: ignore[misc]
        return wrap_result_dataframe_x(input, result)

    @overload
    def _transform(
        self, input: XyDataFrame[Any, Any, Any]
    ) -> XyDataFrame[Any, Any, Any]: ...

    @overload
    def _transform(self, input: XDataFrame[Any, Any]) -> XDataFrame[Any, Any]: ...

    def _transform(self, input: XDataFrame[Any, Any]) -> XDataFrame[Any, Any]:
        result = cast(Any, self.transformer.transform(input.X, **self.transform_args))  # type: ignore[misc]
        return wrap_result_dataframe_x(input, result)


class XyDataFrameTransformer(
    SKTransformer[XyDataFrame[Any, Any, Any], XyDataFrame[Any, Any, Any]],
):
    def _fit_transform(
        self, input: XyDataFrame[Any, Any, Any]
    ) -> XyDataFrame[Any, Any, Any]:
        result = cast(
            Any,
            self.transformer.fit_transform(input.X, y=input.y, **self.fit_args),  # type: ignore[misc]
        )
        return wrap_result_dataframe_xy(input, result)

    def _transform(
        self, input: XyDataFrame[Any, Any, Any]
    ) -> XyDataFrame[Any, Any, Any]:
        result = cast(
            Any,
            self.transformer.transform(input.X, y=input.y, **self.transform_args),  # type: ignore[misc]
        )
        return wrap_result_dataframe_xy(input, result)


class SKColumnExtractor(
    SKBaseTransformer[
        XDataFrame[Any, Any] | XyDataFrame[Any, Any, Any],
        XSeries[Any] | XySeries[Any, Any],
    ],
):
    column: str

    def __init__(self, column: str, name: str | None = None) -> None:
        super().__init__(name=name)
        self.column = column

    @overload
    def _fit_transform(
        self, input: XyDataFrame[Any, Any, Any]
    ) -> XySeries[Any, Any]: ...

    @overload
    def _fit_transform(self, input: XDataFrame[Any, Any]) -> XSeries[Any]: ...

    def _fit_transform(self, input: XDataFrame[Any, Any]) -> XSeries[Any]:
        return select_column_x(input, self.column)

    @overload
    def _transform(self, input: XyDataFrame[Any, Any, Any]) -> XySeries[Any, Any]: ...

    @overload
    def _transform(self, input: XDataFrame[Any, Any]) -> XSeries[Any]: ...

    def _transform(self, input: XDataFrame[Any, Any]) -> XSeries[Any]:
        return self._fit_transform(input)

    @property
    def signature(self) -> IcoSignature:
        return IcoSignature(i=XDataFrame[Any, Any], c=None, o=XSeries[Any])


def load_orders_xy() -> XyDataFrame[pd.DataFrame, pd.Series, pd.Series]:
    orders = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "customer_id": [101, 102, 103, 104, 105],
            "amount": [250.0, 150.0, 300.0, 200.0, 100.0],
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
        }
    )
    return XDataFrame(X=orders)


if __name__ == "__main__":
    from examples.ml.skrub.data import XDataFrame

    source = XYSource(load_orders_xy)
    select_column = SKColumnExtractor("customer_id")

    pipeline = source | select_column
    pipeline.describe()

    pipeline.fit_mode()
    resut = pipeline(None)
    print(resut)
