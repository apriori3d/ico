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
    XSource,
    XyDataFrame,
    wrap_result_dataframe_x,
    wrap_result_dataframe_xy,
)
from examples.ml.skrub.describe.plan.utils import (
    setup_renderer_show_estimator,
)
from ico.core.operator import I, O


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


# class XSeriesOperator(
#     Generic[TColumn],
#     SKOperator[XSeries[TColumn], XSeries[TColumn]],
# ):
#     pass


# class XySeriesOperator(
#     Generic[TColumn, TTarget],
#     SKOperator[XySeries[TColumn, TTarget], XySeries[TColumn, TTarget]],
# ):
#     pass


# class XDataFrameToSeriesOperator(
#     Generic[TTable, TColumn],
#     SKOperator[XDataFrame[TTable, TColumn], XSeries[TColumn]],
# ):
#     pass


# class XyDataFrameToSeriesOperator(
#     Generic[TTable, TColumn, TTarget],
#     SKOperator[XyDataFrame[TTable, TColumn, TTarget], XySeries[TColumn, TTarget]],
# ):
#     pass


# class SKTableTransformer(Generic[TDataFrame], SKTransformer[TDataFrame, TDataFrame]):
#     pass


# class XDataFrameColumnExtractor(
#     Generic[TTable, TColumn],
#     SKBaseTransformer[XDataFrame[TTable, TColumn], XSeries[TColumn]],
# ):
#     column: str

#     def __init__(self, column: str, name: str | None = None) -> None:
#         super().__init__(name=name)
#         self.column = column

#     def _fit_transform(self, input: XDataFrame[TTable, TColumn]) -> XSeries[TColumn]:
#         match input.X, input.column_type:
#             case pd.DataFrame() as df, pd.Series:
#                 return XSeries(X=cast(TColumn, df[self.column]))

#             case _:
#                 raise ValueError(
#                     f"Unsupported input type for XDataFrameColumnExtractor: "
#                     f"{type(input.X).__name__} with column type {input.column_type}"


#     def _transform(self, input: XDataFrame[TTable, TColumn]) -> XSeries[TColumn]:
#         return self._fit_transform(input)


# class XyDataFrameColumnExtractor(
#     Generic[TTable, TColumn, TTarget],
#     SKBaseTransformer[
#         XyDataFrame[TTable, TColumn, TTarget], XySeries[TColumn, TTarget]
#     ],
# ):
#     column: str

#     def __init__(self, column: str, name: str | None = None) -> None:
#         super().__init__(name=name)
#         self.column = column

#     def _fit_transform(
#         self, input: XyDataFrame[TTable, TColumn, TTarget]
#     ) -> XySeries[TColumn, TTarget]:
#         match input.X, input.column_type:
#             case pd.DataFrame() as df, pd.Series:
#                 return XySeries(X=cast(TColumn, df[self.column]), y=input.y)

#             case _:
#                 raise ValueError(
#                     f"Unsupported input type for XDataFrameColumnExtractor: "
#                     f"{type(input.X).__name__} with column type {input.column_type}"
#                 )

#     def _transform(
#         self, input: XyDataFrame[TTable, TColumn, TTarget]
#     ) -> XySeries[TColumn, TTarget]:
#         return self._fit_transform(input)


# @overload
# def extract_column(
#     source: SKOperator[I, XyDataFrame[TTable, TColumn, TTarget]],
#     column: str,
# ) -> SKChain[I, XyDataFrame[TTable, TColumn, TTarget], XySeries[TColumn, TTarget]]: ...


# @overload
# def extract_column(
#     source: SKOperator[I, XDataFrame[TTable, TColumn]],
#     column: str,
# ) -> SKChain[I, XDataFrame[TTable, TColumn], XSeries[TColumn]]: ...


# def extract_column(
#     source: SKOperator[I, XyDataFrame[TTable, TColumn, TTarget]]
#     | SKOperator[I, XDataFrame[TTable, TColumn]],
#     column: str,
# ) -> (
#     SKChain[I, XyDataFrame[TTable, TColumn, TTarget], XySeries[TColumn, TTarget]]
#     | SKChain[I, XDataFrame[TTable, TColumn], XSeries[TColumn]]
# ):
#     if source.signature.o_type is XyDataFrame:
#         extractor = XyDataFrameColumnExtractor[TTable, TColumn, TTarget](column=column)
#         xy_source = cast(SKOperator[I, XyDataFrame[TTable, TColumn, TTarget]], source)
#         return xy_source | extractor

#     if source.signature.o_type is XDataFrame:
#         extractor = XDataFrameColumnExtractor[TTable, TColumn](column=column)
#         x_source = cast(SKOperator[I, XDataFrame[TTable, TColumn]], source)
#         return x_source | extractor

#     raise ValueError(
#         f"Unsupported output type for column extraction: {source.signature.o_type}"
#     )


# class SKSupervisedTransformer(Generic[I, O], SKTransformer[I, O]):
#     def _get_fit_args(self, input: I) -> dict[str, Any]:
#         assert isinstance(input, (XyDataFrame | XySeries))
#         return {"y": input.y}

#     def _get_transform_args(self, input: I) -> dict[str, Any]:
#         assert isinstance(input, (XyDataFrame | XySeries))
#         return {"y": input.y}


# class XyDataFrameTransformer(SKSupervisedTransformer[XyDataFrame, XyDataFrame]):
#     pass


# class XDataFrameTransformer(SKTransformer[XDataFrame, XDataFrame]):
#     pass


# class XySeriesTransformer(SKSupervisedTransformer[XySeries, XySeries]):
#     pass


# class XSeriesTransformer(SKTransformer[XSeries, XSeries]):
#     pass


# class XySeriesToDataFrameTransformer(SKSupervisedTransformer[XySeries, XyDataFrame]):
#     pass


# class XSeriesToDataFrameTransformer(SKTransformer[XSeries, XDataFrame]):
#     pass


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
    from examples.ml.skrub.ops import SafeTruncatedSVD

    # source = XYSource(load_orders)
    source = XSource(load_orders_xy)
    svd = SafeTruncatedSVD()

    pipeline = source | svd
    pipeline.describe()

    pipeline.fit_mode()
    resut = pipeline(None)
    print(resut)
