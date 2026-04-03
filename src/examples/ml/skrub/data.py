from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    TypeAlias,
    TypeGuard,
    TypeVar,
    overload,
)

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import scipy.sparse as sp  # type: ignore[import-untyped]

from examples.ml.skrub.base import SKChain, SKOperatorProtocol
from ico.core.operator import O2, IcoOperator, IcoOperatorProtocol
from ico.core.signature import IcoSignature
from ico.core.signature_utils import infer_from_callable

TTable = TypeVar("TTable", bound=pd.DataFrame)
TColumn = TypeVar("TColumn", bound=pd.Series)
TTarget = TypeVar("TTarget", bound=pd.Series)
TColumnCov = TypeVar("TColumnCov", covariant=True)


@dataclass(slots=True)
class XDataFrame(Generic[TTable, TColumn]):
    X: TTable


@dataclass(slots=True)
class XyDataFrame(Generic[TTable, TColumn, TTarget], XDataFrame[TTable, TColumn]):
    y: TTarget


@dataclass(slots=True)
class XSeries(Generic[TColumn]):
    X: TColumn


@dataclass(slots=True)
class XySeries(Generic[TColumn, TTarget], XSeries[TColumn]):
    y: TTarget


AnyDataFrameType = XyDataFrame[Any, Any, Any] | XDataFrame[Any, Any]
AnySeriesType = XySeries[Any, Any] | XSeries[Any]
AnyDataType = AnyDataFrameType | AnySeriesType

TData = TypeVar("TData", bound=AnyDataType)
TDataFrame = TypeVar("TDataFrame", bound=AnyDataFrameType)
TSeries = TypeVar("TSeries", bound=AnySeriesType)


SKDataFrameResultTypes: TypeAlias = pd.DataFrame | sp.spmatrix | np.ndarray[Any, Any]
SKResultTypes: TypeAlias = pd.Series | pd.DataFrame | sp.spmatrix | np.ndarray[Any, Any]


def _is_xy_dataframe(
    input: AnyDataFrameType,
) -> TypeGuard[XyDataFrame[Any, Any, Any]]:
    return isinstance(input, XyDataFrame)


def _is_same_table_type(expected: TTable, data: Any) -> TypeGuard[TTable]:
    return isinstance(data, type(expected))


def _is_panda_dataframe(data: Any) -> TypeGuard[pd.DataFrame]:
    return isinstance(data, pd.DataFrame)


def _is_ndarray(data: Any) -> TypeGuard[np.ndarray[Any, Any]]:
    return isinstance(data, np.ndarray)


def _is_panda_series(data: Any) -> TypeGuard[pd.Series]:
    return isinstance(data, pd.Series)


@overload
def wrap_result_dataframe_x(
    input: XyDataFrame[TTable, TColumn, TTarget], x1: TTable
) -> XyDataFrame[TTable, TColumn, TTarget]: ...


@overload
def wrap_result_dataframe_x(
    input: XDataFrame[TTable, TColumn], x1: TTable
) -> XDataFrame[TTable, TColumn]: ...


def wrap_result_dataframe_x(
    input: XyDataFrame[Any, Any, Any] | XDataFrame[Any, Any],
    x1: Any,
) -> XyDataFrame[Any, Any, Any] | XDataFrame[Any, Any]:
    match input, x1:
        case input, x1 if (
            _is_xy_dataframe(input)
            and _is_panda_dataframe(input.X)
            and _is_panda_series(input.y)
            and _is_panda_dataframe(x1)
        ):
            return XyDataFrame[pd.DataFrame, pd.Series, pd.Series](X=x1, y=input.y)

        case input, x1 if (
            _is_xy_dataframe(input)
            and _is_panda_dataframe(input.X)
            and _is_panda_series(input.y)
            and _is_ndarray(x1)
        ):
            return XyDataFrame[pd.DataFrame, pd.Series, pd.Series](
                X=pd.DataFrame(x1), y=input.y
            )

        case XDataFrame() as input, x1 if (
            _is_panda_dataframe(input.X) and _is_panda_dataframe(x1)
        ):
            return XDataFrame[pd.DataFrame, pd.Series](X=x1)

        case XDataFrame() as input, x1 if (
            _is_panda_dataframe(input.X) and _is_ndarray(x1)
        ):
            return XDataFrame[pd.DataFrame, pd.Series](X=pd.DataFrame(x1))

        case _:
            raise TypeError(
                "Unsupported input/result combination: "
                f"input={type(input).__name__}, result={type(x1).__name__}"
            )


def wrap_result_dataframe_xy(
    input: XyDataFrame[TTable, TColumn, TTarget],
    x1: Any,
) -> XyDataFrame[TTable, TColumn, TTarget]:
    match input, x1:
        case input, x1 if _is_xy_dataframe(input) and _is_same_table_type(input.X, x1):
            return XyDataFrame[TTable, TColumn, TTarget](X=x1, y=input.y)

        case _:
            raise TypeError(
                "Unsupported input/result combination: "
                f"input={type(input).__name__}, result={type(x1).__name__}"
            )


# @overload
# def select_column(frame: XyDataFrame, column: str) -> XySeries: ...


# @overload
# def select_column(frame: XDataFrame, column: str) -> XSeries: ...


# def select_column(frame: XyDataFrame | XDataFrame, column: str) -> XySeries | XSeries:
#     match frame:
#         case XyDataFrame(y=y) as df:
#             return XySeries(X=cast(pd.Series, cast(Any, df.X)[column]), y=y)

#         case XDataFrame() as df:
#             return XSeries(X=cast(pd.Series, cast(Any, df.X)[column]))


# def sparse_to_dataframe(matrix: sp.spmatrix) -> pd.DataFrame:
#     return cast(
#         pd.DataFrame,
#         pd.DataFrame.sparse.from_spmatrix(matrix),  # pyright: ignore[reportUnknownMemberType]
#     )


# def ndarray_to_dataframe(array: np.ndarray[Any, Any]) -> pd.DataFrame:
#     return pd.DataFrame(array)


# def is_dataframe_result(x1: SKResultTypes) -> TypeGuard[SKDataFrameResultTypes]:
#     return isinstance(x1, pd.DataFrame | sp.spmatrix | np.ndarray)


# def get_data_shape(data: AnyDataType) -> tuple[int, int]:
#     match data:
#         case XyDataFrame() as df:
#             return df.X.shape
#         case XDataFrame() as df:
#             return df.X.shape
#         case XySeries() as series:
#             return (series.X.shape[0], 1)
#         case XSeries() as series:
#             return (series.X.shape[0], 1)


class XYSource(
    Generic[TTable, TColumn, TTarget],
    IcoOperator[None, XyDataFrame[TTable, TColumn, TTarget]],
):
    provider: Callable[[], XyDataFrame[TTable, TColumn, TTarget]]

    def __init__(
        self,
        provider: Callable[[], XyDataFrame[TTable, TColumn, TTarget]],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(self._source_fn, name=name)
        self.provider = provider

    def _source_fn(self, _: None) -> XyDataFrame[TTable, TColumn, TTarget]:
        return self.provider()

    @property
    def signature(self) -> IcoSignature:
        signature = super().signature

        if not signature.infered:
            signature = infer_from_callable(self.provider) or signature

        signature = (
            IcoSignature(i=type(None), c=None, o=type[Any])
            if not signature.infered
            else signature
        )
        return signature

    @overload
    def __or__(
        self, other: SKOperatorProtocol[XyDataFrame[TTable, TColumn, TTarget], O2]
    ) -> SKOperatorProtocol[None, O2]: ...

    @overload
    def __or__(
        self, other: IcoOperatorProtocol[XyDataFrame[TTable, TColumn, TTarget], O2]
    ) -> SKOperatorProtocol[None, O2]: ...

    def __or__(
        self, other: IcoOperatorProtocol[XyDataFrame[TTable, TColumn, TTarget], O2]
    ) -> SKOperatorProtocol[None, O2]:
        return SKChain(self, other)


class XSource(
    Generic[TTable, TColumn],
    IcoOperator[None, XDataFrame[TTable, TColumn]],
):
    provider: Callable[[], XDataFrame[TTable, TColumn]]

    def __init__(
        self,
        provider: Callable[[], XDataFrame[TTable, TColumn]],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(self._source_fn, name=name)
        self.provider = provider

    def _source_fn(self, _: None) -> XDataFrame[TTable, TColumn]:
        return self.provider()

    @property
    def signature(self) -> IcoSignature:
        signature = super().signature

        if not signature.infered:
            signature = infer_from_callable(self.provider) or signature

        signature = (
            IcoSignature(i=type(None), c=None, o=type[Any])
            if not signature.infered
            else signature
        )
        return signature

    @overload
    def __or__(
        self, other: SKOperatorProtocol[XDataFrame[TTable, TColumn], O2]
    ) -> SKOperatorProtocol[None, O2]: ...

    @overload
    def __or__(
        self, other: IcoOperatorProtocol[XDataFrame[TTable, TColumn], O2]
    ) -> SKOperatorProtocol[None, O2]: ...

    def __or__(
        self, other: IcoOperatorProtocol[XDataFrame[TTable, TColumn], O2]
    ) -> SKOperatorProtocol[None, O2]:
        return SKChain(self, other)


class SKSink(
    Generic[TTable, TColumn, TTarget],
    IcoOperator[
        XDataFrame[TTable, TColumn] | XyDataFrame[TTable, TColumn, TTarget], None
    ],
):
    consumer: Callable[
        [XDataFrame[TTable, TColumn] | XyDataFrame[TTable, TColumn, TTarget]], None
    ]

    def __init__(
        self,
        consumer: Callable[
            [XDataFrame[TTable, TColumn] | XyDataFrame[TTable, TColumn, TTarget]], None
        ],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(self._sink_fn, name=name)
        self.consumer = consumer

    def _sink_fn(
        self, input: XDataFrame[TTable, TColumn] | XyDataFrame[TTable, TColumn, TTarget]
    ) -> None:
        self.consumer(input)

    @property
    def signature(self) -> IcoSignature:
        signature = super().signature
        if not signature.infered:
            signature = infer_from_callable(self.consumer) or signature

        signature = (
            IcoSignature(i=type[Any], c=None, o=type(None))
            if not signature or not signature.infered
            else signature
        )
        return signature
