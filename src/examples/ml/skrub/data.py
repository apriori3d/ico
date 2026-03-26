from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    TypeAlias,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import scipy.sparse as sp  # type: ignore[import-untyped]

from ico.core.operator import IcoOperator
from ico.core.signature import IcoSignature
from ico.core.signature_utils import infer_from_callable


@dataclass(slots=True)
class XDataFrame:
    X: pd.DataFrame


@dataclass(slots=True)
class XyDataFrame(XDataFrame):
    y: pd.Series


@dataclass(slots=True)
class XSeries:
    X: pd.Series


@dataclass(slots=True)
class XySeries(XSeries):
    y: pd.Series


SKDataFrame = XyDataFrame | XDataFrame
SKSeries = XySeries | XSeries
SKData = SKDataFrame | SKSeries

SKDataFrameResultTypes: TypeAlias = pd.DataFrame | sp.spmatrix | np.ndarray[Any, Any]
SKResultTypes: TypeAlias = pd.Series | SKDataFrameResultTypes


@overload
def wrap_result(input: XyDataFrame, x1: pd.DataFrame) -> XyDataFrame: ...


@overload
def wrap_result(input: XDataFrame, x1: pd.DataFrame) -> XDataFrame: ...


@overload
def wrap_result(input: XySeries, x1: pd.Series) -> XySeries: ...


@overload
def wrap_result(input: XSeries, x1: pd.Series) -> XSeries: ...


@overload
def wrap_result(input: XySeries, x1: pd.DataFrame) -> XyDataFrame: ...


@overload
def wrap_result(input: XSeries, x1: pd.DataFrame) -> XDataFrame: ...


@overload
def wrap_result(input: XyDataFrame, x1: sp.spmatrix) -> XyDataFrame: ...


@overload
def wrap_result(input: XDataFrame, x1: sp.spmatrix) -> XDataFrame: ...


@overload
def wrap_result(input: XySeries, x1: sp.spmatrix) -> XyDataFrame: ...


@overload
def wrap_result(input: XSeries, x1: sp.spmatrix) -> XDataFrame: ...


@overload
def wrap_result(input: XyDataFrame, x1: np.ndarray[Any, Any]) -> XyDataFrame: ...


@overload
def wrap_result(input: XDataFrame, x1: np.ndarray[Any, Any]) -> XDataFrame: ...


@overload
def wrap_result(input: XySeries, x1: np.ndarray[Any, Any]) -> XyDataFrame: ...


@overload
def wrap_result(input: XSeries, x1: np.ndarray[Any, Any]) -> XDataFrame: ...


def wrap_result(input: SKData, x1: SKResultTypes) -> SKData:  # type: ignore[misc]
    return wrap_result_data(input, x1)


def wrap_result_data(input: SKData, x1: SKResultTypes) -> SKData:
    match (input, x1):
        case (XyDataFrame(y=y), np.ndarray() as ndarray):
            return XyDataFrame(X=ndarray_to_dataframe(ndarray), y=y)

        case (XDataFrame(), np.ndarray() as ndarray):
            return XDataFrame(X=ndarray_to_dataframe(ndarray))

        case (XySeries(y=y), np.ndarray() as ndarray):
            return XyDataFrame(X=ndarray_to_dataframe(ndarray), y=y)

        case (XSeries(), np.ndarray() as ndarray):
            return XDataFrame(X=ndarray_to_dataframe(ndarray))

        case (XyDataFrame(y=y), sp.spmatrix() as sparse_matrix):
            return XyDataFrame(X=sparse_to_dataframe(sparse_matrix), y=y)

        case (XDataFrame(), sp.spmatrix() as sparse_matrix):
            return XDataFrame(X=sparse_to_dataframe(sparse_matrix))

        case (XySeries(y=y), sp.spmatrix() as sparse_matrix):
            return XyDataFrame(X=sparse_to_dataframe(sparse_matrix), y=y)

        case (XSeries(), sp.spmatrix() as sparse_matrix):
            return XDataFrame(X=sparse_to_dataframe(sparse_matrix))

        case (XyDataFrame(y=y), pd.DataFrame() as data_frame):
            return XyDataFrame(X=data_frame, y=y)

        case (XDataFrame(), pd.DataFrame() as data_frame):
            return XDataFrame(X=data_frame)

        case (XySeries(y=y), pd.Series() as series):
            return XySeries(X=series, y=y)

        case (XSeries(), pd.Series() as series):
            return XSeries(X=series)

        case _:
            raise TypeError(
                "Unsupported input/result combination: "
                f"input={type(input).__name__}, result={type(x1).__name__}"
            )


@overload
def select_column(frame: XyDataFrame, column: str) -> XySeries: ...


@overload
def select_column(frame: XDataFrame, column: str) -> XSeries: ...


def select_column(frame: XyDataFrame | XDataFrame, column: str) -> XySeries | XSeries:
    match frame:
        case XyDataFrame(y=y) as df:
            return XySeries(X=cast(pd.Series, cast(Any, df.X)[column]), y=y)

        case XDataFrame() as df:
            return XSeries(X=cast(pd.Series, cast(Any, df.X)[column]))


def sparse_to_dataframe(matrix: sp.spmatrix) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        pd.DataFrame.sparse.from_spmatrix(matrix),  # pyright: ignore[reportUnknownMemberType]
    )


def ndarray_to_dataframe(array: np.ndarray[Any, Any]) -> pd.DataFrame:
    return pd.DataFrame(array)


def is_dataframe_result(x1: SKResultTypes) -> TypeGuard[SKDataFrameResultTypes]:
    return isinstance(x1, pd.DataFrame | sp.spmatrix | np.ndarray)


def get_data_shape(data: SKData) -> tuple[int, int]:
    match data:
        case XyDataFrame() as df:
            return df.X.shape
        case XDataFrame() as df:
            return df.X.shape
        case XySeries() as series:
            return (series.X.shape[0], 1)
        case XSeries() as series:
            return (series.X.shape[0], 1)


IData = TypeVar("IData", bound=SKData)
OData = TypeVar("OData", bound=SKData)
ISeries = TypeVar("ISeries", bound=SKSeries)
IDataFrame = TypeVar("IDataFrame", bound=SKDataFrame)
ODataFrame = TypeVar("ODataFrame", bound=SKDataFrame)


class SKSource(Generic[OData], IcoOperator[None, OData]):
    provider: Callable[[], OData]

    def __init__(
        self,
        provider: Callable[[], OData],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(self._source_fn, name=name)
        self.provider = provider

    def _source_fn(self, _: None) -> OData:
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


class SKSink(Generic[IData], IcoOperator[IData, None]):
    consumer: Callable[[IData], None]

    def __init__(
        self,
        consumer: Callable[[IData], None],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(self._sink_fn, name=name)
        self.consumer = consumer

    def _sink_fn(self, input: IData) -> None:
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
