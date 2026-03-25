from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Generic, Literal, TypeAlias, overload

import pandas as pd  # type: ignore[import-untyped]
import scipy.sparse as sp  # type: ignore[import-untyped]

from ico.core.operator import I, IcoOperator, O

SKMode = Literal["fit", "predict"]


class ModeMixin:
    mode: SKMode = "fit"

    def fit_mode(self) -> None:
        self.mode = "fit"

    def predict_mode(self) -> None:
        self.mode = "predict"


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

SKResultTypes: TypeAlias = pd.Series | pd.DataFrame | sp.spmatrix


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


def wrap_result(input: SKData, x1: SKResultTypes) -> SKData:  # type: ignore[misc]
    return wrap_result_data(input, x1)


def wrap_result_data(input: SKData, x1: SKResultTypes) -> SKData:
    match (input, x1):
        case (XyDataFrame(y=y), sp.spmatrix() as sparse_matrix):
            return XyDataFrame(X=pd.DataFrame.sparse.from_spmatrix(sparse_matrix), y=y)

        case (XDataFrame(), sp.spmatrix() as sparse_matrix):
            return XDataFrame(X=pd.DataFrame.sparse.from_spmatrix(sparse_matrix))

        case (XySeries(y=y), sp.spmatrix() as sparse_matrix):
            return XyDataFrame(X=pd.DataFrame.sparse.from_spmatrix(sparse_matrix), y=y)

        case (XSeries(), sp.spmatrix() as sparse_matrix):
            return XDataFrame(X=pd.DataFrame.sparse.from_spmatrix(sparse_matrix))

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


class SKBaseEstimator(Generic[I, O], IcoOperator[I, O], abc.ABC, ModeMixin):
    def __init__(
        self,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(self._estimator_fn, name=name)

    @abc.abstractmethod
    def _estimator_fn(self, input: I) -> O: ...
