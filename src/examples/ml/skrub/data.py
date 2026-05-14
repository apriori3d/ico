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
    get_args,
    get_origin,
    overload,
)

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import scipy.sparse as sp  # type: ignore[import-untyped]

from examples.ml.skrub.base import SKChain, SKOperatorProtocol
from ico.core.operator import O2, IcoOperator, IcoOperatorProtocol
from ico.core.signature import IcoSignature
from ico.core.signature_utils import infer_from_callable, type_contain_any_typevar

TTable = TypeVar("TTable", bound=pd.DataFrame | sp.spmatrix)
TColumn = TypeVar("TColumn", bound=pd.Series | sp.spmatrix)
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


AnyXDataFrame = XDataFrame[Any, Any]
AnyXyDataFrame = XyDataFrame[Any, Any, Any]
AnyDataFrame = AnyXDataFrame | AnyXyDataFrame
AnyXSeries = XSeries[Any]
AnyXySeries = XySeries[Any, Any]
AnySeries = AnyXySeries | AnyXSeries
AnyDataType = AnyDataFrame | AnySeries

TData = TypeVar("TData", bound=AnyDataType)
TDataFrame = TypeVar("TDataFrame", bound=AnyDataFrame)
TSeries = TypeVar("TSeries", bound=AnySeries)


def is_dataframe_output_type(output_type: Any) -> bool:
    variants = get_args(AnyDataFrame)
    origins = {get_origin(t) or t for t in variants}
    candidate_origin = get_origin(output_type) or output_type
    return output_type is AnyDataFrame or candidate_origin in origins


def is_series_output_type(output_type: Any) -> bool:
    variants = get_args(AnySeries)
    origins = {get_origin(t) or t for t in variants}
    candidate_origin = get_origin(output_type) or output_type
    return output_type is AnySeries or candidate_origin in origins


SKDataFrameResultTypes: TypeAlias = pd.DataFrame | sp.spmatrix | np.ndarray[Any, Any]
SKResultTypes: TypeAlias = pd.Series | pd.DataFrame | sp.spmatrix | np.ndarray[Any, Any]


def _is_x_dataframe(
    input: AnyDataType,
) -> TypeGuard[XDataFrame[Any, Any]]:
    return isinstance(input, XDataFrame)


def _is_x_series(
    input: AnyDataType,
) -> TypeGuard[XSeries[Any]]:
    return isinstance(input, XSeries)


def _is_xy_dataframe(
    input: AnyDataType,
) -> TypeGuard[XyDataFrame[Any, Any, Any]]:
    return isinstance(input, XyDataFrame)


def _is_xy_series(
    input: AnyDataType,
) -> TypeGuard[XySeries[Any, Any]]:
    return isinstance(input, XySeries)


def _is_panda_dataframe(data: Any) -> TypeGuard[pd.DataFrame]:
    return isinstance(data, pd.DataFrame)


def _is_ndarray(data: Any) -> TypeGuard[np.ndarray[Any, Any]]:
    return isinstance(data, np.ndarray)


def _is_spmatrix(data: Any) -> TypeGuard[sp.spmatrix]:
    return isinstance(data, sp.spmatrix)


def _is_panda_series(data: Any) -> TypeGuard[pd.Series]:
    return isinstance(data, pd.Series)


@overload
def wrap_result_dataframe_x(
    input: XyDataFrame[Any, Any, Any], x1: Any
) -> XyDataFrame[Any, Any, Any]: ...


@overload
def wrap_result_dataframe_x(
    input: XDataFrame[Any, Any], x1: Any
) -> XDataFrame[Any, Any]: ...


@overload
def wrap_result_dataframe_x(input: XSeries[Any], x1: Any) -> XDataFrame[Any, Any]: ...


def wrap_result_dataframe_x(input: AnyDataType, x1: Any) -> AnyDataFrame:
    if is_xy_data(input):
        return wrap_result_dataframe_xy(input, x1)

    match input, x1:
        case input, x1 if (
            (_is_x_dataframe(input) or _is_x_series(input))
            and (_is_panda_dataframe(input.X) or _is_panda_series(input.X))
            and _is_panda_dataframe(x1)
        ):
            return XDataFrame[pd.DataFrame, pd.Series](X=x1)

        case input, x1 if (
            (_is_x_dataframe(input) or _is_x_series(input))
            and (
                _is_panda_dataframe(input.X)
                or _is_panda_series(input.X)
                or _is_spmatrix(input.X)
            )
            and _is_ndarray(x1)
        ):
            return XDataFrame[pd.DataFrame, pd.Series](X=pd.DataFrame(x1))

        case input, x1 if (
            (_is_x_dataframe(input) or _is_x_series(input))
            and (
                _is_panda_dataframe(input.X)
                or _is_panda_series(input.X)
                or _is_spmatrix(input.X)
            )
            and _is_spmatrix(x1)
        ):
            return XDataFrame[sp.spmatrix, sp.spmatrix](X=x1)

        case _:
            raise TypeError(
                "Unsupported input/result combination: "
                f"input={type(input).__name__}, result={type(x1).__name__}"
            )


def is_xy_data(
    input: AnyDataType,
) -> TypeGuard[XyDataFrame[Any, Any, Any] | XySeries[Any, Any]]:
    return _is_xy_dataframe(input) or _is_xy_series(input)


def wrap_result_dataframe_xy(
    input: AnyDataType,
    x1: Any,
) -> XyDataFrame[Any, Any, Any]:
    match input, x1:
        case input, x1 if (
            (_is_xy_dataframe(input) or _is_xy_series(input))
            and (_is_panda_dataframe(input.X) or _is_panda_series(input.X))
            and _is_panda_series(input.y)
            and _is_panda_dataframe(x1)
        ):
            return XyDataFrame[pd.DataFrame, pd.Series, pd.Series](X=x1, y=input.y)

        case input, x1 if (
            (_is_xy_dataframe(input) or _is_xy_series(input))
            and (
                _is_panda_dataframe(input.X)
                or _is_panda_series(input.X)
                or _is_spmatrix(input.X)
            )
            and _is_panda_series(input.y)
            and _is_ndarray(x1)
        ):
            return XyDataFrame[pd.DataFrame, pd.Series, pd.Series](
                X=pd.DataFrame(x1), y=input.y
            )

        case input, x1 if (
            (_is_xy_dataframe(input) or _is_xy_series(input))
            and (
                _is_panda_dataframe(input.X)
                or _is_panda_series(input.X)
                or _is_spmatrix(input.X)
            )
            and _is_panda_series(input.y)
            and _is_spmatrix(x1)
        ):
            return XyDataFrame[sp.spmatrix, sp.spmatrix, pd.Series](X=x1, y=input.y)

        case _:
            raise TypeError(
                "Unsupported input/result combination: "
                f"input={type(input).__name__}, result={type(x1).__name__}"
            )


@overload
def select_column_x(
    input: XyDataFrame[Any, Any, Any], name: str
) -> XySeries[Any, Any]: ...


@overload
def select_column_x(input: XDataFrame[Any, Any], name: str) -> XSeries[Any]: ...


def select_column_x(input: AnyDataFrame, name: str) -> AnySeries:
    match input:
        case input if (
            _is_xy_dataframe(input)
            and _is_panda_dataframe(input.X)
            and _is_panda_series(input.y)
            and _is_panda_series(x1 := cast(Any, input.X[name]))
        ):
            return XySeries[pd.Series, pd.Series](X=x1, y=input.y)

        case XDataFrame() as input if (
            _is_panda_dataframe(input.X)
            and _is_panda_series(x1 := cast(Any, input.X[name]))
        ):
            return XSeries[pd.Series](X=x1)

        case _:
            raise TypeError(f"Unsupported input type: {type(input).__name__}")


def select_column_xy(
    input: XyDataFrame[Any, Any, Any],
    name: str,
) -> XySeries[Any, Any]:
    match input:
        case input if (
            _is_xy_dataframe(input)
            and _is_panda_dataframe(input.X)
            and _is_panda_series(input.y)
            and _is_panda_series(x1 := cast(Any, input.X[name]))
        ):
            return XySeries[pd.Series, pd.Series](X=x1, y=input.y)

        case _:
            raise TypeError(f"Unsupported input type: {type(input).__name__}")


@overload
def wrap_result_series_x(input: XySeries[Any, Any], x1: Any) -> XySeries[Any, Any]: ...


@overload
def wrap_result_series_x(input: XSeries[Any], x1: Any) -> XSeries[Any]: ...


def wrap_result_series_x(input: AnySeries, x1: Any) -> AnySeries:
    if result := try_wrap_result_series_xy(input, x1):
        return result

    match input, x1:
        case XSeries() as input, x1 if (
            _is_panda_series(input.X) and _is_panda_series(x1)
        ):
            return XSeries[pd.Series](X=x1)

        case XSeries() as input, x1 if (_is_panda_series(input.X) and _is_ndarray(x1)):
            return XSeries[pd.Series](X=pd.Series(x1))

        case _:
            raise TypeError(
                "Unsupported input/result combination: "
                f"input={type(input).__name__}, result={type(x1).__name__}"
            )


def wrap_result_series_xy(
    input: AnySeries,
    x1: Any,
) -> XySeries[Any, Any]:
    if result := try_wrap_result_series_xy(input, x1):
        return result

    raise TypeError(
        "Unsupported input/result combination: "
        f"input={type(input).__name__}, result={type(x1).__name__}"
    )


def try_wrap_result_series_xy(input: AnySeries, x1: Any) -> XySeries[Any, Any] | None:
    match input, x1:
        case input, x1 if (
            _is_xy_series(input) and _is_panda_series(input.X) and _is_panda_series(x1)
        ):
            return XySeries[pd.Series, pd.Series](X=x1, y=input.y)

        case input, x1 if (
            _is_xy_series(input) and _is_panda_series(input.X) and _is_ndarray(x1)
        ):
            return XySeries[pd.Series, pd.Series](X=pd.Series(x1), y=input.y)

        case _:
            return None


def replace_series_column(
    input_df: AnyDataFrame, column: AnySeries, column_name: str
) -> AnyDataFrame:
    input_df.X[column_name] = column.X
    return input_df


def replace_dataframe_column(
    input_df: AnyDataFrame,
    columns: AnyDataFrame,
    column_name: str,
) -> AnyDataFrame:
    input_df.X = input_df.X.drop(columns=[column_name])
    input_df.X = pd.concat([input_df.X, columns.X], axis=1)  # pyright: ignore[reportUnknownMemberType]
    return input_df


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

        # The signature for SK-family can contain generic type variables for Table, Column and Target,
        # but would have infered == True.
        if not (
            type_contain_any_typevar(signature.i)
            or type_contain_any_typevar(signature.o)
        ):
            return signature

        provider_signature = infer_from_callable(self.provider)
        if provider_signature:
            return provider_signature

        return IcoSignature(i=type(None), c=None, o=type[Any])

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


class XySource(
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

        # The signature for SK-family can contain generic type variables for Table, Column and Target,
        # but would have infered == True.
        if not (
            type_contain_any_typevar(signature.i)
            or type_contain_any_typevar(signature.o)
        ):
            return signature

        provider_signature = infer_from_callable(self.provider)
        if provider_signature:
            return provider_signature

        return IcoSignature(i=type(None), c=None, o=type[Any])

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
