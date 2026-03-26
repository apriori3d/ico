from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast

import pandas as pd  # type: ignore[import-untyped]

from examples.ml.skrub.base import SKOperator, SKOperatorProtocol
from examples.ml.skrub.data import (
    XDataFrame,
    XSeries,
    XyDataFrame,
    XySeries,
    select_column,
    wrap_result,
)
from ico.core.signature import IcoSignature

I = TypeVar("I", bound=XDataFrame | XyDataFrame)  # noqa: E741
IColumn = TypeVar("IColumn", bound=XySeries | XSeries)


class SKApplyToCols(Generic[I], SKOperator[I, I]):
    estimator_factory: Callable[[], SKOperatorProtocol[XSeries | XySeries, I]]
    cols: list[str]
    _col_estimators: dict[str, _ApplyToCol[XSeries | XySeries, I]]

    def __init__(
        self,
        estimator_factory: Callable[[], SKOperatorProtocol[XSeries | XySeries, I]],
        cols: list[str],
        name: str | None = None,
    ) -> None:
        col_estimators = {col: _ApplyToCol(estimator_factory()) for col in cols}

        super().__init__(
            self._apply_fn,
            children=list(col_estimators.values()),
            name=name,
        )

        self.estimator_factory = estimator_factory
        self.cols = cols
        self._col_estimators = col_estimators

    def _apply_fn(self, input: I) -> I:
        transformed_frames: list[pd.DataFrame] = []

        for col, estimator in self._col_estimators.items():
            col_series = select_column(input, col)
            result_frame = estimator(col_series)

            renamed_frame = cast(
                pd.DataFrame,
                cast(Any, result_frame.X).copy(),
            )
            renamed_frame.columns = [
                f"{col}_{index}" for index in range(1, renamed_frame.shape[1] + 1)
            ]
            transformed_frames.append(renamed_frame)

        result_df = pd.concat(transformed_frames, axis=1)  # pyright: ignore[reportUnknownMemberType]

        if isinstance(input, XyDataFrame):
            return cast(I, wrap_result(input, result_df))

        return cast(I, wrap_result(input, result_df))

    # @property
    # def signature(self) -> IcoSignature:
    #     signature = super().signature

    #     if not signature.infered:
    #         # If the signature is not infered, we can attempt to infer it from the column estimator.
    #         col_signature = next(iter(self._col_estimators.values())).signature
    #         if col_signature.infered:
    #             match col_signature.i:
    #                 case type[XSeries]():
    #                     input_type = XDataFrame
    #                 case type[XySeries] ():
    #                     input_type = XyDataFrame
    #                 case _:
    #                     input_type = type(Any)

    #             return IcoSignature(i=input_type, c=None, o=input_type)

    #     return signature


class _ApplyToCol(Generic[IColumn, I], SKOperator[IColumn, I]):
    column_estimator: SKOperatorProtocol[IColumn, I]

    def __init__(
        self,
        column_estimator: SKOperatorProtocol[IColumn, I],
    ) -> None:
        super().__init__(
            self._apply_fn,
            children=[column_estimator],
            name=column_estimator.name,
        )
        self.column_estimator = column_estimator

    def _apply_fn(self, input: IColumn) -> I:
        return self.column_estimator(input)

    @property
    def signature(self) -> IcoSignature:
        signature = super().signature

        return self.column_estimator.signature if not signature.infered else signature
