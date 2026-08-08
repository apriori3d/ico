from __future__ import annotations

import warnings
from typing import Any, Generic, Literal, cast

import pandas as pd  # type: ignore[import-untyped]
from sklearn.decomposition import TruncatedSVD  # type: ignore[import-untyped]

from examples.ml.skrub.base import (
    SKContextOperator,
    SKOperator,
    SKOperatorProtocol,
)
from examples.ml.skrub.data import (
    AnyDataFrame,
    TDataFrame,
    TSeries,
    replace_dataframe_column,
    wrap_result_dataframe_x,
)
from examples.ml.skrub.describe.plan.utils import (
    setup_renderer_show_args,
)
from examples.ml.skrub.transformer import (
    ColumnExtractor,
    DataFrameTransformer,
    SeriesToDataFrameTransformer,
    SeriesTransformer,
    SKBaseTransformer,
)
from skrub._scaling_factor import (  # type: ignore[import-untyped]
    scaling_factor,  # pyright: ignore[reportUnknownVariableType]
)
from skrub._to_str import (  # type: ignore[import-untyped]
    ToStr,
)


@setup_renderer_show_args("n_components")
class SafeTruncatedSVD(
    Generic[TDataFrame],
    DataFrameTransformer[TDataFrame],
):
    n_components: int
    random_state: int | None

    def __init__(
        self,
        n_components: int = 2,
        random_state: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            TruncatedSVD(n_components=n_components, random_state=random_state),
            name=name,
        )
        self.n_components = n_components
        self.random_state = random_state

    def _estimator_fn(self, input: TDataFrame) -> TDataFrame:
        if (min_shape := min(input.X.shape)) > self.n_components:
            match self.mode:
                case "fit":
                    return self._fit_transform(input)

                case "predict":
                    return self._transform(input)

        elif input.X.shape[1] == self.n_components:
            x1 = input.X

        else:
            warnings.warn(
                f"The matrix shape is {(input.X.shape)}, and its minimum is "
                f"{min_shape}, which is too small to fit a truncated SVD with "
                f"n_components={self.n_components}. "
                "The embeddings will be truncated by keeping the first "
                f"{self.n_components} dimensions instead. ",
                stacklevel=1,
            )
            # To avoid a reference to X_out
            x1 = input.X[:, : self.n_components].copy()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

        return cast(TDataFrame, wrap_result_dataframe_x(cast(AnyDataFrame, input), x1))


class BlockNormalize(
    Generic[TDataFrame],
    SKBaseTransformer[TDataFrame, TDataFrame],
):
    scaling_factor_: float | None = None

    def _fit_transform(self, input: TDataFrame) -> TDataFrame:
        self.scaling_factor_ = cast(float, scaling_factor(input.X))
        return self._transform(input)

    def _transform(self, input: TDataFrame) -> TDataFrame:
        # TODO: figure out design for cases where fit is required
        if self.scaling_factor_ is None:
            raise ValueError(
                "BlockNormalize transformer has not been fitted yet. Call fit or fit_transform before transform."
            )
        x1 = cast(pd.DataFrame, input.X / self.scaling_factor_)

        return cast(TDataFrame, wrap_result_dataframe_x(cast(AnyDataFrame, input), x1))


class StringEncoder(SKOperator[TSeries, TDataFrame]):
    flow: SKOperatorProtocol[TSeries, TDataFrame]

    def __init__(
        self,
        n_components: int = 30,
        vectorizer: Literal["tfidf", "hashing"] = "tfidf",
        ngram_range: tuple[int, int] = (3, 4),
        analyzer: Literal["word", "char", "char_wb"] = "char_wb",
        stop_words: list[str] | None = None,
        random_state: int | None = None,
        vocabulary: dict[str, int] | None = None,
        name: str | None = None,
    ) -> None:
        from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
            HashingVectorizer,
            TfidfTransformer,
            TfidfVectorizer,
        )

        to_str = SeriesTransformer[TSeries](ToStr())

        truncated_svd = SafeTruncatedSVD[TDataFrame](
            n_components=n_components, random_state=random_state
        )

        block_normalize = BlockNormalize[TDataFrame]()

        tf_idf_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            analyzer=analyzer,
            stop_words=stop_words,
            vocabulary=vocabulary,
        )

        # Case 1: Using TfidfVectorizer directly as vectorizer
        if vectorizer == "tfidf":
            column_tf_idf = SeriesToDataFrameTransformer[TSeries, TDataFrame](
                tf_idf_vectorizer
            )
            flow = to_str | column_tf_idf | truncated_svd | block_normalize

        # Case 2: Adding HashingVectorizer before TfidfTransformer
        if vocabulary is not None:
            raise ValueError(
                "Custom vocabulary passed to StringEncoder, unsupported by"
                "HashingVectorizer. Rerun without a 'vocabulary' parameter."
            )

        hashing = SeriesToDataFrameTransformer[TSeries, TDataFrame](
            HashingVectorizer(
                ngram_range=ngram_range,
                analyzer=analyzer,
                stop_words=stop_words,
            )
        )

        # HashingVectorizer returns sparse counts; apply IDF weighting with TfidfTransformer.
        hash_tf_idf = DataFrameTransformer[TDataFrame](TfidfTransformer())

        flow = to_str | hashing | hash_tf_idf | truncated_svd | block_normalize

        super().__init__(name=name, children=[flow])
        self.flow = flow

    def _estimator_fn(self, input: TSeries) -> TDataFrame:
        return self.flow(input)


@setup_renderer_show_args("prefix")
class AddPrefixToColumns(
    Generic[TDataFrame],
    SKBaseTransformer[TDataFrame, TDataFrame],
):
    prefix: str

    def __init__(self, prefix: str, name: str | None = None):
        super().__init__(name=name)
        self.prefix = prefix

    def _fit_transform(self, input: TDataFrame) -> TDataFrame:
        return self._transform(input)

    def _transform(self, input: TDataFrame) -> TDataFrame:
        input.X = input.X.rename(columns=lambda c: f"{self.prefix}_{c}")  # pyright: ignore[reportUnknownMemberType]
        return input


class ApplyToColumn(
    Generic[TDataFrame, TSeries],
    SKBaseTransformer[TDataFrame, TDataFrame],
):
    full_column_flow: SKOperatorProtocol[TDataFrame, TDataFrame]
    replace_column_op: Any
    column_name: Any

    def __init__(
        self,
        column_flow: SKOperatorProtocol[TSeries, TDataFrame],
        column_name: str,
        output_prefix: str | None = None,
        name: str | None = None,
    ):
        extract_column_op = ColumnExtractor[TDataFrame, TSeries](column_name)
        full_column_flow = extract_column_op | column_flow

        if output_prefix is not None:
            full_column_flow = full_column_flow | AddPrefixToColumns[TDataFrame](
                prefix=output_prefix
            )

        replace_column_op = SKContextOperator[AnyDataFrame, TDataFrame, TDataFrame](
            lambda i, c: cast(TDataFrame, replace_dataframe_column(c, i, column_name))
        )

        super().__init__(name=name, children=[full_column_flow])

        self.full_column_flow = full_column_flow
        self.replace_column_op = replace_column_op
        self.column_name = column_name

    def _fit_transform(self, input: TDataFrame) -> TDataFrame:
        return self._transform(input)

    def _transform(self, input: TDataFrame) -> TDataFrame:
        single_column_data = self.full_column_flow(input)
        return self.replace_column_op(single_column_data, input)


# class ApplyToColumns(
#     Generic[TDataFrame],
#     SKBaseTransformer[TDataFrame, TDataFrame],
# ):
#     column_flow: SKOperatorProtocol[TDataFrame, TDataFrame]
#     replace_column_op: SKContextOperator[TDataFrame, TDataFrame, TDataFrame]
#     column_name: Any

#     def __init__(
#         self,
#         column_flow: SKOperatorProtocol[TDataFrame, TDataFrame],
#         column_pattern: s.Selector,
#         output_column_prefix: str,
#         name: str | None = None,
#     ):
#         full_column_flow = (
#             ColumnsExtractor[TDataFrame](column_pattern)
#             | column_flow
#             | AddPrefixToColumns[TDataFrame](prefix=output_column_prefix)
#         )
#         replace_column_op = SKContextOperator[TDataFrame, TDataFrame, TDataFrame](
#             lambda i, c: cast(
#                 TDataFrame,
#                 replace_dataframe_columns(i, c, column_pattern),
#             )
#         )

#         super().__init__(name=name, children=[full_column_flow])

#         self.column_flow = full_column_flow
#         self.replace_column_op = replace_column_op
#         self.column_name = column_pattern

#     def _fit_transform(self, input: TDataFrame) -> TDataFrame:
#         return self._transform(input)

#     def _transform(self, input: TDataFrame) -> TDataFrame:
#         # Process column flow with data frame output (many columns as result)
#         multi_column_data = self.column_flow(input)
#         return self.replace_column_op(input, multi_column_data)
