from __future__ import annotations

import warnings
from typing import Any, Literal, cast, overload

from sklearn.decomposition import TruncatedSVD  # type: ignore[import-untyped]

from examples.ml.skrub.base import SKContextOperator, SKOperatorProtocol
from examples.ml.skrub.data import (
    AnyDataFrame,
    AnySeries,
    AnyXDataFrame,
    AnyXyDataFrame,
    XDataFrame,
    XyDataFrame,
    is_dataframe_output_type,
    is_series_output_type,
    replace_dataframe_column,
    replace_series_column,
    wrap_result_dataframe_x,
)
from examples.ml.skrub.describe.plan.utils import (
    setup_renderer_show_args,
)
from examples.ml.skrub.transformer import (
    ColumnExtractor,
    DataFrameTransformer,
    SeriesToDataFrameSparseTransformer,
    SeriesToDataFrameTransformer,
    SeriesTransformer,
    SKBaseTransformer,
)
from ico.core.node import IcoNodeProtocol
from ico.core.signature import IcoSignature
from skrub._scaling_factor import (  # type: ignore[import-untyped]
    scaling_factor,  # pyright: ignore[reportUnknownVariableType]
)
from skrub._to_str import (  # type: ignore[import-untyped]
    ToStr,
)


@setup_renderer_show_args("n_components")
class SafeTruncatedSVD(DataFrameTransformer):
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

    @overload
    def _estimator_fn(
        self, input: XyDataFrame[Any, Any, Any]
    ) -> XyDataFrame[Any, Any, Any]: ...

    @overload
    def _estimator_fn(self, input: XDataFrame[Any, Any]) -> XDataFrame[Any, Any]: ...

    def _estimator_fn(self, input: XDataFrame[Any, Any]) -> XDataFrame[Any, Any]:
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
            x1 = input.X[:, : self.n_components].copy()  # To avoid a reference to X_out

        return wrap_result_dataframe_x(input, x1)


class BlockNormalize(SKBaseTransformer[AnyDataFrame, AnyDataFrame]):
    scaling_factor_: float | None = None

    @overload
    def _fit_transform(self, input: AnyXyDataFrame) -> AnyXyDataFrame: ...

    @overload
    def _fit_transform(self, input: AnyXDataFrame) -> AnyXDataFrame: ...

    def _fit_transform(self, input: AnyXDataFrame) -> AnyXDataFrame:
        self.scaling_factor_ = cast(float, scaling_factor(input.X))
        return self._transform(input)

    @overload
    def _transform(self, input: AnyXyDataFrame) -> AnyXyDataFrame: ...

    @overload
    def _transform(self, input: AnyXDataFrame) -> AnyXDataFrame: ...

    def _transform(self, input: AnyXDataFrame) -> AnyXDataFrame:
        # TODO: figure out design for cases where fit is required
        if self.scaling_factor_ is None:
            raise ValueError(
                "BlockNormalize transformer has not been fitted yet. Call fit or fit_transform before transform."
            )
        x1 = input.X / self.scaling_factor_

        return wrap_result_dataframe_x(input, x1)

    @property
    def signature(self) -> IcoSignature:
        return IcoSignature(i=AnyXDataFrame, c=None, o=AnyXDataFrame)


class AddPrefixToColumns(SKBaseTransformer[AnyDataFrame, AnyDataFrame]):
    prefix: str

    def __init__(self, prefix: str, name: str | None = None):
        super().__init__(name=name)
        self.prefix = prefix

    @overload
    def _fit_transform(self, input: AnyXyDataFrame) -> AnyXyDataFrame: ...

    @overload
    def _fit_transform(self, input: AnyXDataFrame) -> AnyXDataFrame: ...

    def _fit_transform(self, input: AnyXDataFrame) -> AnyXDataFrame:
        return self._transform(input)

    @overload
    def _transform(self, input: AnyXyDataFrame) -> AnyXyDataFrame: ...

    @overload
    def _transform(self, input: AnyXDataFrame) -> AnyXDataFrame: ...

    def _transform(self, input: AnyXDataFrame) -> AnyXDataFrame:
        input.X = input.X.rename(columns=lambda c: f"{self.prefix}_{c}")  # pyright: ignore[reportUnknownLambdaType]
        return input

    @property
    def signature(self) -> IcoSignature:
        return IcoSignature(i=AnyXDataFrame, c=None, o=AnyXDataFrame)


# @overload
# def apply_to_column(
#     column_flow: SKOperator[AnySeries, AnyDataFrame], column_name: str
# ) -> SKOperator[AnyDataFrame, AnyDataFrame]: ...


# @overload
# def apply_to_column(
#     column_flow: SKOperator[AnySeries, AnySeries], column_name: str
# ) -> SKOperator[AnyDataFrame, AnyDataFrame]: ...


# def apply_to_column(
#     column_flow: SKOperator[AnySeries, AnySeries | AnyDataFrame],
#     column_name: str,
# ) -> SKOperator[AnyDataFrame, AnyDataFrame]:
#     pass


class ApplyToColumn(SKBaseTransformer[AnyDataFrame, AnyDataFrame]):
    dataframe_column_flow: SKOperatorProtocol[AnyDataFrame, AnyDataFrame] | None
    replace_dataframe_column_op: (
        SKContextOperator[AnyDataFrame, AnyDataFrame, AnyDataFrame] | None
    )
    series_column_flow: SKOperatorProtocol[AnyDataFrame, AnySeries] | None
    replace_series_column_op: (
        SKContextOperator[AnySeries, AnyDataFrame, AnyDataFrame] | None
    )
    column_name: str

    @overload
    def __init__(
        self,
        column_flow: SKOperatorProtocol[AnySeries, AnyDataFrame],
        column_name: str,
        name: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        column_flow: SKOperatorProtocol[AnySeries, AnySeries],
        column_name: str,
        name: str | None = None,
    ) -> None: ...

    def __init__(
        self,
        column_flow: SKOperatorProtocol[AnySeries, Any],
        column_name: str,
        name: str | None = None,
    ):
        # The column_flow can output either a Series or a DataFrame.
        # If it outputs a DataFrame, we will add a prefix to the column names before merging to input data frame.
        # If it outputs a Series, we will keep the same column name.
        dataframe_column_flow: SKOperatorProtocol[AnyDataFrame, AnyDataFrame] | None
        series_column_flow: SKOperatorProtocol[AnyDataFrame, AnySeries] | None

        replace_dataframe_column_op: (
            SKContextOperator[AnyDataFrame, AnyDataFrame, AnyDataFrame] | None
        )
        replace_series_column_op: (
            SKContextOperator[AnySeries, AnyDataFrame, AnyDataFrame] | None
        )

        children: list[IcoNodeProtocol] = []

        if is_dataframe_output_type(column_flow.signature.o):
            dataframe_column_flow = (
                ColumnExtractor(column_name)
                | cast(SKOperatorProtocol[AnySeries, AnyDataFrame], column_flow)
                | AddPrefixToColumns(prefix=column_name)
            )
            series_column_flow = None

            replace_dataframe_column_op = SKContextOperator[
                AnyDataFrame, AnyDataFrame, AnyDataFrame
            ](lambda i, c: replace_dataframe_column(i, c, column_name))
            replace_series_column_op = None

            children += [dataframe_column_flow, replace_dataframe_column_op]

        elif is_series_output_type(column_flow.signature.o):
            dataframe_column_flow = None
            series_column_flow = ColumnExtractor(column_name) | cast(
                SKOperatorProtocol[AnySeries, AnySeries], column_flow
            )
            replace_series_column_op = SKContextOperator[
                AnySeries, AnyDataFrame, AnyDataFrame
            ](lambda i, c: replace_series_column(c, i, column_name))
            replace_dataframe_column_op = None

            children += [series_column_flow, replace_series_column_op]
        else:
            raise ValueError(
                f"Unsupported column_flow output type: {column_flow.signature.o}. "
                "Expected AnyDataFrame or AnySeries."
            )
        super().__init__(name=name, children=children)

        self.dataframe_column_flow = dataframe_column_flow
        self.replace_dataframe_column_op = replace_dataframe_column_op

        self.series_column_flow = series_column_flow
        self.replace_series_column_op = replace_series_column_op

        self.column_name = column_name

    @overload
    def _fit_transform(self, input: AnyXyDataFrame) -> AnyXyDataFrame: ...

    @overload
    def _fit_transform(self, input: AnyXDataFrame) -> AnyXDataFrame: ...

    def _fit_transform(self, input: AnyXDataFrame) -> AnyXDataFrame:
        return self._transform(input)

    @overload
    def _transform(self, input: AnyXyDataFrame) -> AnyXyDataFrame: ...

    @overload
    def _transform(self, input: AnyXDataFrame) -> AnyXDataFrame: ...

    def _transform(self, input: AnyXDataFrame) -> AnyXDataFrame:
        # Process column flow with data frame output (many columns as result)
        if (
            self.dataframe_column_flow is not None
            and self.replace_dataframe_column_op is not None
        ):
            multi_column_data = self.dataframe_column_flow(input)
            return self.replace_dataframe_column_op(input, multi_column_data)

        # Process column flow with series output (single column as result)
        elif (
            self.series_column_flow is not None
            and self.replace_series_column_op is not None
        ):
            single_column_data = self.series_column_flow(input)
            return self.replace_series_column_op(single_column_data, input)

        raise ValueError(
            "Invalid state in ApplyToColumn: both dataframe_column_flow and series_column_flow are None."
        )

    @property
    def signature(self) -> IcoSignature:
        return IcoSignature(i=AnyXDataFrame, c=None, o=AnyXDataFrame)


def create_string_encoder(
    n_components: int = 30,
    vectorizer: Literal["tfidf", "hashing"] = "tfidf",
    ngram_range: tuple[int, int] = (3, 4),
    analyzer: Literal["word", "char", "char_wb"] = "char_wb",
    stop_words: list[str] | None = None,
    random_state: int | None = None,
    vocabulary: dict[str, int] | None = None,
) -> SKOperatorProtocol[AnySeries, AnyDataFrame]:
    from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
        HashingVectorizer,
        TfidfTransformer,
        TfidfVectorizer,
    )

    to_str = SeriesTransformer(ToStr())

    truncated_svd = SafeTruncatedSVD(
        n_components=n_components, random_state=random_state
    )

    block_normalize = BlockNormalize()

    tf_idf_vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        analyzer=analyzer,
        stop_words=stop_words,
        vocabulary=vocabulary,
    )

    # Case 1: Using TfidfVectorizer directly as vectorizer
    if vectorizer == "tfidf":
        column_tf_idf = SeriesToDataFrameTransformer(tf_idf_vectorizer)
        return to_str | column_tf_idf | truncated_svd | block_normalize

    # Case 2: Adding HashingVectorizer before TfidfTransformer
    if vocabulary is not None:
        raise ValueError(
            "Custom vocabulary passed to StringEncoder, unsupported by"
            "HashingVectorizer. Rerun without a 'vocabulary' parameter."
        )

    hashing = SeriesToDataFrameSparseTransformer(
        HashingVectorizer(
            ngram_range=ngram_range,
            analyzer=analyzer,
            stop_words=stop_words,
        )
    )

    # HashingVectorizer returns sparse counts; apply IDF weighting with TfidfTransformer.
    hash_tf_idf = DataFrameTransformer(TfidfTransformer())

    return to_str | hashing | hash_tf_idf | truncated_svd | block_normalize
