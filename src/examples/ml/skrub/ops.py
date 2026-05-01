from __future__ import annotations

import warnings
from typing import Any, Literal, cast, overload

from sklearn.decomposition import TruncatedSVD  # type: ignore[import-untyped]

from examples.ml.skrub.base import SKOperatorProtocol
from examples.ml.skrub.data import (
    AnyDataFrame,
    AnySeries,
    AnyXDataFrame,
    AnyXyDataFrame,
    XDataFrame,
    XyDataFrame,
    wrap_result_dataframe_x,
)
from examples.ml.skrub.describe.plan.utils import (
    setup_renderer_show_args,
)
from examples.ml.skrub.transformer import (
    DataFrameTransformer,
    SeriesToDataFrameSparseTransformer,
    SeriesToDataFrameTransformer,
    SeriesTransformer,
    SKBaseTransformer,
)
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
