from __future__ import annotations

import warnings
from typing import Any, Generic, Literal, cast

import pandas as pd  # type: ignore[import-untyped]
from sklearn.decomposition import TruncatedSVD  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)

from examples.ml.skrub.base import SKOperator
from examples.ml.skrub.data import (
    IDataFrame,
    ISeries,
    ODataFrame,
    XDataFrame,
    XSeries,
    is_dataframe_result,
    wrap_result,
)
from examples.ml.skrub.describe.plan.utils import (
    setup_renderer_show_args,
)
from examples.ml.skrub.transformer import (
    SKBaseTransformer,
    SKTransformer,
)
from ico.core.signature import IcoSignature  # type: ignore[import-untyped]
from skrub._scaling_factor import (  # type: ignore[import-untyped]
    scaling_factor,  # pyright: ignore[reportUnknownVariableType]
)
from skrub._to_str import (  # type: ignore[import-untyped]
    ToStr,
)


@setup_renderer_show_args("convert_category")
class SKColumnToStr(Generic[ISeries], SKTransformer[ISeries, ISeries]):
    def __init__(self, convert_category: bool = True, name: str | None = None):
        super().__init__(ToStr(convert_category=convert_category), name=name)

    @property
    def signature(self) -> IcoSignature:
        return IcoSignature(i=XSeries, c=None, o=XSeries)


@setup_renderer_show_args("n_components")
class SafeTruncatedSVD(Generic[IDataFrame], SKTransformer[IDataFrame, IDataFrame]):
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

    def _estimator_fn(self, input: IDataFrame) -> IDataFrame:
        if (min_shape := min(input.X.shape)) > self.n_components:
            match self.mode:
                case "fit":
                    x1 = self._fit_transform(input)

                case "predict":
                    x1 = self._transform(input)

            assert is_dataframe_result(x1)

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
            x1 = cast(
                pd.DataFrame,
                input.X[:, : self.n_components].copy(),  # pyright: ignore[reportUnknownMemberType]  # To avoid a reference to X_out
            )
        return cast(IDataFrame, wrap_result(input, x1))

    @property
    def signature(self) -> IcoSignature:
        return IcoSignature(i=XDataFrame, c=None, o=XDataFrame)


class BlockNormalize(Generic[IDataFrame], SKBaseTransformer[IDataFrame, IDataFrame]):
    scaling_factor_: float | None = None

    def _fit_transform(self, input: IDataFrame) -> pd.DataFrame:
        xarray = cast(Any, input.X.to_numpy())  # pyright: ignore[reportUnknownMemberType]
        scaling_factor_ = scaling_factor(xarray)
        x1 = xarray / scaling_factor_
        self.scaling_factor_ = scaling_factor_

        return pd.DataFrame(x1)

    def _transform(self, input: IDataFrame) -> pd.DataFrame:
        if self.scaling_factor_ is None:
            raise ValueError(
                "BlockNormalize transformer has not been fitted yet. Call fit or fit_transform before transform."
            )
        xarray = cast(Any, input.X.to_numpy())  # pyright: ignore[reportUnknownMemberType]
        x1 = xarray / self.scaling_factor_

        return pd.DataFrame(x1)

    @property
    def signature(self) -> IcoSignature:
        return IcoSignature(i=XDataFrame, c=None, o=XDataFrame)


class SKStringEncoder(Generic[ISeries, ODataFrame], SKOperator[ISeries, ODataFrame]):
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
        to_str = SKColumnToStr[ISeries]()

        truncated_svd = SafeTruncatedSVD[ODataFrame](
            n_components=n_components, random_state=random_state
        )

        block_normalize = BlockNormalize[ODataFrame]()

        tf_idf_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            analyzer=analyzer,
            stop_words=stop_words,
            vocabulary=vocabulary,
        )

        if vectorizer == "tfidf":
            # Case 1: Using TfidfVectorizer directly as vectorizer

            # Note: we wrap tf_idf_transformer with SKTransformer,
            # because input is XSeries from to_str
            tf_idf = SKTransformer[ISeries, ODataFrame](tf_idf_vectorizer)

            encoder_op = to_str | tf_idf | truncated_svd | block_normalize

        else:
            # Case 2: Adding HashingVectorizer before TfidfTransformer

            if vocabulary is not None:
                raise ValueError(
                    "Custom vocabulary passed to StringEncoder, unsupported by"
                    "HashingVectorizer. Rerun without a 'vocabulary' parameter."
                )

            hashing = SKTransformer[ISeries, ODataFrame](
                HashingVectorizer(
                    ngram_range=ngram_range,
                    analyzer=analyzer,
                    stop_words=stop_words,
                )
            )

            # HashingVectorizer returns sparse counts; apply IDF weighting with TfidfTransformer.
            tf_idf_transformer = SKTransformer[ODataFrame, ODataFrame](
                TfidfTransformer()
            )

            encoder_op = (
                to_str | hashing | tf_idf_transformer | truncated_svd | block_normalize
            )

        super().__init__(encoder_op, children=[encoder_op], name=name)

    @property
    def signature(self) -> IcoSignature:
        return IcoSignature(i=XSeries, c=None, o=XDataFrame)
