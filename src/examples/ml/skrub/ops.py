from __future__ import annotations

import warnings
from typing import Any, overload

from sklearn.decomposition import TruncatedSVD  # type: ignore[import-untyped]

from examples.ml.skrub.data import (
    XDataFrame,
    XyDataFrame,
    wrap_result_dataframe_x,
)
from examples.ml.skrub.describe.plan.utils import (
    setup_renderer_show_args,
)
from examples.ml.skrub.transformer import (
    XDataFrameTransformer,
    XSeriesTransformer,
)
from skrub._to_str import (  # type: ignore[import-untyped]
    ToStr,
)


@setup_renderer_show_args("convert_category")
class SKColumnToStr(XSeriesTransformer):
    def __init__(self, convert_category: bool = True, name: str | None = None):
        super().__init__(ToStr(convert_category=convert_category), name=name)


@setup_renderer_show_args("n_components")
class SafeTruncatedSVD(
    XDataFrameTransformer,
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


# class BlockNormalize(Generic[TTable], SKBaseTransformer[TTable, TTable]):
#     scaling_factor_: float | None = None

#     def _fit_transform(self, input: TTable) -> pd.DataFrame:
#         xarray = cast(Any, input.X.to_numpy())  # pyright: ignore[reportUnknownMemberType]
#         scaling_factor_ = scaling_factor(xarray)
#         x1 = xarray / scaling_factor_
#         self.scaling_factor_ = scaling_factor_

#         return pd.DataFrame(x1)

#     def _transform(self, input: TTable) -> pd.DataFrame:
#         if self.scaling_factor_ is None:
#             raise ValueError(
#                 "BlockNormalize transformer has not been fitted yet. Call fit or fit_transform before transform."
#             )
#         xarray = cast(Any, input.X.to_numpy())  # pyright: ignore[reportUnknownMemberType]
#         x1 = xarray / self.scaling_factor_

#         return pd.DataFrame(x1)

#     @property
#     def signature(self) -> IcoSignature:
#         return IcoSignature(i=XDataFrame, c=None, o=XDataFrame)


# class SKStringEncoder(Generic[TSeries, ODataFrame], SKOperator[TSeries, ODataFrame]):
#     def __init__(
#         self,
#         n_components: int = 30,
#         vectorizer: Literal["tfidf", "hashing"] = "tfidf",
#         ngram_range: tuple[int, int] = (3, 4),
#         analyzer: Literal["word", "char", "char_wb"] = "char_wb",
#         stop_words: list[str] | None = None,
#         random_state: int | None = None,
#         vocabulary: dict[str, int] | None = None,
#         name: str | None = None,
#     ) -> None:
#         to_str = SKColumnToStr[TSeries]()

#         truncated_svd = SafeTruncatedSVD[ODataFrame](
#             n_components=n_components, random_state=random_state
#         )

#         block_normalize = BlockNormalize[ODataFrame]()

#         tf_idf_vectorizer = TfidfVectorizer(
#             ngram_range=ngram_range,
#             analyzer=analyzer,
#             stop_words=stop_words,
#             vocabulary=vocabulary,
#         )

#         if vectorizer == "tfidf":
#             # Case 1: Using TfidfVectorizer directly as vectorizer

#             # Note: we wrap tf_idf_transformer with SKTransformer,
#             # because input is XSeries from to_str
#             tf_idf = SKTransformer[TSeries, ODataFrame](tf_idf_vectorizer)

#             encoder_op = to_str | tf_idf | truncated_svd | block_normalize

#         else:
#             # Case 2: Adding HashingVectorizer before TfidfTransformer

#             if vocabulary is not None:
#                 raise ValueError(
#                     "Custom vocabulary passed to StringEncoder, unsupported by"
#                     "HashingVectorizer. Rerun without a 'vocabulary' parameter."
#                 )

#             hashing = SKTransformer[TSeries, ODataFrame](
#                 HashingVectorizer(
#                     ngram_range=ngram_range,
#                     analyzer=analyzer,
#                     stop_words=stop_words,
#                 )
#             )

#             # HashingVectorizer returns sparse counts; apply IDF weighting with TfidfTransformer.
#             tf_idf_transformer = SKTransformer[ODataFrame, ODataFrame](
#                 TfidfTransformer()
#             )

#             encoder_op = (
#                 to_str | hashing | tf_idf_transformer | truncated_svd | block_normalize
#             )

#         super().__init__(encoder_op, children=[encoder_op], name=name)

#     @property
#     def signature(self) -> IcoSignature:
#         return IcoSignature(i=XSeries, c=None, o=XDataFrame)
