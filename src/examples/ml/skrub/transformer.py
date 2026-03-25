from __future__ import annotations

import abc
import warnings
from typing import Any, Generic, Literal, TypeVar, cast

import pandas as pd  # type: ignore[import-untyped]
import scipy.sparse as sp  # type: ignore[import-untyped]
import sklearn  # type: ignore[import-untyped]
from sklearn.decomposition import TruncatedSVD  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from skrub._scaling_factor import (  # type: ignore[import-untyped]
    scaling_factor,  # pyright: ignore[reportUnknownVariableType]
)
from skrub._to_str import ToStr  # type: ignore[import-untyped]

from examples.ml.skrub.base import (
    SKBaseEstimator,
    SKOperatorProtocol,
)
from examples.ml.skrub.data import (
    SKData,
    SKResultTypes,
    XDataFrame,
    XSeries,
    XyDataFrame,
    XySeries,
    wrap_result_data,
)
from examples.ml.skrub.renderer import (
    setup_renderer_show_args,
    setup_renderer_show_estimator,
)

# from ico.core.operator import I, O

I = TypeVar("I", bound=SKData)  # noqa: E741
O = TypeVar("O", bound=SKData)  # noqa: E741


def _sparse_to_dataframe(matrix: sp.spmatrix) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        pd.DataFrame.sparse.from_spmatrix(matrix),  # pyright: ignore[reportUnknownMemberType]
    )


class SKBaseTransformer(Generic[I, O], SKBaseEstimator[I, O], abc.ABC):
    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name=name)

    def _estimator_fn(self, input: I) -> O:
        match self.mode:
            case "fit":
                x1 = self._fit_transform(input)

            case "predict":
                x1 = self._transform(input)

        return self._wrap_result(input, x1)

    @abc.abstractmethod
    def _fit_transform(self, input: I) -> SKResultTypes: ...

    @abc.abstractmethod
    def _transform(self, input: I) -> SKResultTypes: ...

    def _wrap_result(self, input: I, x1: SKResultTypes) -> O:
        return cast(O, wrap_result_data(cast(SKData, input), x1))


@setup_renderer_show_estimator()
class SKTransformer(Generic[I, O], SKBaseTransformer[I, O]):
    transformer: sklearn.base.BaseEstimator

    def __init__(
        self,
        transformer: sklearn.base.BaseEstimator,
        *,
        name: str | None = None,
    ) -> None:
        if not hasattr(transformer, "fit_transform"):
            raise ValueError(
                f"{transformer} does not have a fit_transform method for fit mode"
            )

        if not hasattr(transformer, "transform"):
            raise ValueError(
                f"{transformer} does not have a transform method for predict mode"
            )

        super().__init__(name=name)

        self.transformer = transformer

    def _get_fit_args(self, input: I) -> dict[str, Any]:
        return {}

    def _get_transform_args(self, input: I) -> dict[str, Any]:
        return {}

    def _fit_transform(self, input: I) -> SKResultTypes:
        args = self._get_fit_args(input)
        return self.transformer.fit_transform(input.X, **args)  # type: ignore[misc]

    def _transform(self, input: I) -> SKResultTypes:
        args = self._get_transform_args(input)
        return self.transformer.transform(input.X, **args)  # type: ignore[misc]


class SKSupervisedTransformer(Generic[I, O], SKTransformer[I, O]):
    def _get_fit_args(self, input: I) -> dict[str, Any]:
        assert isinstance(input, (XyDataFrame | XySeries))
        return {"y": input.y}

    def _get_transform_args(self, input: I) -> dict[str, Any]:
        assert isinstance(input, (XyDataFrame | XySeries))
        return {"y": input.y}


class XyDataFrameTransformer(SKSupervisedTransformer[XyDataFrame, XyDataFrame]):
    pass


class XDataFrameTransformer(SKTransformer[XDataFrame, XDataFrame]):
    pass


class XySeriesTransformer(SKSupervisedTransformer[XySeries, XySeries]):
    pass


class XSeriesTransformer(SKTransformer[XSeries, XSeries]):
    pass


class XySeriesToDataFrameTransformer(SKSupervisedTransformer[XySeries, XyDataFrame]):
    pass


class XSeriesToDataFrameTransformer(SKTransformer[XSeries, XDataFrame]):
    pass


@setup_renderer_show_args("convert_category")
class ColumnToStr(XSeriesTransformer):
    def __init__(self, convert_category: bool = True, name: str | None = None):
        super().__init__(ToStr(convert_category=convert_category), name=name)


@setup_renderer_show_args("n_components")
class SafeTruncatedSVD(XDataFrameTransformer):
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

    def _estimator_fn(self, input: XDataFrame) -> XDataFrame:
        x1: SKResultTypes

        if (min_shape := min(input.X.shape)) > self.n_components:
            match self.mode:
                case "fit":
                    x1 = self._fit_transform(input)

                case "predict":
                    x1 = self._transform(input)

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
            # self.n_components can be greater than the number
            # of dimensions of result.
            # Therefore, self.n_components_ below stores the resulting
            # number of dimensions of result.
            x1 = cast(
                pd.DataFrame,
                input.X[:, : self.n_components].copy(),  # pyright: ignore[reportUnknownMemberType]  # To avoid a reference to X_out
            )

        match x1:
            case sp.spmatrix() as sparse_matrix:
                return XDataFrame(X=_sparse_to_dataframe(sparse_matrix))
            case _:
                return XDataFrame(X=pd.DataFrame(cast(Any, x1)))


class BlockNormalize(SKBaseTransformer[XDataFrame, XDataFrame]):
    scaling_factor_: float | None = None

    def _fit_transform(self, input: XDataFrame) -> SKResultTypes:
        xarray = cast(Any, input.X.to_numpy())  # pyright: ignore[reportUnknownMemberType]
        scaling_factor_ = scaling_factor(xarray)
        x1 = xarray / scaling_factor_
        self.scaling_factor_ = scaling_factor_

        return pd.DataFrame(x1)

    def _transform(self, input: XDataFrame) -> SKResultTypes:
        if self.scaling_factor_ is None:
            raise ValueError(
                "BlockNormalize transformer has not been fitted yet. Call fit or fit_transform before transform."
            )
        xarray = cast(Any, input.X.to_numpy())  # pyright: ignore[reportUnknownMemberType]
        x1 = xarray / self.scaling_factor_

        return pd.DataFrame(x1)


def create_string_encoder(
    n_components: int = 30,
    vectorizer: Literal["tfidf", "hashing"] = "tfidf",
    ngram_range: tuple[int, int] = (3, 4),
    analyzer: Literal["word", "char", "char_wb"] = "char_wb",
    stop_words: list[str] | None = None,
    random_state: int | None = None,
    vocabulary: dict[str, int] | None = None,
) -> SKOperatorProtocol[XSeries, XDataFrame]:
    to_str = ColumnToStr()

    tf_idf_vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        analyzer=analyzer,
        stop_words=stop_words,
        vocabulary=vocabulary,
    )

    truncated_svd = SafeTruncatedSVD(
        n_components=n_components, random_state=random_state
    )

    # Case 1: Using TfidfVectorizer directly as vectorizer

    if vectorizer == "tfidf":
        # Note: we wrap tf_idf_transformer with XSeriesToDataFrameTransformer,
        # because input is XSeries from to_str
        tf_idf = XSeriesToDataFrameTransformer(tf_idf_vectorizer)

        return to_str | tf_idf | truncated_svd | BlockNormalize()

    # Case 2: Adding HashingVectorizer before TfidfTransformer

    if vocabulary is not None:
        raise ValueError(
            "Custom vocabulary passed to StringEncoder, unsupported by"
            "HashingVectorizer. Rerun without a 'vocabulary' parameter."
        )

    hashing = XSeriesToDataFrameTransformer(
        HashingVectorizer(
            ngram_range=ngram_range,
            analyzer=analyzer,
            stop_words=stop_words,
        )
    )

    # HashingVectorizer returns sparse counts; apply IDF weighting with TfidfTransformer.
    tf_idf_transformer = XDataFrameTransformer(TfidfTransformer())

    return to_str | hashing | tf_idf_transformer | truncated_svd | BlockNormalize()
