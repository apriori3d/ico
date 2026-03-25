from __future__ import annotations

import abc
import warnings
from typing import Generic, Literal, cast

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
from ico.core.operator import I, IcoOperator, O


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

    def _fit_transform(self, input: I) -> SKResultTypes:
        return self._call_transformer_method("fit_transform", input)

    def _transform(self, input: I) -> SKResultTypes:
        return self._call_transformer_method("transform", input)

    def _call_transformer_method(
        self,
        method_name: Literal["fit_transform", "transform"],
        input: I,
    ) -> SKResultTypes:
        match cast(SKData, input):
            case XyDataFrame(X=x, y=y) | XySeries(X=x, y=y):
                method = getattr(self.transformer, method_name)
                return cast(SKResultTypes, method(x, y=y))  # type: ignore[misc]

            case XDataFrame(X=x) | XSeries(X=x):
                method = getattr(self.transformer, method_name)
                return cast(SKResultTypes, method(x))  # type: ignore[misc]

            case _:
                raise TypeError(
                    f"Unsupported transformer input type: {type(input).__name__}"
                )


@setup_renderer_show_estimator()
class XyDataFrameTransformer(SKTransformer[XyDataFrame, XyDataFrame]):
    pass


@setup_renderer_show_estimator()
class XDataFrameTransformer(SKTransformer[XDataFrame, XDataFrame]):
    pass


@setup_renderer_show_estimator()
class XySeriesTransformer(SKTransformer[XySeries, XySeries]):
    pass


@setup_renderer_show_estimator()
class XSeriesTransformer(SKTransformer[XSeries, XSeries]):
    pass


@setup_renderer_show_estimator()
class XySeriesToDataFrameTransformer(SKTransformer[XySeries, XyDataFrame]):
    pass


@setup_renderer_show_estimator()
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
            x1 = input.X[:, : self.n_components].copy()  # To avoid a reference to X_out

        match x1:
            case sp.spmatrix() as sparse_matrix:
                return XDataFrame(X=pd.DataFrame.sparse.from_spmatrix(sparse_matrix))
            case _:
                return XDataFrame(X=pd.DataFrame(x1))


class BlockNormalize(SKBaseTransformer[XDataFrame, XDataFrame]):
    scaling_factor_: float | None = None

    def _fit_transform(self, input: XDataFrame) -> SKResultTypes:
        xarray = input.X.to_numpy()
        scaling_factor_ = scaling_factor(xarray)
        x1 = xarray / scaling_factor_
        self.scaling_factor_ = scaling_factor_

        return pd.DataFrame(x1)

    def _transform(self, input: XDataFrame) -> SKResultTypes:
        if self.scaling_factor_ is None:
            raise ValueError(
                "BlockNormalize transformer has not been fitted yet. Call fit or fit_transform before transform."
            )
        xarray = input.X.to_numpy()
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
) -> IcoOperator[XSeries, XDataFrame]:
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

        return to_str | tf_idf | truncated_svd

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


if __name__ == "__main__":
    from ico.describe import PlanRendererDefaultOptions

    PlanRendererDefaultOptions.renderers_paths.insert(0, "examples.ml")

    import skrub  # type: ignore[import-untyped]

    orders = skrub.datasets.toy_orders()  # type: ignore[attr-defined,no-any-return]

    xydata = XyDataFrame(X=orders.X, y=orders.y)  # type: ignore[arg-type]

    se = create_string_encoder(n_components=2, vectorizer="tfidf")
    se.describe()

    xyseries = XySeries(X=orders.X["product"], y=orders.y)  # type: ignore[arg-type]
    print(xyseries)

    # se.fit_mode()
    result_fit = se(xyseries)
    print(result_fit)
