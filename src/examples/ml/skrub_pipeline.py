from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Generic, Literal

import pandas as pd
import sklearn
from rich.text import Text

from ico.core.node import IcoNode
from ico.core.operator import I, IcoOperator, O
from ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from ico.describe.plan.rich_renderer.row_renderer import (
    RowRenderer,
)
from ico.describe.rich_utils import (
    render_node_class,
)
from ico.describe.utils import match_icon

SKMode = Literal["fit", "predict"]


class SKModeMixin:
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

SKResultTypes = pd.Series | pd.DataFrame


class SKBaseOperator(Generic[I, O], IcoOperator[I, O], abc.ABC, SKModeMixin):
    estimator: sklearn.base.BaseEstimator

    def __init__(
        self,
        estimator: sklearn.base.BaseEstimator,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(self._estimator_fn, name=name)
        self.estimator = estimator

    @abc.abstractmethod
    def _estimator_fn(self, input: I) -> O: ...


class SKBaseTransformer(Generic[I, O], SKBaseOperator[I, O], abc.ABC):
    def __init__(
        self,
        estimator: sklearn.base.BaseEstimator,
        *,
        name: str | None = None,
    ) -> None:
        if not hasattr(estimator, "fit_transform"):
            raise ValueError(
                f"{estimator} does not have a fit_transform method for fit mode"
            )

        if not hasattr(estimator, "transform"):
            raise ValueError(
                f"{estimator} does not have a transform method for predict mode"
            )

        super().__init__(estimator, name=name)

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

    @abc.abstractmethod
    def _wrap_result(self, input: I, x1: SKResultTypes) -> O: ...


class XyDataFrameTransformer(SKBaseTransformer[XyDataFrame, XyDataFrame]):
    def _fit_transform(self, input: XyDataFrame) -> SKResultTypes:
        return self.estimator.fit_transform(input.X, y=input.y)  # type: ignore

    def _transform(self, input: XyDataFrame) -> SKResultTypes:
        return self.estimator.transform(input.X, y=input.y)  # type: ignore

    def _wrap_result(self, input: XyDataFrame, x1: SKResultTypes) -> XyDataFrame:
        assert isinstance(
            x1, pd.DataFrame
        ), f"Expected transformer to return a DataFrame, got {type(x1)}"  # pyright: ignore[reportUnknownArgumentType]
        return XyDataFrame(X=x1, y=input.y)


class XDataFrameTransformer(SKBaseTransformer[XDataFrame, XDataFrame]):
    def _fit_transform(self, input: XDataFrame) -> SKResultTypes:
        return self.estimator.fit_transform(input.X)  # type: ignore

    def _transform(self, input: XDataFrame) -> SKResultTypes:
        return self.estimator.transform(input.X)  # type: ignore

    def _wrap_result(self, input: XDataFrame, x1: SKResultTypes) -> XDataFrame:
        assert isinstance(
            x1, pd.DataFrame
        ), f"Expected transformer to return a DataFrame, got {type(x1)}"  # pyright: ignore[reportUnknownArgumentType]
        return XDataFrame(X=x1)


class XySeriesTransformer(SKBaseTransformer[XySeries, XySeries]):
    def _fit_transform(self, input: XySeries) -> SKResultTypes:
        return self.estimator.fit_transform(input.X, y=input.y)  # type: ignore

    def _transform(self, input: XySeries) -> SKResultTypes:
        return self.estimator.transform(input.X, y=input.y)  # type: ignore

    def _wrap_result(self, input: XySeries, x1: SKResultTypes) -> XySeries:
        assert isinstance(
            x1, pd.Series
        ), f"Expected transformer to return a Series, got {type(x1)}"  # pyright: ignore[reportUnknownArgumentType]
        return XySeries(X=x1, y=input.y)


class XSeriesTransformer(SKBaseTransformer[XSeries, XSeries]):
    def _fit_transform(self, input: XSeries) -> SKResultTypes:
        return self.estimator.fit_transform(input.X)  # type: ignore

    def _transform(self, input: XSeries) -> SKResultTypes:
        return self.estimator.transform(input.X)  # type: ignore

    def _wrap_result(self, input: XSeries, x1: SKResultTypes) -> XSeries:
        assert isinstance(
            x1, pd.Series
        ), f"Expected transformer to return a Series, got {type(x1)}"  # pyright: ignore[reportUnknownArgumentType]
        return XSeries(X=x1)


class XySeriesToDataFrameTransformer(SKBaseTransformer[XySeries, XyDataFrame]):
    def _fit_transform(self, input: XySeries) -> SKResultTypes:
        return self.estimator.fit_transform(input.X, y=input.y)  # type: ignore

    def _transform(self, input: XySeries) -> SKResultTypes:
        return self.estimator.transform(input.X, y=input.y)  # type: ignore

    def _wrap_result(self, input: XySeries, x1: SKResultTypes) -> XyDataFrame:
        assert isinstance(
            x1, pd.DataFrame
        ), f"Expected transformer to return a DataFrame, got {type(x1)}"  # pyright: ignore[reportUnknownArgumentType]
        return XyDataFrame(X=x1, y=input.y)


class XSeriesToDataFrameTransformer(SKBaseTransformer[XSeries, XDataFrame]):
    def _fit_transform(self, input: XSeries) -> SKResultTypes:
        return self.estimator.fit_transform(input.X)  # type: ignore

    def _transform(self, input: XSeries) -> SKResultTypes:
        return self.estimator.transform(input.X)  # type: ignore

    def _wrap_result(self, input: XSeries, x1: SKResultTypes) -> XDataFrame:
        assert isinstance(
            x1, pd.DataFrame
        ), f"Expected transformer to return a DataFrame, got {type(x1)}"  # pyright: ignore[reportUnknownArgumentType]
        return XDataFrame(X=x1)


from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from skrub._to_str import ToStr  # pyright: ignore[reportMissingTypeStubs]


def create_string_encoder(
    n_components: int = 30,
    vectorizer: Literal["tfidf", "hashing"] = "tfidf",
    ngram_range: tuple[int, int] = (3, 4),
    analyzer: Literal["word", "char", "char_wb"] = "char_wb",
    stop_words: list[str] | None = None,
    random_state: int | None = None,
    vocabulary: dict[str, int] | None = None,
) -> IcoOperator[XSeries, XDataFrame]:
    to_str = XSeriesTransformer(ToStr(convert_category=True))

    tf_idf_transformer = TfidfVectorizer(
        ngram_range=ngram_range,
        analyzer=analyzer,
        stop_words=stop_words,
        vocabulary=vocabulary,
    )

    if vectorizer == "tfidf":
        tf_idf = XSeriesToDataFrameTransformer(tf_idf_transformer)

        return to_str | tf_idf

    # vectorizer == "hashing"

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

    tf_idf = XDataFrameTransformer(tf_idf_transformer)

    return to_str | hashing | tf_idf


@register_renderer(SKBaseOperator)
class SKTransformerRender(RowRenderer):
    def render_flow_column(self, node: IcoNode) -> Text:
        """Render Flow column: icons, class name, and arguments."""
        text = self.flow_column_prefix or Text("")

        if not self.flow_includes_node_info:
            return text

        if self.options.show_node_icons:
            icon = match_icon(self.options.node_icons, node)
            if icon:
                text += Text(icon)

        assert isinstance(node, SKBaseTransformer)

        text += render_node_class(
            node.estimator,
            options=self.options,
        )

        if self.flow_column_postfix:
            text += self.flow_column_postfix

        return text


if __name__ == "__main__":
    from ico.describe import PlanRendererDefaultOptions

    PlanRendererDefaultOptions.renderers_paths.insert(0, "examples.ml")

    se = create_string_encoder(n_components=2, vectorizer="hashing")
    se.describe()

    print(se.signature)
