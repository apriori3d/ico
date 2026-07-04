import pandas as pd  # type: ignore[import-untyped]

import skrub  # type: ignore[import-untyped]
from examples.ml.skrub.data import (
    PandaXyDataFrame,
    PandaXySeries,
    XDataFrame,
    XyDataFrame,
    XySource,
)
from examples.ml.skrub.estimator import DataFrameEstimator
from examples.ml.skrub.ops import (
    ApplyToColumn,
    StringEncoder,
)
from examples.ml.skrub.transformer import (
    DataFrameTransformer,
    SeriesToDataFrameTransformer,
    SeriesTransformer,
)


def load_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "customer_id": [101, 102, 103, 104, 105],
            "product": ["pen", "cup", "cup", "spoon", "cup"],
            "amount": [250.0, 150.0, 300.0, 200.0, 100.0],
            "is_valid": [True, True, False, True, False],
            "date": [
                "2020-04-03",
                "2020-04-04",
                "2020-04-04",
                "2020-04-05",
                "2020-04-11",
            ],
        }
    )


def load_orders_xy() -> XyDataFrame[pd.DataFrame, pd.Series, pd.Series]:
    y = pd.Series([1, 0, 1, 0, 1], name="target")  # Dummy target variable
    return XyDataFrame(X=load_orders(), y=y)


def load_orders_x() -> XDataFrame[pd.DataFrame, pd.Series]:
    return XDataFrame(X=load_orders())


if __name__ == "__main__":
    from sklearn.decomposition import PCA  # type: ignore[import-untyped]
    from sklearn.dummy import DummyClassifier  # type: ignore[import-untyped]

    source = XySource(load_orders_xy)

    FrameType = PandaXyDataFrame
    SeriesType = PandaXySeries

    product_encoder = StringEncoder[SeriesType, FrameType](
        n_components=2, vectorizer="hashing"
    )
    s = product_encoder.signature

    date_encoder = SeriesTransformer[SeriesType](
        skrub.ToDatetime()
    ) | SeriesToDataFrameTransformer[SeriesType, FrameType](
        skrub.DatetimeEncoder(add_total_seconds=False)
    )
    date_pca_encoder = DataFrameTransformer[FrameType](PCA(n_components=2))

    estimator = DataFrameEstimator(DummyClassifier())

    pipeline = (
        source
        | ApplyToColumn(product_encoder, "product", output_prefix="product")
        | ApplyToColumn(date_encoder, "date")
        | estimator
    )

    # | ApplyToColumn(
    #     date_pca_encoder,
    #     column_name=cast(s.Selector, s.glob("date_*")),  # pyright: ignore[reportUnknownMemberType]
    #     output_column_prefix="date_pca",
    # )
    pipeline.describe()
    pipeline.fit_mode()

    # result = pipeline(None)
    # print(result.X)
