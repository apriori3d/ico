import pandas as pd  # type: ignore[import-untyped]

from examples.ml.skrub.data import (
    XDataFrame,
    XyDataFrame,
    XySource,
)
from examples.ml.skrub.ops import AddPrefixToColumns, create_string_encoder
from examples.ml.skrub.transformer import ColumnExtractor


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
    product_encoder = create_string_encoder(n_components=2, vectorizer="hashing")
    product_pipeline = (
        ColumnExtractor("product") | product_encoder | AddPrefixToColumns("product")
    )
    source = XySource(load_orders_xy)
    pipeline = source | product_pipeline
    pipeline.describe()

    result = pipeline(None)
    print(result)
