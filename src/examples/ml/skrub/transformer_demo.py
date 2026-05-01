from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]

from examples.ml.skrub.data import (
    XDataFrame,
    XyDataFrame,
    XySource,
)
from examples.ml.skrub.transformer import (
    ColumnExtractor,
    DataFrameTransformer,
    SeriesToDataFrameTransformer,
    SeriesTransformer,
)


def load_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "customer_id": [101, 102, 103, 104, 105],
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
    import numpy as np
    from sklearn.decomposition import PCA  # type: ignore[import-untyped]

    import skrub  # type: ignore[import-untyped]
    from examples.ml.skrub.data import XDataFrame
    from skrub._to_str import (  # type: ignore[import-untyped]
        ToStr,
    )

    source = XySource(load_orders_xy)

    prepare_is_valid = ColumnExtractor("is_valid") | SeriesTransformer(ToStr())

    prepare_date = ColumnExtractor("date") | SeriesTransformer(skrub.ToDatetime())
    encode_date = SeriesToDataFrameTransformer(
        skrub.DatetimeEncoder(add_total_seconds=False)
    )
    date_pca = DataFrameTransformer(PCA(n_components=2))

    pipeline = source | prepare_date | encode_date | date_pca
    pipeline.describe()

    pipeline.fit_mode()
    result1 = pipeline(None)

    pipeline.predict_mode()
    result2 = pipeline(None)

    print(np.allclose(result1.X, result2.X))
