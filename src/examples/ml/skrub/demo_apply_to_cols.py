from __future__ import annotations

from typing import TypeVar

from examples.ml.skrub.apply_to_cols import SKApplyToCols
from examples.ml.skrub.data import (
    SKSource,
    XDataFrame,
    XSeries,
    XyDataFrame,
    XySeries,
)
from examples.ml.skrub.ops import SKStringEncoder

I = TypeVar("I", bound=XDataFrame | XyDataFrame)  # noqa: E741
IColumn = TypeVar("IColumn", bound=XySeries | XSeries)


def load_orders() -> XyDataFrame:
    orders = skrub.datasets.toy_orders()  # type: ignore[attr-defined,no-any-return]
    return XyDataFrame(X=orders.X, y=orders.y)  # type: ignore[arg-type]


if __name__ == "__main__":
    import skrub  # type: ignore[import-untyped]

    # se.describe()
    orders = SKSource(load_orders, name="Toy Orders")

    str_encoder = SKApplyToCols[XyDataFrame](
        lambda: SKStringEncoder(
            n_components=2, vectorizer="tfidf", name="Encode Column"
        ),
        cols=["product", "date"],
    )
    flow = orders | str_encoder
    flow.describe()

    print(f"{flow.signature=}")

    # flow.fit_mode()
    # result_fit = flow(orders_xy)
    # print(result_fit)

    # flow.predict_mode()
    # result_predict = flow(orders_xy)
    # print(result_predict)
