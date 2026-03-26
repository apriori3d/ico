from examples.ml.skrub.data import XyDataFrame, XySeries
from examples.ml.skrub.ops import SKStringEncoder

if __name__ == "__main__":
    import skrub  # type: ignore[import-untyped]

    endcoder = SKStringEncoder[XySeries, XyDataFrame](
        n_components=2, vectorizer="tfidf"
    )
    endcoder.describe()

    orders = skrub.datasets.toy_orders()  # type: ignore[attr-defined,no-any-return]
    orders_xy = XyDataFrame(X=orders.X, y=orders.y)  # type: ignore[arg-type]

    products = XySeries(X=orders.X["product"], y=orders.y)  # type: ignore[arg-type]
    print(products)

    endcoder.fit_mode()
    result_fit = endcoder(products)
    print(result_fit)

    endcoder.predict_mode()
    result_predict = endcoder(products)
    print(result_predict)
