from examples.ml.skrub.base import SKChain
from examples.ml.skrub.data import XyDataFrame, XySeries
from examples.ml.skrub.transformer import create_string_encoder

if __name__ == "__main__":
    from ico.describe import PlanRendererDefaultOptions

    PlanRendererDefaultOptions.renderers_paths.insert(0, "examples.ml")
    PlanRendererDefaultOptions.flatten_node_type.add(SKChain)

    import skrub  # type: ignore[import-untyped]

    se = create_string_encoder(n_components=2, vectorizer="tfidf")
    se.describe()

    orders = skrub.datasets.toy_orders()  # type: ignore[attr-defined,no-any-return]

    orders_xy = XyDataFrame(X=orders.X, y=orders.y)  # type: ignore[arg-type]

    products = XySeries(X=orders.X["product"], y=orders.y)  # type: ignore[arg-type]
    print(products)

    se.fit_mode()
    result_fit = se(products)
    print(result_fit)

    se.predict_mode()
    result_predict = se(products)
    print(result_predict)
