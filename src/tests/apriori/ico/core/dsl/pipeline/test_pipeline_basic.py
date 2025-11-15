from apriori.ico.core import IcoOperator, IcoPipeline


def test_pipeline_execution_order() -> None:
    p = IcoPipeline[int, int, int](
        context=IcoOperator[int, int](lambda x: x + 1),
        body=[
            IcoOperator[int, int](lambda x: x * 2),
            IcoOperator[int, int](lambda x: x - 3),
        ],
        output=IcoOperator[int, int](lambda x: x * 10),
    )
    assert p(2) == ((2 + 1) * 2 - 3) * 10  # 30


def test_pipeline_callable_init() -> None:
    p = IcoPipeline[int, int, int](
        context=lambda x: x + 1,
        body=[
            lambda x: x * 2,
            lambda x: x - 3,
        ],
        output=lambda x: x * 10,
    )
    assert p(2) == ((2 + 1) * 2 - 3) * 10  # 30


def test_pipeline_len_and_iter() -> None:
    p = IcoPipeline[int, int, int](
        context=IcoOperator[int, int](lambda x: x + 1),
        body=[
            IcoOperator[int, int](lambda x: x + 1),
            IcoOperator[int, int](lambda x: x + 2),
        ],
        output=IcoOperator[int, int](lambda x: x),
    )
    assert len(p) == 2
    assert list(p) == list(p.body)
