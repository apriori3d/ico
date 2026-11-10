from ico import IcoOperator, plan_view, show_params
from ico.describe.plan.json_renderer import PlanCell


def text(cell: PlanCell) -> str:
    return "".join(segment["text"] for segment in cell)


@show_params("factor", "label")
class Scale(IcoOperator[float, float]):
    def __init__(self, factor: float, label: str = "x", offset: float = 0.0) -> None:
        self.factor = factor
        self.label = label
        self.offset = offset
        super().__init__(fn=self._apply)

    def _apply(self, x: float) -> float:
        return x * self.factor + self.offset


class Double(Scale):
    def __init__(self) -> None:
        super().__init__(factor=2.0, label="double")


@show_params()
class Plain(Scale):
    pass


class Hidden(IcoOperator[float, float]):
    def __init__(self, factor: float) -> None:
        self.factor = factor
        super().__init__(fn=self._apply)

    def _apply(self, x: float) -> float:
        return x * self.factor


def flow_column(op: IcoOperator[float, float]) -> str:
    return text(plan_view(op)["rows"][0][0])


def test_shown_fields_follow_the_class_name() -> None:
    assert flow_column(Scale(1.5)) == "Scale(factor=1.5, label='x')"


def test_shown_fields_are_meta_styled() -> None:
    cell = plan_view(Scale(1.5))["rows"][0][0]
    assert [s["text"] for s in cell if s["style"] == "meta"] == [
        "factor=1.5, label='x'"
    ]


def test_subclasses_inherit_or_override_the_fields() -> None:
    assert flow_column(Double()) == "Double(factor=2.0, label='double')"
    assert flow_column(Plain(1.5)) == "Plain()"


def test_operators_without_the_decorator_show_no_fields() -> None:
    assert flow_column(Hidden(1.5)) == "Hidden()"


def test_the_operator_still_runs() -> None:
    assert Scale(2.0, offset=1.0)(3.0) == 7.0
