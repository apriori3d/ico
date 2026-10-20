"""spec_assets: asset ids from AssetId params and resource fields."""

from dataclasses import dataclass
from typing import Any

import pytest

from ico import IcoOperator
from ico.assets import AssetId
from ico.registry import IcoRegistry, register
from ico.resources import Resource, resource
from ico.spec import NodeSpec, SpecError, spec_assets


@resource("test.expert", 1)
@dataclass(frozen=True)
class Expert(Resource[object]):
    """A transformer with an optional LoRA."""

    weights: AssetId = "wan22-i2v-high"
    lora: AssetId = ""
    storage: str = "fp8"

    def load(self) -> object:
        return object()


HIGH = Expert(lora="wan22-lightx2v-high")
REGISTRY = IcoRegistry()


@register("test.animate", 1, registry=REGISTRY)
class Animate(IcoOperator[str, str]):
    """Animate with an expert and a lookup table."""

    def __init__(
        self, expert: Expert = HIGH, table: AssetId = "luts", label: str = "x"
    ) -> None:
        self.expert = expert
        self.table = table
        self.label = label
        super().__init__(fn=self._apply)

    def _apply(self, text: str) -> str:
        return text


def op(**params: Any) -> NodeSpec:
    return NodeSpec(kind="op", op="test.animate@1", params=params)


def test_assets_from_defaults_and_overrides() -> None:
    assert HIGH.assets() == ["wan22-i2v-high", "wan22-lightx2v-high"]
    assert spec_assets(op(), registry=REGISTRY) == [
        "wan22-i2v-high", "wan22-lightx2v-high", "luts",
    ]  # fmt: skip
    low = {"ref": "test.expert@1", "params": {"weights": "wan22-i2v-low"}}
    seq = NodeSpec(kind="seq", children=(op(), op(expert=low, label="y")))
    assert spec_assets(seq, registry=REGISTRY) == [
        "wan22-i2v-high", "wan22-lightx2v-high", "luts", "wan22-i2v-low",
    ]  # fmt: skip


def test_parameters_are_bound_first() -> None:
    spec = op(table={"$param": "table", "default": "luts"})
    assert spec_assets(spec, {"table": "luts-v2"}, registry=REGISTRY)[-1] == "luts-v2"
    assert spec_assets(spec, registry=REGISTRY)[-1] == "luts"


def test_errors_point_at_the_node() -> None:
    bad = NodeSpec(kind="seq", children=(op(), op(expert={"ref": "test.nope@1"})))
    with pytest.raises(SpecError, match="param 'expert'") as raised:
        spec_assets(bad, registry=REGISTRY)
    assert (
        str(raised.value.path) == str(raised.value.path)
        and raised.value.path is not None
    )
    with pytest.raises(SpecError, match="Cannot resolve operator"):
        spec_assets(NodeSpec(kind="op", op="test.missing@1"), registry=REGISTRY)
    ref = NodeSpec(kind="pipeline_ref", params={"pipeline": "other"})
    with pytest.raises(SpecError, match="inline pipeline_ref"):
        spec_assets(ref, registry=REGISTRY)


def test_cards_mark_asset_params() -> None:
    params: Any = REGISTRY.get("test.animate@1").card()["params"]
    by_name = {p["name"]: p for p in params}
    assert by_name["table"] == {
        "name": "table", "required": False, "type": "str", "asset": True, "default": "luts",
    }  # fmt: skip
    assert "asset" not in by_name["label"]
    assert by_name["expert"]["resource"] == "test.expert@1"
