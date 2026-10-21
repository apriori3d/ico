"""Resources: declaration, the pool (use / release, budgets, sharing),
resource params of operators in cards and specs, and profile rows."""

import threading
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from ico import IcoOperator
from ico.registry import IcoRegistry, register
from ico.resources import (
    Resource,
    ResourcePool,
    from_spec,
    pool,
    resource,
    set_pool,
)
from ico.spec import NodeSpec, SpecError, build, to_spec
from ico.spec.run import run
from ico.tools.profiler import ProfilerTool

GB = 10**9
LOADS: Counter[str] = Counter()
CLOSED: list[str] = []


class Model:
    def __init__(self, name: str) -> None:
        self.name = name
        self.device = "cpu"
        self.moves: list[str] = []

    def to(self, device: str) -> "Model":
        self.device = device
        self.moves.append(device)
        return self


@resource("test.model", 1)
@dataclass(frozen=True)
class FakeModel(Resource[Model]):
    """A model of ``gb`` gigabytes."""

    name: str = "a"
    gb: float = 1.0
    fail: bool = False
    slow: float = 0.0

    def load(self) -> Model:
        LOADS[self.name] += 1
        time.sleep(self.slow)
        if self.fail:
            raise OSError("no weights")
        return Model(self.name)

    def size(self, obj: Model) -> int:
        return int(self.gb * GB)

    def close(self, obj: Model) -> None:
        CLOSED.append(obj.name)


@resource("test.other", 1)
@dataclass(frozen=True)
class OtherModel(Resource[Model]):
    def load(self) -> Model:
        return Model("other")


@pytest.fixture(autouse=True)
def fresh_pool() -> Iterator[ResourcePool]:
    LOADS.clear()
    CLOSED.clear()
    new = ResourcePool(ram=10 * GB, gpu=0)
    old = set_pool(new)
    yield new
    set_pool(old)


def test_a_resource_loads_once_and_returns_to_ram(fresh_pool: ResourcePool) -> None:
    res = FakeModel()
    with res.use("cuda") as model:
        assert model.device == "cuda"
    assert model.device == "cpu"  # released: back to RAM (no GPU budget)
    with FakeModel().use("cuda") as again:  # an equal instance: the same object
        assert again is model
    assert LOADS["a"] == 1 and model.moves == ["cuda", "cpu", "cuda", "cpu"]
    with FakeModel(name="b").use() as other:
        assert other is not model
    assert [h["ref"] for h in fresh_pool.held()] == ["test.model@1", "test.model@1"]


def test_the_ram_budget_drops_the_least_recently_used() -> None:
    set_pool(ResourcePool(ram=int(2.5 * GB)))
    for name in ("a", "b", "c"):
        with FakeModel(name=name).use():
            pass
    assert CLOSED == ["a"]
    with FakeModel(name="a").use():
        pass
    assert LOADS["a"] == 2 and CLOSED == ["a", "b"]


def test_a_resource_in_use_is_never_dropped() -> None:
    set_pool(ResourcePool(ram=1 * GB))
    with FakeModel(name="a").use() as held:
        for name in ("b", "c"):
            with FakeModel(name=name).use():
                pass
        assert held.name == "a" and CLOSED == ["b", "c"]
    assert CLOSED == ["b", "c"]  # a alone fits the budget


def test_a_gpu_budget_keeps_idle_models_on_the_device() -> None:
    set_pool(ResourcePool(ram=10 * GB, gpu=1 * GB))
    with FakeModel().use("cuda") as model:
        pass
    with FakeModel().use("cuda"):
        pass
    assert model.moves == ["cuda"]
    with FakeModel(name="b").use("cuda") as other:  # over the budget: a goes back
        pass
    assert model.moves == ["cuda", "cpu"] and other.device == "cuda"


def test_two_users_must_agree_on_the_device() -> None:
    with FakeModel().use("cuda"):
        with (
            pytest.raises(RuntimeError, match="in use on cuda, asked for cpu"),
            FakeModel().use("cpu"),
        ):
            pass
        with FakeModel().use("cuda") as shared:
            assert shared.device == "cuda"


def test_concurrent_users_share_one_load() -> None:
    seen: list[Model] = []

    def work() -> None:
        with FakeModel(slow=0.05).use() as model:
            seen.append(model)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert LOADS["a"] == 1 and len({id(m) for m in seen}) == 1


def test_a_failed_load_leaves_nothing(fresh_pool: ResourcePool) -> None:
    with pytest.raises(OSError, match="no weights"), FakeModel(fail=True).use():
        pass
    assert fresh_pool.held() == []
    with pytest.raises(OSError):
        FakeModel(fail=True).preload()
    assert LOADS["a"] == 2


def test_a_label_tells_instances_apart() -> None:
    assert FakeModel().label == "test.model@1"
    assert FakeModel(name="high", gb=2.5).label == "test.model@1 (name=high, gb=2.5)"
    assert OtherModel().label == "test.other@1"


def test_declaration_rules() -> None:
    res = FakeModel(name="x", gb=2)
    assert res.ref == "test.model@1"
    assert res.spec() == {
        "ref": "test.model@1",
        "params": {"name": "x", "gb": 2, "fail": False, "slow": 0.0},
    }
    assert from_spec(res.spec(), FakeModel) == res
    with pytest.raises(ValueError, match="is a OtherModel, expected a FakeModel"):
        from_spec({"ref": "test.other@1"}, FakeModel)
    with pytest.raises(ValueError, match="cannot create test.model@1"):
        from_spec({"ref": "test.model@1", "params": {"nope": 1}}, FakeModel)
    with pytest.raises(TypeError, match="frozen dataclass"):
        resource("test.loose", 1)(Loose)
    with pytest.raises(ValueError, match="already"):
        resource("test.model", 1)(Clash)
    with pytest.raises(TypeError, match="field\\(s\\) size that hide methods"):
        resource("test.sized", 1)(Sized)


@dataclass
class Loose(Resource[int]):
    pass


@dataclass(frozen=True)
class Clash(Resource[int]):
    pass


@dataclass(frozen=True)
class Sized(Resource[int]):
    size: int = 1  # type: ignore[assignment]  # the hidden method under test


# ─── Operators ───

REGISTRY = IcoRegistry()
MODEL_A = FakeModel()


@register("test.describe", 1, registry=REGISTRY)
class Describe(IcoOperator[str, str]):
    """Name the model that handles a text."""

    def __init__(self, model: FakeModel = MODEL_A, prefix: str = "") -> None:
        self.model = model
        self.prefix = prefix
        super().__init__(fn=self._apply)

    def _apply(self, text: str) -> str:
        with self.model.use("cuda") as m:
            return f"{self.prefix}{m.name}:{text}:{m.device}"


def test_resource_params_in_cards_and_specs() -> None:
    card = REGISTRY.get("test.describe@1").card()
    params: Any = card["params"]
    assert params[0] == {
        "name": "model",
        "required": False,
        "type": "FakeModel",
        "resource": "test.model@1",
        "default": FakeModel().spec(),
    }

    exported = to_spec(Describe(FakeModel(name="b")))
    assert exported.root.params["model"] == FakeModel(name="b").spec()

    spec = NodeSpec(
        kind="op",
        op="test.describe@1",
        params={"model": {"ref": "test.model@1", "params": {"name": "c"}}},
    )
    node: Any = build(spec, registry=REGISTRY)
    assert node.model == FakeModel(name="c") and node("hi") == "c:hi:cuda"
    wrong = NodeSpec(
        kind="op", op="test.describe@1", params={"model": {"ref": "test.other@1"}}
    )
    with pytest.raises(SpecError, match="param 'model': test.other@1 is a OtherModel"):
        build(wrong, registry=REGISTRY)


def test_profile_rows_tell_uses_waits_and_reloads() -> None:
    set_pool(ResourcePool(ram=1 * GB))
    spec = NodeSpec(kind="seq", children=(
        NodeSpec(kind="op", op="test.describe@1"),
        NodeSpec(kind="op", op="test.describe@1", params={
            "model": {"ref": "test.model@1", "params": {"name": "b"}}}),
        NodeSpec(kind="op", op="test.describe@1"),
    ))  # fmt: skip
    profiler = ProfilerTool(interval=None)
    result = run(spec, "x", registry=REGISTRY, profiler=profiler)

    assert result.outputs == ("a:b:a:x:cuda:cuda:cuda",)
    rows = {r["key"]: r for r in profiler.snapshot()["resources"]}
    a = rows[FakeModel().key]
    assert rows[FakeModel(name="b").key]["label"] == "test.model@1 (name=b)"
    assert (a["uses"], a["loads"], a["reloads"], a["drops"]) == (2, 2, 1, 1)
    assert a["moves"] == 4 and a["devices"] == ["cuda"] and a["bytes"] == GB
    assert a["in_use"] >= 0 and a["load_seconds"] >= 0
    assert profiler.snapshot()["resource_peaks"]["cuda"] == GB
    assert pool().held()[0]["device"] == "cpu"
