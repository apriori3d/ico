"""Resources: declaration, the pool (use / release, budgets, sharing),
resource params of operators in cards and specs, and profile rows."""

import os
import threading
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

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
from ico.resources.device import is_out_of_memory
from ico.resources.pool import gpu_headroom, memory, ram_budget
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


@resource("test.engine", 1)
@dataclass(frozen=True)
class FakeEngine(Resource[Model]):
    """An engine of ``gb`` gigabytes built on the GPU."""

    resident: ClassVar[str | None] = "cuda"
    name: str = "e"
    gb: float = 4.0

    def load(self) -> Model:
        LOADS[self.name] += 1
        engine = Model(self.name)
        engine.device = "cuda"
        return engine

    def to(self, obj: Model, device: str) -> Model:
        raise AssertionError("a resident object never moves")

    def size(self, obj: Model) -> int:
        return int(self.gb * GB)

    def close(self, obj: Model) -> None:
        CLOSED.append(obj.name)


class FakeGpu:
    """One GPU of ``total`` GB shared by pools (processes): its used memory
    is what they hold on it, plus ``other``."""

    def __init__(self, total: float = 10) -> None:
        self.total = int(total * GB)
        self.other = 0
        self.pools: list[ResourcePool] = []
        self.emptied = 0

    def process(self, *, worker: bool = False) -> ResourcePool:
        pool = ResourcePool(
            ram=100 * GB, headroom=0.2, device=_GpuView(self), keep_idle=not worker
        )
        self.pools.append(pool)
        return pool


class _GpuView:
    def __init__(self, gpu: FakeGpu) -> None:
        self.gpu = gpu

    def memory(self, device: str) -> tuple[int, int] | None:
        if device != "cuda":
            return None
        held = [h for p in self.gpu.pools for h in p.held() if h["device"] == "cuda"]
        used = self.gpu.other + sum(h["bytes"] or 0 for h in held)
        return self.gpu.total - used, self.gpu.total

    def empty_cache(self, device: str) -> None:
        self.gpu.emptied += 1


@pytest.fixture(autouse=True)
def fresh_pool() -> Iterator[ResourcePool]:
    LOADS.clear()
    CLOSED.clear()
    new = ResourcePool(ram=10 * GB)  # no CUDA here: idle objects go back to RAM
    old = set_pool(new)
    yield new
    set_pool(old)


def test_a_resource_loads_once_and_returns_to_ram(fresh_pool: ResourcePool) -> None:
    res = FakeModel()
    with res.use("cuda") as model:
        assert model.device == "cuda"
    assert model.device == "cpu"  # released: back to RAM (the GPU is unknown)
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


def test_idle_objects_stay_on_the_gpu_while_it_has_headroom() -> None:
    gpu = FakeGpu(total=10)
    set_pool(gpu.process())
    with FakeModel(gb=3).use("cuda") as a:
        pass
    with FakeModel(name="b", gb=3).use("cuda") as b:
        pass
    assert a.moves == ["cuda"] and b.moves == ["cuda"]  # 4 GB free, 2 kept
    with FakeModel(name="c", gb=3).use("cuda") as c:
        assert a.moves == ["cuda", "cpu"]  # room for c: the least recently used
        assert b.device == "cuda"
    assert c.device == "cuda" and a.device == "cpu"


def test_below_the_headroom_idle_objects_go_back_to_ram() -> None:
    gpu = FakeGpu(total=10)
    set_pool(gpu.process())
    with FakeModel(gb=3).use("cuda") as a:
        pass
    gpu.other = 6 * GB  # another process takes memory: 1 GB free
    with FakeModel(name="b", gb=0.5).use():  # a release checks the headroom
        pass
    assert a.device == "cpu"


def test_resident_objects_live_on_the_gpu_and_are_dropped_for_room() -> None:
    gpu = FakeGpu(total=10)
    pool = gpu.process()
    pool.ram = 1 * GB  # engines do not count against RAM
    set_pool(pool)
    with FakeEngine().use("cuda") as engine:
        assert engine.device == "cuda"
    assert [(h["device"], h["bytes"]) for h in pool.held()] == [("cuda", 4 * GB)]
    with FakeEngine(name="f").use("cuda"):  # 2 GB free: both stay
        pass
    with FakeEngine(name="g").use("cuda"):  # never loaded: sized as e and f
        assert CLOSED == ["e"]  # room made before its load
    assert CLOSED == ["e"] and gpu.emptied >= 1
    with FakeEngine().use("cuda"):  # its size is known: room before the load
        assert CLOSED == ["e", "f"]
    assert LOADS["e"] == 2
    with (
        pytest.raises(RuntimeError, match="lives on cuda, asked for cpu"),
        FakeEngine().use("cpu"),
    ):
        pass


def test_a_worker_process_keeps_only_resident_objects_on_the_gpu() -> None:
    gpu = FakeGpu(total=10)
    worker = gpu.process(worker=True)  # cannot give memory to the main one
    with worker.use(FakeModel(gb=1), "cuda") as model:
        pass
    with worker.use(FakeEngine(), "cuda") as engine:
        pass
    assert model.device == "cpu"  # back to RAM at once
    assert engine.device == "cuda" and CLOSED == []  # the headroom holds


def test_after_an_out_of_memory_error_idle_objects_leave_the_gpu() -> None:
    gpu = FakeGpu(total=10)
    set_pool(gpu.process())

    class OutOfMemoryError(RuntimeError):
        pass

    with FakeModel(gb=1).use("cuda") as a:
        pass
    with pytest.raises(OutOfMemoryError), FakeModel(name="b").use("cuda") as b:
        raise OutOfMemoryError("CUDA out of memory")
    assert a.device == "cpu" and b.device == "cpu"
    assert is_out_of_memory(RuntimeError("CUDA error: out of memory"))
    assert not is_out_of_memory(ValueError("out of memory"))


def test_the_headroom_is_a_share_of_the_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    assert gpu_headroom(None) == gpu_headroom("") == 0.2
    assert gpu_headroom("35%") == 0.35 and gpu_headroom("0%") == 0
    for bad in ("100%", "-1%", "2GB", "x%"):
        with pytest.raises(ValueError, match="not a share of the GPU"):
            gpu_headroom(bad)
    monkeypatch.setenv("ICO_RESOURCES_GPU_FREE", "30%")
    assert ResourcePool().headroom == 0.3


def test_a_ram_budget_is_a_size_or_a_share_of_the_memory(tmp_path: Path) -> None:
    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    assert memory(tmp_path) == physical  # no cgroup files: the host
    (tmp_path / "memory.max").write_text("max\n")
    assert memory(tmp_path) == physical  # v2 without a limit
    (tmp_path / "memory.max").write_text(f"{4 * GB}\n")
    assert memory(tmp_path) == 4 * GB  # a container: its limit, not the host
    (tmp_path / "memory.max").unlink()
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text(f"{2 * GB}\n")
    assert memory(tmp_path) == 2 * GB  # v1
    assert ram_budget("80%", tmp_path) == int(1.6 * GB)
    assert ram_budget(None, tmp_path) == ram_budget("", tmp_path) == 1 * GB
    assert ram_budget("3GB", tmp_path) == 3 * GB
    for bad in ("0%", "120%", "x%"):
        with pytest.raises(ValueError, match="not a share"):
            ram_budget(bad, tmp_path)


def test_the_pool_reads_its_ram_budget_from_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ICO_RESOURCES_RAM", "25%")
    assert ResourcePool().ram == ram_budget("25%")
    monkeypatch.setenv("ICO_RESOURCES_RAM", "7GB")
    assert ResourcePool().ram == 7 * GB


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
