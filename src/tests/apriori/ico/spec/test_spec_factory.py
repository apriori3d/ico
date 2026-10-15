"""Rebuilding factory-based async streams and agents with SpecFactory."""

import os
import pickle
from typing import cast

import pytest

from ico.core.async_stream import IcoAsyncStream, RetryPolicy
from ico.core.node import IcoNodeProtocol
from ico.core.operator import IcoOperator
from ico.core.runtime.progress import IcoProgress
from ico.core.source import IcoSource
from ico.registry import IcoRegistry, register
from ico.runtime.agent.mp.mp_agent import MPAgent
from ico.spec import (
    NodeSpec,
    PipelineSpec,
    SpecError,
    SpecFactory,
    build,
    from_json,
    to_json,
    to_spec,
)

# Module-level registrations: a spawned MPAgent worker imports this module
# to unpickle the SpecFactory, which re-registers them in its own process.
REGISTRY = IcoRegistry()


@register("inc", 1, registry=REGISTRY)
def inc(x: int) -> int:
    return x + 1


@register("scale", 1, registry=REGISTRY)
class Scale(IcoOperator[int, int]):
    """Multiply by a factor; the factor would be config such as a model path."""

    def __init__(self, factor: int) -> None:
        self.factor = factor
        super().__init__(fn=self._apply)

    def _apply(self, x: int) -> int:
        return x * self.factor


@register("worker_pid", 1, registry=REGISTRY)
def worker_pid(x: int) -> int:
    return os.getpid()


def make_pid_worker() -> IcoOperator[int, int]:
    return cast(IcoOperator[int, int], worker_pid)


def make_worker() -> IcoOperator[int, int]:
    """A factory with configuration in its closure, as in existing flows."""
    return cast(IcoOperator[int, int], Scale(3) | inc)


def roundtrip(flow: IcoNodeProtocol) -> IcoNodeProtocol:
    spec = to_spec(flow, strict=True)
    rebuilt = build(from_json(to_json(spec)), REGISTRY)
    assert to_spec(rebuilt, strict=True) == spec
    return rebuilt


def run(flow: IcoNodeProtocol, items: list[int]) -> list[int]:
    return sorted(flow(iter(items)))  # type: ignore[operator]


# ─── Factory-based async stream ───


def test_factory_stream_rebuilt_with_distinct_workers() -> None:
    stream = IcoAsyncStream(make_worker, pool_size=3, ordered=True)
    node = roundtrip(stream)
    assert isinstance(node, IcoAsyncStream)
    rebuilt = cast(IcoAsyncStream[int, int], node)

    assert rebuilt.pool_from_factory and rebuilt.pool_size == 3
    assert list(rebuilt(iter(range(10)))) == list(stream(iter(range(10))))

    workers = rebuilt.pool
    assert len({id(w) for w in workers}) == 3
    scales = [w.children[0] for w in workers]
    assert all(isinstance(s, Scale) and s.factor == 3 for s in scales)
    assert len({id(s) for s in scales}) == 3
    assert len({id(w.children[1]) for w in workers}) == 3  # wrapped functions too


def test_explicit_pool_stream_roundtrip() -> None:
    stream = IcoAsyncStream([Scale(2), Scale(5)])
    rebuilt = roundtrip(stream)
    assert run(rebuilt, [1, 2]) == run(stream, [1, 2])


def test_error_handling_roundtrip() -> None:
    policy = RetryPolicy(
        max_retries=2, delay=0.0, backoff=2.0, retry_on=(TimeoutError,)
    )
    stream = IcoAsyncStream(make_worker, pool_size=2, on_error="skip", retry=policy)
    rebuilt = roundtrip(stream)

    assert isinstance(rebuilt, IcoAsyncStream)
    assert rebuilt.on_error == "skip"
    assert rebuilt.retry == policy


def test_spec_factory_is_picklable() -> None:
    rebuilt = build(to_spec(MPAgent(make_worker), strict=True), REGISTRY)
    assert isinstance(rebuilt, MPAgent)
    factory = rebuilt.flow_factory
    assert isinstance(factory, SpecFactory)

    restored = pickle.loads(pickle.dumps(factory))
    assert restored == factory
    worker = restored()
    assert worker(4) == 13
    assert worker is not restored()
    assert restored.probe_signature().format() == "int → int"


# ─── Agents ───


def test_agent_roundtrip_and_run() -> None:
    agent = MPAgent(make_worker)
    node = roundtrip(agent)
    assert isinstance(node, MPAgent)
    rebuilt = cast(MPAgent[int, int], node)

    assert isinstance(rebuilt.flow_factory, SpecFactory)
    assert rebuilt.signature.format() == agent.signature.format() == "int → int"

    active = rebuilt.activate()
    try:
        flow = IcoSource[int](lambda: iter([1, 2, 3])) | IcoAsyncStream([active])
        assert list(flow(None)) == [4, 7, 10]
    finally:
        active.deactivate()


def test_rebuilt_agent_runs_in_another_process() -> None:
    node = build(to_spec(MPAgent(make_pid_worker), strict=True), REGISTRY)
    assert isinstance(node, MPAgent)
    rebuilt = cast(MPAgent[int, int], node)

    active = rebuilt.activate()
    try:
        flow = IcoSource[int](lambda: iter([0])) | IcoAsyncStream([active])
        (pid,) = list(flow(None))
        assert pid != os.getpid()
    finally:
        active.deactivate()


def make_untyped_worker() -> IcoOperator[int, int]:
    # An unparametrized progress node makes the worker signature uninferred.
    progress: IcoProgress[int] = IcoProgress(total=3)
    return cast(IcoOperator[int, int], progress | inc)


def test_agent_with_uninferred_worker_roundtrip() -> None:
    # No return annotation on the factory: the agent signature is uninferred.
    agent = MPAgent(lambda: make_untyped_worker())  # pyright: ignore[reportUnknownLambdaType]
    assert not agent.signature.infered

    node = roundtrip(agent)
    assert isinstance(node, MPAgent)
    assert not cast(MPAgent[int, int], node).signature.infered


def test_agent_spec_has_no_factory_ref() -> None:
    root = to_spec(MPAgent(make_worker), strict=True).root
    assert root.ref is None
    assert root.children[0].kind == "seq"


def test_unknown_agent_type() -> None:
    root = NodeSpec(
        kind="agent",
        params={"type": "somewhere:OtherAgent"},
        children=(NodeSpec(kind="op", op="inc@1"),),
    )
    with pytest.raises(SpecError, match="only 'ico.runtime.agent.mp.mp_agent:MPAgent'"):
        build(PipelineSpec(root=root), REGISTRY)


# ─── Restrictions ───


def test_registered_instance_rejected_in_factory_workers() -> None:
    registry = IcoRegistry()
    flow = register("composite", 1, registry=registry)(
        IcoOperator(inc) | IcoOperator(inc)
    )
    stream = IcoAsyncStream(lambda: cast(IcoOperator[int, int], flow), pool_size=2)
    spec = to_spec(stream)

    with pytest.raises(SpecError, match="registered instance and cannot be recreated"):
        build(spec, registry)


def test_non_builtin_retry_exception_rejected() -> None:
    root = NodeSpec(
        kind="async_stream",
        params={
            "pool_size": 1,
            "ordered": False,
            "factory": True,
            "retry": {
                "max_retries": 1,
                "delay": 0.0,
                "backoff": 1.0,
                "retry_on": ["os:error"],
            },
        },
        children=(NodeSpec(kind="op", op="inc@1"),),
    )
    with pytest.raises(SpecError, match="only builtin exception types"):
        build(PipelineSpec(root=root), REGISTRY)
