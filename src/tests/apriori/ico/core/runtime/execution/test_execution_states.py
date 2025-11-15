import pytest

from apriori.ico.core import IcoExecutionMixin, IcoExecutionState

# ──── Dummy operator for testing ────


class DummyOp(IcoExecutionMixin[int, int]):
    """Simple operator that doubles input using execution tracking."""

    def __call__(self, x: int) -> int:
        return self.track(lambda v: v * 2, x)


# ──── Test: normal execution ────


def test_execution_state_transitions_success() -> None:
    op = DummyOp()
    # Initially idle
    assert op.exec_state is IcoExecutionState.idle

    # Call operator
    result = op(3)

    # Verify result and final state
    assert result == 6
    assert op.exec_state is IcoExecutionState.done  # type: ignore[comparison-overlap]


# ──── Dummy operator that raises ────


class FailingOp(IcoExecutionMixin[int, int]):
    """Operator that raises an error during execution."""

    def __call__(self, x: int) -> int:
        return self.track(lambda _: int(1 / 0), x)


# ──── Test: faulted execution ────


def test_execution_state_transitions_failure() -> None:
    op = FailingOp()
    assert op.exec_state is IcoExecutionState.idle

    with pytest.raises(ZeroDivisionError):
        op(1)

    assert op.exec_state is IcoExecutionState.faulted  # type: ignore[comparison-overlap]


# ──── Test: on_exec_event hook ────


def test_on_exec_event_called() -> None:
    op = DummyOp()
    called = []

    # Monkeypatch hook
    def hook(state: IcoExecutionState) -> None:
        called.append(state)

    op.on_exec_event = hook  # type: ignore

    op(2)

    # Should receive 'running' and 'done'
    assert [s.name for s in called] == ["running", "done"]
