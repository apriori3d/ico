import asyncio
from typing import Any

import pytest

from ico.core.async_operator import IcoAsyncOperator
from ico.core.async_stream import (
    IcoAsyncStream,
    IcoStreamErrors,
    ItemError,
    OnError,
    RetryPolicy,
)
from ico.core.operator import IcoOperator
from ico.spec import to_spec


async def fail_on_odd(x: int) -> int:
    await asyncio.sleep(0.001 * (x % 3))
    if x % 2:
        raise ValueError(f"odd {x}")
    return x * 10


def make_stream(
    on_error: OnError,
    ordered: bool,
    retry: int | RetryPolicy | None = None,
    op: Any = fail_on_odd,
) -> IcoAsyncStream[int, int]:
    return IcoAsyncStream(
        [IcoAsyncOperator[int, int](op) for _ in range(3)],
        ordered=ordered,
        on_error=on_error,
        retry=retry,
    )


class Flaky:
    """Fails the first `failures` calls per item, then succeeds."""

    def __init__(self, failures: int, error: type[Exception] = TimeoutError) -> None:
        self.failures = failures
        self.error = error
        self.calls: dict[int, int] = {}

    async def __call__(self, x: int) -> int:
        self.calls[x] = self.calls.get(x, 0) + 1
        if self.calls[x] <= self.failures:
            raise self.error(f"attempt {self.calls[x]} for {x}")
        return x


# ─── on_error ───


def test_default_is_raise() -> None:
    stream = IcoAsyncStream([IcoAsyncOperator(fail_on_odd)])
    assert stream.on_error == "raise"
    assert stream.retry == RetryPolicy()


@pytest.mark.parametrize("ordered", [False, True])
def test_raise(ordered: bool) -> None:
    with pytest.raises(ValueError, match="odd"):
        list(make_stream("raise", ordered)(iter(range(6))))


@pytest.mark.parametrize("ordered", [False, True])
def test_skip(ordered: bool) -> None:
    result = list(make_stream("skip", ordered)(iter(range(10))))
    expected = [0, 20, 40, 60, 80]
    assert (result if ordered else sorted(result)) == expected


@pytest.mark.parametrize("ordered", [False, True])
def test_collect_emits_results_then_raises(ordered: bool) -> None:
    emitted: list[int] = []
    with pytest.raises(IcoStreamErrors) as exc_info:
        for value in make_stream("collect", ordered)(iter(range(10))):
            emitted.append(value)

    assert (emitted if ordered else sorted(emitted)) == [0, 20, 40, 60, 80]

    errors = exc_info.value.errors
    assert [e.index for e in errors] == [1, 3, 5, 7, 9]
    assert [e.item for e in errors] == [1, 3, 5, 7, 9]
    assert all(isinstance(e.error, ValueError) for e in errors)
    assert all(e.attempts == 1 for e in errors)
    assert isinstance(exc_info.value, RuntimeError)


def test_collect_message_lists_first_errors() -> None:
    errors = [ItemError(i, i, ValueError(f"e{i}"), 1) for i in range(7)]
    message = str(IcoStreamErrors(errors))
    assert message.startswith("7 stream item(s) failed: [0] ValueError: e0; ")
    assert message.endswith("; and 2 more")


def test_collect_without_failures_does_not_raise() -> None:
    async def ok(x: int) -> int:
        return x

    assert sorted(make_stream("collect", False, op=ok)(iter(range(5)))) == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_skip_everything() -> None:
    async def always_fail(x: int) -> int:
        raise ValueError(x)

    assert list(make_stream("skip", True, op=always_fail)(iter(range(5)))) == []


def test_input_error_takes_precedence_over_collected_errors() -> None:
    def source() -> Any:
        yield from range(4)
        raise OSError("source broke")

    with pytest.raises(OSError, match="source broke"):
        list(make_stream("collect", True)(source()))


def test_invalid_on_error() -> None:
    with pytest.raises(ValueError, match="on_error"):
        IcoAsyncStream([IcoOperator[int, int](abs)], on_error="ignore")  # type: ignore[arg-type,call-overload]


# ─── retry ───


@pytest.mark.parametrize("ordered", [False, True])
def test_retry_succeeds(ordered: bool) -> None:
    flaky = Flaky(failures=2)
    result = list(make_stream("raise", ordered, retry=2, op=flaky)(iter(range(5))))

    assert (result if ordered else sorted(result)) == [0, 1, 2, 3, 4]
    assert all(calls == 3 for calls in flaky.calls.values())


def test_retry_exhausted_reports_attempts() -> None:
    flaky = Flaky(failures=5)
    with pytest.raises(IcoStreamErrors) as exc_info:
        list(make_stream("collect", True, retry=2, op=flaky)(iter([7])))

    (error,) = exc_info.value.errors
    assert (error.index, error.item, error.attempts) == (0, 7, 3)
    assert str(error.error) == "attempt 3 for 7"


def test_retry_only_listed_exceptions() -> None:
    flaky = Flaky(failures=1, error=KeyError)
    policy = RetryPolicy(max_retries=3, retry_on=(TimeoutError,))

    with pytest.raises(KeyError):
        list(make_stream("raise", True, retry=policy, op=flaky)(iter([1])))
    assert flaky.calls == {1: 1}


def test_retry_delays_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float, *args: Any) -> None:
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    policy = RetryPolicy(max_retries=3, delay=0.5, backoff=2.0)
    stream = IcoAsyncStream(
        [IcoAsyncOperator[int, int](Flaky(failures=3))], retry=policy
    )
    assert list(stream(iter([1]))) == [1]
    assert delays == [0.5, 1.0, 2.0]


def test_retry_sync_operator() -> None:
    calls: list[int] = []

    def flaky_sync(x: int) -> int:
        calls.append(x)
        if len(calls) < 2:
            raise TimeoutError
        return x

    stream = IcoAsyncStream([IcoOperator(flaky_sync)], retry=1)
    assert list(stream(iter([5]))) == [5]
    assert calls == [5, 5]


@pytest.mark.parametrize(
    "kwargs",
    [{"max_retries": -1}, {"delay": -0.1}, {"backoff": 0}],
)
def test_invalid_retry_policy(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


def test_retry_int_shorthand() -> None:
    stream = make_stream("raise", False, retry=4)
    assert stream.retry == RetryPolicy(max_retries=4)


# ─── Spec and describe ───


def double(x: int) -> int:
    return x * 2


def test_spec_omits_default_error_handling() -> None:
    root = to_spec(IcoAsyncStream([IcoOperator(double)])).root
    assert set(root.params) == {"pool_size", "ordered", "factory"}


def test_spec_exports_error_handling() -> None:
    policy = RetryPolicy(
        max_retries=2, delay=0.1, backoff=2.0, retry_on=(TimeoutError,)
    )
    stream = IcoAsyncStream([IcoOperator(double)], on_error="collect", retry=policy)
    params = to_spec(stream).root.params

    assert params["on_error"] == "collect"
    assert params["retry"] == {
        "max_retries": 2,
        "delay": 0.1,
        "backoff": 2.0,
        "retry_on": ["builtins:TimeoutError"],
    }


def test_describe_shows_error_handling(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")  # avoid truncating the header
    IcoAsyncStream([IcoOperator(double)], on_error="skip", retry=3).describe()
    out = capsys.readouterr().out
    assert "on_error=skip" in out
    assert "retry=3" in out
