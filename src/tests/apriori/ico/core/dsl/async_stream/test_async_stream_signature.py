from typing import Any, TypeVar

from ico.core.async_operator import IcoAsyncOperator
from ico.core.async_stream import IcoAsyncStream
from ico.core.operator import IcoOperator

T = TypeVar("T")


def to_str(x: int) -> str:
    return str(x)


async def async_to_str(x: int) -> str:
    return str(x)


def identity(x: T) -> T:
    return x


def test_signature_from_worker() -> None:
    stream = IcoAsyncStream([IcoOperator(to_str)])
    assert stream.signature.format() == "Iterator[int] → Iterator[str]"


def test_signature_from_async_worker() -> None:
    stream = IcoAsyncStream([IcoAsyncOperator(async_to_str)])
    assert stream.signature.format() == "Iterator[int] → Iterator[str]"


def test_parametrized_stream_is_not_wrapped_twice() -> None:
    stream = IcoAsyncStream[int, str]([IcoOperator(to_str)])
    assert stream.signature.format() == "Iterator[int] → Iterator[str]"


def test_factory_stream_signature() -> None:
    def make_worker() -> IcoOperator[int, str]:
        return IcoOperator(to_str)

    stream = IcoAsyncStream(make_worker, pool_size=2)
    assert stream.signature.format() == "Iterator[int] → Iterator[str]"


def test_uninferred_worker_gives_uninferred_stream() -> None:
    worker: IcoOperator[Any, Any] = IcoOperator(identity)
    assert not worker.signature.infered
    assert not IcoAsyncStream([worker]).signature.infered
