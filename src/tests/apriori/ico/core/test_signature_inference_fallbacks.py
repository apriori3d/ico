"""Signature inference must degrade to "uninferred" instead of raising."""

from collections.abc import Callable, Iterator
from typing import Any, TypeVar, cast

import pytest

from ico.core.batcher import IcoBatcher
from ico.core.operator import IcoOperator
from ico.core.signature_check import validate_signatures
from ico.core.signature_utils import (
    format_ico_type,
    infer_from_callable,
    infer_from_flow_factory,
)
from ico.core.sink import IcoSink
from ico.core.source import IcoSource


def numbers() -> Iterator[int]:
    return iter([1, 2, 3])


T = TypeVar("T")


# ─── Unbound TypeVars from generic resolution ───


def test_unparametrized_generic_node_is_uninferred() -> None:
    signature = IcoBatcher(3).signature
    assert not signature.infered
    assert "IInv" not in signature.format()


def test_parametrized_generic_node_is_inferred() -> None:
    signature = IcoBatcher[int](3).signature
    assert signature.infered
    assert signature.format() == "Iterator[int] → Iterator[Iterator[int]]"


def test_unbound_typevar_link_reported_as_uninferred() -> None:
    batcher: IcoBatcher[Any] = IcoBatcher(2)  # not parametrized at runtime
    flow = IcoSource(numbers) | batcher
    assert [i.kind for i in validate_signatures(flow)] == ["uninferred"]


# ─── Callables that cannot be introspected ───


def no_signature(x: int) -> int:
    return x


# inspect.signature() raises for an invalid __signature__ on every Python
# version, like it does for builtins without signature metadata (e.g. print
# on 3.10; print has a text signature since 3.11).
no_signature.__signature__ = "invalid"  # type: ignore[attr-defined]


def test_callable_without_signature_is_uninferred() -> None:
    assert infer_from_callable(no_signature) is None
    assert not IcoOperator(no_signature).signature.infered


def test_builtin_sink_signature_does_not_raise() -> None:
    assert IcoSink(print).signature is not None


def test_describe_with_builtin_sink(capsys: pytest.CaptureFixture[str]) -> None:
    (IcoSource(numbers) | IcoSink(print)).describe()
    assert "IcoSink" in capsys.readouterr().out


def test_unresolvable_forward_reference_is_uninferred() -> None:
    def broken(x: "MissingType") -> int:  # type: ignore[name-defined]  # noqa: F821
        return 0

    fn = cast(Callable[[int], int], broken)
    assert infer_from_callable(fn) is None
    assert not IcoOperator(fn).signature.infered


# ─── Flow factories ───


def test_untyped_flow_factory_is_uninferred() -> None:
    assert infer_from_flow_factory(lambda: IcoOperator[int, int](abs)) is None


def test_flow_factory_with_non_operator_return_is_uninferred() -> None:
    def factory() -> int:
        return 0

    assert infer_from_flow_factory(factory) is None


def test_typed_flow_factory_is_inferred() -> None:
    def factory() -> IcoOperator[int, str]:
        return IcoOperator[int, str](str)

    signature = infer_from_flow_factory(factory)
    assert signature is not None
    assert signature.format() == "int → str"


# ─── Formatting ───


def test_format_any_encodings() -> None:
    assert format_ico_type(Any) == "Any"
    assert format_ico_type(type(Any)) == "Any"
    assert format_ico_type(type[Any]) == "Any"


def test_chain_fallback_signature_formats_as_any() -> None:
    # A TypeVar-only operator is uninferred on every Python version (an
    # `Any` annotation is inferred since 3.11, where typing.Any is a class).
    untyped: IcoOperator[Any, Any] = IcoOperator(identity_generic)
    signature = (untyped | IcoOperator(to_str_typed)).signature
    assert not signature.infered
    assert signature.format() == "Any → Any"


def identity_generic(x: T) -> T:
    return x


def to_str_typed(x: int) -> str:
    return str(x)
