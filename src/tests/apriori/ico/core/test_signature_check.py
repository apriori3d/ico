from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Literal, Optional, Protocol, TypeVar, Union

import pytest

from ico.core.chain import IcoChain, chain
from ico.core.operator import IcoOperator
from ico.core.signature_check import (
    IcoSignatureError,
    is_assignable,
    is_strict,
    strict_signatures,
    validate_signatures,
)

T = TypeVar("T")


# ─── Fixtures ───


def identity(x: T) -> T:
    return x


def to_str(x: int) -> str:
    return str(x)


def str_len(x: str) -> int:
    return len(x)


def double(x: int) -> int:
    return x * 2


def to_ints(x: str) -> list[int]:
    return [len(x)]


def total(xs: Iterable[int]) -> int:
    return sum(xs)


def maybe_int(x: str) -> int | None:
    return len(x) or None


class Base:
    pass


class Derived(Base):
    pass


def make_derived(x: int) -> Derived:
    return Derived()


def use_base(x: Base) -> int:
    return 0


class Sized(Protocol):
    def size(self) -> int: ...


# ─── is_assignable ───


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (int, int),
        (bool, int),
        (Derived, Base),
        (int, object),
        (int, int | str),
        (int, Optional[int]),  # noqa: UP007 - legacy typing form on purpose
        (type(None), Optional[int]),  # noqa: UP007
        (Union[int, str], int | str | float),  # noqa: UP007
        (list[int], Iterable[int]),
        (list[int], Sequence[int]),
        (Iterator[int], Iterator[int]),
        (dict[str, int], Mapping[str, int]),
        (list[bool], list[int]),
        (list[int], list),
        (tuple[int, int], tuple[int, ...]),
        (tuple[int, str], tuple[int, str]),
    ],
)
def test_assignable(src: Any, dst: Any) -> None:
    assert is_assignable(src, dst) is True


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (str, int),
        (Base, Derived),
        (float, int),
        (int | str, int),
        (list[str], Iterable[int]),
        (Iterable[int], list[int]),
        (dict[str, int], Mapping[str, str]),
        (tuple[int, int], tuple[int, int, int]),
        (tuple[str, ...], tuple[int, ...]),
    ],
)
def test_not_assignable(src: Any, dst: Any) -> None:
    assert is_assignable(src, dst) is False


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (Any, int),
        (int, Any),
        (type(Any), str),
        (str, type[Any]),
    ],
)
def test_any_is_assignable(src: Any, dst: Any) -> None:
    assert is_assignable(src, dst) is True


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (T, int),
        (int, T),
        (Literal["a"], str),
        (int, Sized),
        (list, list[int]),
        (dict[str, int], Iterable[str]),
        (tuple[int, ...], tuple[int, int]),
    ],
)
def test_undecidable(src: Any, dst: Any) -> None:
    assert is_assignable(src, dst) is None


def test_union_unknown_member_is_undecidable() -> None:
    assert is_assignable(int, str | Sized) is None


# ─── Strict mode ───


def test_strict_mode_off_by_default() -> None:
    assert not is_strict()
    flow = IcoOperator(to_str) | IcoOperator(double)  # type: ignore[arg-type]
    assert isinstance(flow, IcoChain)


def test_strict_context_restores_previous_value() -> None:
    with strict_signatures():
        assert is_strict()
        with strict_signatures(False):
            assert not is_strict()
        assert is_strict()
    assert not is_strict()


def test_strict_accepts_compatible_chain() -> None:
    with strict_signatures():
        flow = IcoOperator(to_str) | IcoOperator(str_len) | IcoOperator(double)
    assert flow(123) == 6


def test_strict_rejects_mismatch() -> None:
    with strict_signatures(), pytest.raises(IcoSignatureError) as exc_info:
        _ = IcoOperator(to_str) | IcoOperator(double)  # type: ignore[arg-type]

    issue = exc_info.value.issue
    assert issue.kind == "mismatch"
    assert issue.output is str
    assert issue.input is int
    assert issue.step == 2
    assert str(exc_info.value) == "step 2 (to_str → double): expected int, got str"


def test_strict_error_is_type_error() -> None:
    with strict_signatures(), pytest.raises(TypeError):
        _ = IcoOperator(to_str) | IcoOperator(double)  # type: ignore[arg-type]


def test_strict_accepts_subclass_and_generics() -> None:
    with strict_signatures():
        _ = IcoOperator(make_derived) | IcoOperator(use_base)
        _ = IcoOperator(to_ints) | IcoOperator(total)


def test_strict_rejects_optional_into_plain() -> None:
    with strict_signatures(), pytest.raises(IcoSignatureError):
        _ = IcoOperator(maybe_int) | IcoOperator(double)  # type: ignore[arg-type]


def test_strict_skips_untyped_operators() -> None:
    with strict_signatures():
        flow = IcoOperator(to_str) | IcoOperator(lambda x: x) | IcoOperator(double)  # type: ignore
    assert flow(2) == "22"  # type: ignore


def test_strict_uses_inner_boundary_of_nested_chains() -> None:
    left = IcoOperator(double) | IcoOperator(to_str)
    right = IcoOperator(str_len) | IcoOperator(double)

    with strict_signatures():
        _ = chain(left, right)
        with pytest.raises(IcoSignatureError):
            _ = chain(left, left)  # type: ignore[arg-type]


def test_strict_flag_on_chain_overrides_context() -> None:
    with pytest.raises(IcoSignatureError):
        _ = chain(IcoOperator(to_str), IcoOperator(double), strict=True)  # type: ignore[arg-type]

    with strict_signatures():
        _ = chain(IcoOperator(to_str), IcoOperator(double), strict=False)  # type: ignore[arg-type]


# ─── validate_signatures ───


def test_validate_compatible_flow() -> None:
    flow = IcoOperator(to_str) | IcoOperator(str_len) | IcoOperator(double)
    assert validate_signatures(flow) == []


def test_validate_collects_all_mismatches() -> None:
    # Typed as Any → Any so that only the runtime check sees the mismatch.
    mistyped: IcoOperator[Any, Any] = IcoOperator(double)
    flow = IcoOperator(to_str) | mistyped | IcoOperator(str_len)
    issues = validate_signatures(flow)

    assert [i.kind for i in issues] == ["mismatch", "mismatch"]
    assert all(i.is_error for i in issues)
    assert {(i.left, i.right) for i in issues} == {
        ("to_str", "double"),
        ("double", "str_len"),
    }


def test_validate_reports_uninferred_as_warning() -> None:
    flow = IcoOperator(to_str) | IcoOperator(identity) | IcoOperator(str_len)
    assert not IcoOperator(identity).signature.infered

    issues = validate_signatures(flow)

    assert [i.kind for i in issues] == ["uninferred", "uninferred"]
    assert not any(i.is_error for i in issues)
    assert "not checked" in issues[0].message


def test_strict_does_not_raise_on_uninferred() -> None:
    with strict_signatures():
        flow = IcoOperator(to_str) | IcoOperator(identity) | IcoOperator(str_len)
    assert flow(100) == 3
