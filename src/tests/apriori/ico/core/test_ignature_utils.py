# pyright: reportGeneralTypeIssues=false
# mypy: ignore-errors

from types import NoneType
from typing import Any, Generic

from ico.core.context_operator import C
from ico.core.operator import I, O
from ico.core.signature_utils import (
    replace_typevar,
    resolve_types_from_generic,
    type_contain_any_typevar,
)


def test_replace_typevar() -> None:
    # Simple replacement
    assert replace_typevar(I, I, int) is int
    assert replace_typevar(I, O, int) is I

    # Generuic type replacement
    assert replace_typevar(list[I], I, int) == list[int]
    assert replace_typevar(list[O], I, int) == list[O]
    # Nested typevars
    assert replace_typevar(list[list[I]], I, int) == list[list[int]]


def test_args_contain_typevars() -> None:
    # Simple typevars
    assert type_contain_any_typevar(I, I) is True
    assert type_contain_any_typevar(O, I) is False

    # Generic types with typevars
    assert type_contain_any_typevar(list[I], I) is True
    assert type_contain_any_typevar(list[O], I) is False
    assert type_contain_any_typevar(dict[str, I], I) is True
    assert type_contain_any_typevar(dict[str, O], I) is False

    # Nested typevars
    assert type_contain_any_typevar(list[list[I]], I) is True
    assert type_contain_any_typevar(list[list[O]], I) is False

    # Multimple typevars in target
    assert type_contain_any_typevar(tuple[I, O], I) is True
    assert type_contain_any_typevar(tuple[I, O], O) is True
    assert type_contain_any_typevar(tuple[I, O], C) is False

    # Multiple typevars in source
    assert type_contain_any_typevar(tuple[I, O], I, O) is True
    assert type_contain_any_typevar(tuple[I, O], I, C) is True
    assert type_contain_any_typevar(tuple[I, O], C, O) is True
    assert type_contain_any_typevar(tuple[I, O], C, C) is False


def test_resolve_types_from_generic_single_class_io() -> None:
    class Op(Generic[I, O]):
        pass

    # Resolved cases with generic hints
    assert resolve_types_from_generic(Op[int, str](), Op, I, O) == [int, str]
    assert resolve_types_from_generic(Op[str, int](), Op, I, O) == [str, int]
    assert resolve_types_from_generic(Op[None, int](), Op, I, O) == [NoneType, int]
    assert resolve_types_from_generic(Op[None, None](), Op, I, O) == [
        NoneType,
        NoneType,
    ]
    assert resolve_types_from_generic(Op[Any, Any](), Op, I, O) == [
        Any,
        Any,
    ]

    # Resolved cases with nested generic hints
    assert resolve_types_from_generic(Op[list[int], list[str]](), Op, I, O) == [
        list[int],
        list[str],
    ]

    # Unresolved case without generic hints
    assert resolve_types_from_generic(Op(), Op, I, O) == [None, None]  # pyright: ignore[reportUnknownArgumentType]


def test_resolve_types_from_generic_single_class_ico() -> None:
    class Op(Generic[I, C, O]):
        pass

    # Resolved cases with generic hints
    assert resolve_types_from_generic(Op[int, float, str](), Op, I, C, O) == [
        int,
        float,
        str,
    ]
    assert resolve_types_from_generic(Op[str, float, int](), Op, I, C, O) == [
        str,
        float,
        int,
    ]
    assert resolve_types_from_generic(Op[None, str, int](), Op, I, C, O) == [
        NoneType,
        str,
        int,
    ]
    assert resolve_types_from_generic(Op[None, None, None](), Op, I, C, O) == [
        NoneType,
        NoneType,
        NoneType,
    ]
    assert resolve_types_from_generic(Op[Any, Any, Any](), Op, I, C, O) == [
        Any,
        Any,
        Any,
    ]

    # Unresolved case without generic hints
    assert resolve_types_from_generic(Op(), Op, I, C, O) == [None, None, None]  # pyright: ignore[reportUnknownArgumentType]


def test_resolve_types_from_generic_multi_class_io() -> None:
    class OpA(Generic[I, O]):
        pass

    # Resolved case with one generic arg defined

    class OpB(Generic[O], OpA[int, O]):
        pass

    assert resolve_types_from_generic(OpB[str](), OpA, I, O) == [int, str]

    # Resolved case with single generic arg defined

    class OpBSingle(Generic[I], OpA[I, I]):
        pass

    assert resolve_types_from_generic(OpBSingle[int](), OpA, I, O) == [int, int]
    assert resolve_types_from_generic(OpBSingle(), OpA, I, O) == [None, None]  # pyright: ignore[reportUnknownArgumentType]

    # Resolved case with all generic args defined
    class OpC(OpB[str]):
        pass

    assert resolve_types_from_generic(OpC(), OpA, I, O) == [int, str]

    class OpCSingle(OpBSingle[str]):
        pass

    assert resolve_types_from_generic(OpCSingle(), OpA, I, O) == [str, str]


def test_resolve_types_from_generic_multi_class_ico() -> None:
    class OpA(Generic[I, C, O]):
        pass

    # Resolved case with one generic arg defined

    class OpB(Generic[C, O], OpA[int, C, O]):
        pass

    assert resolve_types_from_generic(OpB[float, str](), OpA, I, C, O) == [
        int,
        float,
        str,
    ]

    # Resolved case with single generic arg defined

    class OpBSingle(Generic[I], OpA[I, I, I]):
        pass

    assert resolve_types_from_generic(OpBSingle[int](), OpA, I, C, O) == [int, int, int]
    assert resolve_types_from_generic(OpBSingle(), OpA, I, C, O) == [None, None, None]  # pyright: ignore[reportUnknownArgumentType]

    # Resolved case with two generic args defined
    class OpC(Generic[C], OpB[C, str]):
        pass

    assert resolve_types_from_generic(OpC[float](), OpA, I, C, O) == [int, float, str]

    # Resolved case with all generic args defined
    class OpD(OpC[float]):
        pass

    assert resolve_types_from_generic(OpD(), OpA, I, C, O) == [int, float, str]

    class OpCSingle(OpBSingle[str]):
        pass

    assert resolve_types_from_generic(OpCSingle(), OpA, I, C, O) == [str, str, str]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
