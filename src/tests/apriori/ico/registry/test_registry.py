import json
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, TypeVar

import pytest

from ico.core.operator import IcoOperator, IcoOperatorProtocol, operator
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.registry import (
    EXACT,
    SUBCLASS,
    IcoRegistry,
    OperatorMeta,
    default_registry,
    format_ref,
    get_meta,
    match_type,
    parse_ref,
    register,
)

T = TypeVar("T")


@pytest.fixture
def registry() -> IcoRegistry:
    return IcoRegistry()


def tokenize(text: str) -> list[str]:
    """Split text into words.

    Whitespace-separated; empty strings are dropped.
    """
    return text.split()


def count(xs: Sequence[str]) -> int:
    """Count items."""
    return len(xs)


def is_long(x: int) -> bool:
    """Check whether a number is large."""
    return x > 100


def to_str(x: int) -> str:
    return str(x)


def identity(x: T) -> T:
    return x


# ─── match_type ───


@pytest.mark.parametrize(
    ("sub", "sup", "expected"),
    [
        (int, int, EXACT),
        (list[int], list[int], EXACT),
        (int | None, int | None, EXACT),
        (bool, int, SUBCLASS),
        (list[int], Sequence[int], SUBCLASS),
        (dict[str, int], dict[str, int], EXACT),
        (list[bool], list[int], None),
        (list[int], Sequence[object], None),
        (Sequence[int], list[int], None),
        (int, bool, None),
        (str, int, None),
        (Any, int, None),
        (int, Any, None),
        (list, list[int], None),
        (int, int | None, None),
    ],
)
def test_match_type(sub: Any, sup: Any, expected: int | None) -> None:
    assert match_type(sub, sup) == expected


# ─── References ───


def test_refs() -> None:
    assert format_ref("tokenize", 2) == "tokenize@2"
    assert parse_ref("tokenize@2") == ("tokenize", 2)


@pytest.mark.parametrize(
    "ref", ["tokenize", "tokenize@", "@1", "tokenize@x", "a b@1", "t@0"]
)
def test_parse_invalid_ref(ref: str) -> None:
    with pytest.raises(ValueError):
        parse_ref(ref)


# ─── Registration ───


def test_register_function_wraps_and_attaches_meta(registry: IcoRegistry) -> None:
    op = register("tokenize", 1, tags={"text"}, caps={"pure"}, registry=registry)(
        tokenize
    )

    assert isinstance(op, IcoOperator)
    assert op("a b") == ["a", "b"]
    assert get_meta(op) == OperatorMeta(
        "tokenize", 1, frozenset({"text"}), frozenset({"pure"}), "draft"
    )
    assert "tokenize@1" in registry
    assert registry.get("tokenize@1").node is op


def test_register_returns_node_unchanged(registry: IcoRegistry) -> None:
    op = IcoOperator(count)
    assert register("count", 1, registry=registry)(op) is op


def test_register_as_decorator_over_operator(registry: IcoRegistry) -> None:
    @register("square", 1, registry=registry)
    @operator()
    def square(x: int) -> int:
        """Square a number."""
        return x * x

    assert square(3) == 9
    assert registry.get("square@1").description == "Square a number."


def test_register_source_sink_and_composite(registry: IcoRegistry) -> None:
    def numbers() -> Iterator[int]:
        """Yield a few numbers."""
        return iter([1, 2])

    def show(x: int) -> None:
        """Print a number."""

    register("numbers", 1, registry=registry)(IcoSource(numbers))
    register("show", 1, registry=registry)(IcoSink(show))
    flow = IcoOperator(to_str) | IcoOperator(tokenize)
    register("to_words", 1, description="Number to words.", registry=registry)(flow)

    assert registry.get("numbers@1").description == "Yield a few numbers."
    assert registry.get("show@1").signature.format() == "Iterator[int] → ()"
    assert registry.get("to_words@1").card()["sig"] == "int → list[str]"
    assert registry.get("to_words@1").description == "Number to words."


def test_default_registry_used_when_not_given() -> None:
    op = register("tests.registry.default", 1)(to_str)
    try:
        assert default_registry.get("tests.registry.default@1").node is op
    finally:
        default_registry.unregister("tests.registry.default@1")


def test_static_types_preserved(registry: IcoRegistry) -> None:
    # Checked by pyright and mypy in `make ci`.
    op: IcoOperatorProtocol[int, str] = register("to_str", 1, registry=registry)(to_str)
    node: IcoOperator[str, list[str]] = register("tok", 1, registry=registry)(
        IcoOperator(tokenize)
    )
    assert op(1) == "1"
    assert node("a") == ["a"]


@pytest.mark.parametrize(
    ("name", "version", "status", "message"),
    [
        ("", 1, "draft", "Invalid operator name"),
        ("a@b", 1, "draft", "Invalid operator name"),
        ("a b", 1, "draft", "Invalid operator name"),
        ("ok", 0, "draft", "positive integer"),
        ("ok", True, "draft", "positive integer"),
        ("ok", 1, "final", "status"),
    ],
)
def test_register_invalid_meta(
    registry: IcoRegistry, name: str, version: Any, status: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        register(name, version, status=status, registry=registry)(to_str)


def test_register_requires_inferred_signature(registry: IcoRegistry) -> None:
    with pytest.raises(ValueError, match="signature cannot be inferred"):
        register("identity", 1, registry=registry)(identity)
    assert len(registry) == 0


def test_duplicate_ref_rejected_unless_replace(registry: IcoRegistry) -> None:
    register("op", 1, registry=registry)(to_str)
    with pytest.raises(ValueError, match="already registered"):
        register("op", 1, registry=registry)(is_long)

    replacement = register("op", 1, replace=True, registry=registry)(is_long)
    assert registry.get("op@1").node is replacement


def test_node_cannot_have_two_refs(registry: IcoRegistry) -> None:
    op = register("op", 1, registry=registry)(to_str)
    with pytest.raises(ValueError, match="already registered as 'op@1'"):
        register("other", 1, registry=registry)(op)


def test_versions_and_unregister(registry: IcoRegistry) -> None:
    register("op", 2, registry=registry)(to_str)
    v1 = register("op", 1, registry=registry)(to_str)
    register("other", 1, registry=registry)(count)

    assert registry.versions("op") == [1, 2]
    assert [e.ref for e in registry] == ["op@1", "op@2", "other@1"]
    assert len(registry) == 3

    registry.unregister("op@1")
    assert "op@1" not in registry
    assert get_meta(v1) is None
    with pytest.raises(KeyError, match="not registered"):
        registry.get("op@1")


# ─── Cards ───


def test_card(registry: IcoRegistry) -> None:
    register(
        "tokenize",
        3,
        tags={"text", "nlp"},
        caps={"pure"},
        status="verified",
        registry=registry,
    )(tokenize)

    assert registry.get("tokenize@3").card() == {
        "name": "tokenize",
        "version": 3,
        "sig": "str → list[str]",
        "description": "Split text into words.",
        "tags": ["nlp", "text"],
        "caps": ["pure"],
        "status": "verified",
    }


def test_cards_are_json(registry: IcoRegistry) -> None:
    register("tokenize", 1, registry=registry)(tokenize)
    register("count", 1, registry=registry)(count)
    cards = json.loads(json.dumps(registry.cards()))
    assert [c["name"] for c in cards] == ["count", "tokenize"]


def test_missing_docstring_gives_empty_description(registry: IcoRegistry) -> None:
    register("to_str", 1, registry=registry)(to_str)
    assert registry.get("to_str@1").description == ""


# ─── Search ───


@pytest.fixture
def catalog(registry: IcoRegistry) -> IcoRegistry:
    register("tokenize", 1, tags={"text"}, registry=registry)(tokenize)
    register("count", 1, tags={"text", "stats"}, registry=registry)(count)
    register("is_long", 1, tags={"stats"}, registry=registry)(is_long)
    register("to_str", 1, registry=registry)(to_str)

    def total(xs: Iterable[int]) -> int:
        """Sum numbers."""
        return sum(xs)

    register("total", 1, tags={"stats"}, registry=registry)(total)
    return registry


def refs(entries: Iterable[Any]) -> list[str]:
    return [e.ref for e in entries]


def test_search_by_input(catalog: IcoRegistry) -> None:
    assert refs(catalog.search(input=int)) == ["is_long@1", "to_str@1"]
    # bool is a subclass of int: subclass matches come after exact ones.
    assert refs(catalog.search(input=bool)) == ["is_long@1", "to_str@1"]
    assert refs(catalog.search(input=list[str])) == ["count@1"]
    assert refs(catalog.search(input=list[int])) == ["total@1"]


def test_search_by_output(catalog: IcoRegistry) -> None:
    assert refs(catalog.search(output=str)) == ["to_str@1"]
    # Exact `int` outputs first, then `bool` (subclass of int).
    assert refs(catalog.search(output=int)) == ["count@1", "total@1", "is_long@1"]
    assert refs(catalog.search(output=Sequence[str])) == ["tokenize@1"]


def test_search_generic_args_are_invariant(catalog: IcoRegistry) -> None:
    assert refs(catalog.search(input=list[bool])) == []
    assert refs(catalog.search(output=Sequence[object])) == []


def test_search_by_input_and_output(catalog: IcoRegistry) -> None:
    assert refs(catalog.search(input=int, output=str)) == ["to_str@1"]
    assert refs(catalog.search(input=str, output=int)) == []


def test_search_exact_before_subclass(catalog: IcoRegistry) -> None:
    def bool_only(x: bool) -> int:
        return int(x)

    register("bool_only", 1, registry=catalog)(bool_only)
    assert refs(catalog.search(input=bool)) == ["bool_only@1", "is_long@1", "to_str@1"]


def test_search_by_tags_and_text(catalog: IcoRegistry) -> None:
    assert refs(catalog.search(tags={"stats"})) == ["count@1", "is_long@1", "total@1"]
    assert refs(catalog.search(tags={"stats", "text"})) == ["count@1"]
    assert refs(catalog.search(text="WORDS")) == ["tokenize@1"]
    assert refs(catalog.search(text="to_")) == ["to_str@1"]
    assert refs(catalog.search(tags={"stats"}, input=int)) == ["is_long@1"]


def test_search_without_filters_returns_all(catalog: IcoRegistry) -> None:
    assert len(catalog.search()) == len(catalog)


def test_any_typed_operator_not_matched_by_type(registry: IcoRegistry) -> None:
    register("anything", 1, registry=registry)(IcoOperator[Any, Any](lambda x: x))
    assert refs(registry.search(input=int)) == []
    assert refs(registry.search()) == ["anything@1"]
