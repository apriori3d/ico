"""Agent-facing behavior: direct converter search and chain type errors."""

import json
from typing import Any, cast

import pytest

from ico.core.fork import fork
from ico.core.operator import IcoOperator
from ico.core.signature_check import (
    IcoSignatureError,
    strict_signatures,
    validate_signatures,
)
from ico.core.tree_utils import TreePathIndex
from ico.registry import IcoRegistry, register
from ico.spec import NodeSpec, PipelineSpec, SpecError, build, to_spec


class Photo:
    pass


class Image:
    pass


class Mask:
    pass


@pytest.fixture
def registry() -> IcoRegistry:
    registry = IcoRegistry()

    def load(photo: Photo) -> Image:
        """Decode a photo into an image."""
        return Image()

    def segment(image: Image) -> Image:
        """Highlight the garment (returns an Image, not a Mask)."""
        return image

    def mask(image: Image) -> Mask:
        """Extract the garment mask."""
        return Mask()

    def render(mask: Mask) -> Image:
        """Render the try-on result."""
        return Image()

    register("tryon.load", 1, registry=registry)(load)
    register("tryon.segment", 1, registry=registry)(segment)
    register("tryon.mask", 1, registry=registry)(mask)
    register("tryon.render", 1, registry=registry)(render)
    return registry


def op(ref: str) -> NodeSpec:
    return NodeSpec(kind="op", op=ref)


def node(registry: IcoRegistry, ref: str) -> IcoOperator[Any, Any]:
    return cast(IcoOperator[Any, Any], registry.get(ref).node)


# ─── Search for direct converters ───


def test_search_finds_direct_converters(registry: IcoRegistry) -> None:
    # Agents see cards: the compact JSON view of each found operator.
    cards = [entry.card() for entry in registry.search(input=Image, output=Mask)]

    assert cards == [
        {
            "name": "tryon.mask",
            "version": 1,
            "sig": "Image → Mask",
            "description": "Extract the garment mask.",
            "tags": [],
            "caps": [],
            "status": "draft",
        }
    ]
    json.dumps(cards)  # compact and JSON-ready


def test_search_by_input_only(registry: IcoRegistry) -> None:
    refs = [entry.ref for entry in registry.search(input=Image)]
    assert refs == ["tryon.mask@1", "tryon.segment@1"]


# ─── Chain type errors: located in spec-tree coordinates ───


def test_build_error_fields(registry: IcoRegistry) -> None:
    spec = PipelineSpec(
        root=NodeSpec(
            kind="seq",
            name="tryon",
            children=(op("tryon.load@1"), op("tryon.segment@1"), op("tryon.render@1")),
        )
    )
    with pytest.raises(SpecError) as exc_info:
        build(spec, registry)

    error = exc_info.value
    # Structured fields (what an MCP server returns as JSON), not the message.
    assert error.path == TreePathIndex.create(2)
    assert error.expected == "Mask"
    assert error.got == "Image"
    assert error.refs == ("tryon.segment@1", "tryon.render@1")
    assert error.path is not None
    assert spec.root.at(error.path).op == "tryon.render@1"
    assert str(error) == (
        "at 2 (tryon.segment@1 → tryon.render@1): expected Mask, got Image"
    )


def test_build_error_path_in_nested_spec(registry: IcoRegistry) -> None:
    good = NodeSpec(kind="seq", children=(op("tryon.load@1"), op("tryon.mask@1")))
    bad = NodeSpec(
        kind="seq",
        children=(op("tryon.load@1"), op("tryon.segment@1"), op("tryon.render@1")),
    )
    spec = PipelineSpec(root=NodeSpec(kind="fork", children=(good, bad)))

    with pytest.raises(SpecError) as exc_info:
        build(spec, registry)

    error = exc_info.value
    assert error.path == TreePathIndex.create(1, 2)
    assert error.path is not None
    assert spec.root.at(error.path).op == "tryon.render@1"
    assert str(error).startswith("at 1.2 (tryon.segment@1 → tryon.render@1)")


def test_every_build_error_is_located(registry: IcoRegistry) -> None:
    spec = PipelineSpec(
        root=NodeSpec(kind="seq", children=(op("tryon.load@1"), op("tryon.missing@1")))
    )
    with pytest.raises(SpecError) as exc_info:
        build(spec, registry)

    error = exc_info.value
    assert error.path == TreePathIndex.create(1)
    assert error.refs is None and error.expected is None and error.got is None
    assert str(error).startswith("at 1: Cannot resolve operator 'tryon.missing@1'")


def test_root_error_is_located_at_root(registry: IcoRegistry) -> None:
    with pytest.raises(SpecError) as exc_info:
        build(PipelineSpec(root=NodeSpec(kind="monitor")), registry)
    assert exc_info.value.path == TreePathIndex()
    assert str(exc_info.value).startswith("at root: ")


def test_build_accepts_compatible_chain(registry: IcoRegistry) -> None:
    spec = PipelineSpec(
        root=NodeSpec(
            kind="seq",
            children=(op("tryon.load@1"), op("tryon.mask@1"), op("tryon.render@1")),
        )
    )
    flow = build(spec, registry)
    assert isinstance(flow(Photo()), Image)  # type: ignore[operator]


def test_strict_chain_path_is_chain_relative(registry: IcoRegistry) -> None:
    load, segment, render = (
        node(registry, ref)
        for ref in ("tryon.load@1", "tryon.segment@1", "tryon.render@1")
    )
    with strict_signatures(), pytest.raises(IcoSignatureError) as exc_info:
        _ = load | segment | render

    issue = exc_info.value.issue
    assert issue.path == TreePathIndex.create(2)
    assert issue.chain_relative
    assert str(exc_info.value) == (
        "at chain step 2 (tryon.segment@1 → tryon.render@1): expected Mask, got Image"
    )


def test_validate_paths_match_exported_spec(registry: IcoRegistry) -> None:
    good = node(registry, "tryon.load@1") | node(registry, "tryon.mask@1")
    bad = (
        node(registry, "tryon.load@1")
        | node(registry, "tryon.segment@1")
        | node(registry, "tryon.render@1")
    )
    flow = fork(good, bad)

    issues = validate_signatures(flow)

    assert [(str(i.path), i.left, i.right) for i in issues] == [
        ("1.2", "tryon.segment@1", "tryon.render@1")
    ]
    assert to_spec(flow).root.at(issues[0].path).op == "tryon.render@1"  # type: ignore[arg-type]
    assert issues[0].message == (
        "at 1.2 (tryon.segment@1 → tryon.render@1): expected Mask, got Image"
    )


def test_validate_fork_conflict_points_at_branch(registry: IcoRegistry) -> None:
    flow = fork(node(registry, "tryon.load@1"), node(registry, "tryon.render@1"))

    (issue,) = validate_signatures(flow)

    assert issue.kind == "conflict"
    assert issue.path == TreePathIndex.create(1)
    assert str(issue).startswith("at 1: branches of 'fork'")


def test_node_spec_at_reports_missing_path() -> None:
    root = NodeSpec(kind="seq", children=(op("a@1"), op("b@1")))
    assert root.at(TreePathIndex()) is root
    with pytest.raises(SpecError) as exc_info:
        root.at(TreePathIndex.create(1, 0))
    assert exc_info.value.path == TreePathIndex.create(1)
