"""Agent-facing behavior: direct converter search and chain type errors."""

import json
from typing import Any, cast

import pytest

from ico.core.operator import IcoOperator
from ico.core.signature_check import (
    IcoSignatureError,
    strict_signatures,
    validate_signatures,
)
from ico.registry import IcoRegistry, register
from ico.spec import NodeSpec, PipelineSpec, SpecError, build


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


# ─── Chain type errors ───


def test_build_reports_step_refs_and_types(registry: IcoRegistry) -> None:
    spec = PipelineSpec(
        root=NodeSpec(
            kind="seq",
            name="tryon",
            children=(op("tryon.load@1"), op("tryon.segment@1"), op("tryon.render@1")),
        )
    )
    with pytest.raises(SpecError) as exc_info:
        build(spec, registry)

    assert str(exc_info.value) == (
        "Type mismatch in seq 'tryon': "
        "step 3 (tryon.segment@1 → tryon.render@1): expected Mask, got Image"
    )


def test_build_accepts_compatible_chain(registry: IcoRegistry) -> None:
    spec = PipelineSpec(
        root=NodeSpec(
            kind="seq",
            children=(op("tryon.load@1"), op("tryon.mask@1"), op("tryon.render@1")),
        )
    )
    flow = build(spec, registry)
    assert isinstance(flow(Photo()), Image)  # type: ignore[operator]


def test_strict_chain_error_names_step_and_refs(registry: IcoRegistry) -> None:
    load, segment, render = (
        node(registry, ref)
        for ref in ("tryon.load@1", "tryon.segment@1", "tryon.render@1")
    )
    with strict_signatures(), pytest.raises(IcoSignatureError) as exc_info:
        _ = load | segment | render

    issue = exc_info.value.issue
    assert issue.step == 3
    assert str(exc_info.value) == (
        "step 3 (tryon.segment@1 → tryon.render@1): expected Mask, got Image"
    )


def test_validate_reports_step_of_every_link(registry: IcoRegistry) -> None:
    def to_photo(x: int) -> Photo:
        return Photo()

    flow = (
        IcoOperator(to_photo)
        | node(registry, "tryon.load@1")
        | node(registry, "tryon.segment@1")
        | node(registry, "tryon.render@1")
    )
    issues = validate_signatures(flow)

    assert [(i.step, i.left, i.right) for i in issues] == [
        (4, "tryon.segment@1", "tryon.render@1")
    ]
    assert issues[0].message.startswith("step 4 (tryon.segment@1 → tryon.render@1)")
