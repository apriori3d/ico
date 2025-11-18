from collections.abc import Iterator

from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.meta.flow_meta import IcoFlowMeta
from apriori.ico.core.types import IcoNodeType


def test_operator_structure_builds_correct_tree() -> None:
    """
    Verify that composed ICO operators (map + compose)
    correctly build a hierarchical IcoFlow structure.

    The tested dataflow:
        augment.map() | collate
    corresponds to:
        Iterator[float] → Iterator[float] → float

    Steps:
        1. augment: multiply each element by 2
        2. collate: take the max of results
    """

    # ─────────────────────────────
    # 1. Define base operators
    # ─────────────────────────────
    augment = IcoOperator[float, float](lambda x: x * 2, name="augment")
    collate = IcoOperator[Iterator[float], float](max, name="collate")

    # ─────────────────────────────
    # 2. Compose operators into a small pipeline
    # ─────────────────────────────
    pipeline = augment.map() | collate

    # ─────────────────────────────
    # 3. Execute the pipeline
    # ─────────────────────────────
    result = pipeline(iter([1.0, 5.0, 3.0]))
    assert result == 10.0  # (5 * 2) = 10

    # ─────────────────────────────
    # 4. Retrieve and inspect structural description
    # ─────────────────────────────
    flow = IcoFlowMeta.from_operator(pipeline)

    # Root node — composition
    assert flow.node_type == IcoNodeType.chain

    # ─────────────────────────────
    # 5. Validate hierarchy
    # ─────────────────────────────
    assert len(flow.children) == 2
    map_node, collate_node = flow.children

    # Child 1: map
    assert map_node.node_type == IcoNodeType.map

    # Child 2: collate
    assert collate_node.node_type == IcoNodeType.operator
    assert collate_node.name == "collate"

    # ─────────────────────────────
    # 6. Validate nested structure of map node
    # ─────────────────────────────
    assert len(map_node.children) == 1
    inner_augment = map_node.children[0]
    assert inner_augment.node_type == IcoNodeType.operator
    assert inner_augment.name == "augment"

    # ─────────────────────────────
    # 7. Flatten and verify traversal order
    # ─────────────────────────────
    names = _collect_names(flow)
    assert "augment" in names
    assert "collate" in names


def _collect_names(node: IcoFlowMeta) -> list[str]:
    """Recursively traverse an IcoFlow and collect node names."""
    result = [node.name]
    for child in node.children:
        result.extend(_collect_names(child))
    return result


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
