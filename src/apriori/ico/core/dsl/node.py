# ────────────────────────────────────────────────
# Tree traversal Utilities
# ────────────────────────────────────────────────


from collections.abc import Iterator
from typing import Any

from apriori.ico.core.types import IcoNodeProtocol


class IcoTypeInfo:
    ico_input: type | None
    ico_context: type | None
    ico_output: type | None

    def __init__(
        self,
        i: type | None = None,
        c: type | None = None,
        o: type | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.ico_input = i
        self.ico_context = c
        self.ico_output = o


def iterate_nodes(
    node: IcoNodeProtocol,
) -> Iterator[IcoNodeProtocol]:
    """Recursively yield all children operators in the flow tree."""
    yield node
    for c in node.children:
        yield from iterate_nodes(c)


def iterate_parents(
    node: IcoNodeProtocol,
) -> Iterator[IcoNodeProtocol]:
    """Recursively yield all parent operators in the flow tree."""
    if node.parent is None:
        return

    yield node.parent
    yield from iterate_parents(node.parent)
