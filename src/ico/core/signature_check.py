# In this module we compare runtime type annotations, which are inherently
# typed as Any by the typing introspection API.
# mypy: disable-error-code=misc
"""Runtime signature compatibility checks for chained operators.

Static type checkers verify ``a | b`` only when operator types are known at
type-check time. Flows assembled dynamically (e.g. by an agent from a
registry) bypass that, so this module checks at runtime that the output type
of each chain link is assignable to the input type of the next one.

The check is opt-in:

    >>> with strict_signatures():
    ...     flow = parse | score  # raises IcoSignatureError on mismatch

or per node with ``chain(a, b, strict=True)`` / ``fork(..., strict=True)``
(fork branches, and a route's key and branches, must accept a common
input type). :func:`validate_signatures`
reports every issue in an already built flow without raising.

Untyped identity nodes (``IcoIdentity`` and subclasses such as
``IcoProgress``, built as ``IcoProgress[Any]`` from a spec) pass items
through unchanged, so links are checked across them, between their
neighbours.

Compatibility is checked conservatively: anything that cannot be decided
(``Any``, TypeVars, Literal, Callable, non-runtime protocols, ...) is treated
as compatible.
"""

from __future__ import annotations

import types
from collections import abc
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

from ico.core.node import IcoNodeProtocol
from ico.core.signature import IcoSignature
from ico.core.tree_utils import TreePathIndex, format_location

if TYPE_CHECKING:
    from ico.core.chain import IcoChain

# ────────────────────────────────────────────────
# Strict mode switch
# ────────────────────────────────────────────────

_STRICT: ContextVar[bool] = ContextVar("ico_strict_signatures", default=False)


@contextmanager
def strict_signatures(enabled: bool = True) -> Iterator[None]:
    """Enable (or disable) chain signature checks within a block.

    Example:
        >>> with strict_signatures():
        ...     flow = a | b | c
    """
    token = _STRICT.set(enabled)
    try:
        yield
    finally:
        _STRICT.reset(token)


def is_strict() -> bool:
    """Whether chain signature checks are enabled in the current context."""
    return _STRICT.get()


# ────────────────────────────────────────────────
# Issues and errors
# ────────────────────────────────────────────────

IssueKind = Literal["mismatch", "conflict", "uninferred", "counts_iterator"]


@dataclass(frozen=True, slots=True)
class SignatureIssue:
    """A signature problem between two nodes of a flow.

    Attributes:
        kind: ``"mismatch"`` (error) when the output type of ``left`` is not
            assignable to the input type of the chained ``right``;
            ``"conflict"`` (error) when two branches ``left`` and ``right``
            of a fork or route accept input types with no common type;
            ``"uninferred"`` (warning) when a signature could not be
            inferred, so the link was not checked;
            ``"counts_iterator"`` (warning) when a progress node ``right``
            receives a whole iterator from ``left`` (``output``): it counts
            the iterator as one item, not its items.
        left: Name of the upstream node (or the first conflicting branch).
        right: Name of the downstream node (or the second conflicting branch).
        output: Output type of ``left``; for a conflict, its input type
            (``None`` when uninferred).
        input: Input type of ``right`` (``None`` when uninferred).
        node: Name of the fork or route for a conflict, ``None`` otherwise.
        path: Location of the node that rejected its input (``right``, or
            the second conflicting branch) in spec-tree coordinates, i.e.
            the node at ``to_spec(flow).root.at(path)``. ``None`` if unknown.
        chain_relative: ``path`` is relative to a chain being composed (strict
            mode), not to a whole flow: ``at chain step 2``.

    Node names are registry references (``"name@version"``) for registered
    operators, so an agent can relate an error to the spec it wrote.
    """

    kind: IssueKind
    left: str
    right: str
    output: Any
    input: Any
    node: str | None = None
    path: TreePathIndex | None = None
    chain_relative: bool = False

    @property
    def is_error(self) -> bool:
        return self.kind not in ("uninferred", "counts_iterator")

    @property
    def message(self) -> str:
        from ico.core.signature_utils import format_ico_type

        if self.kind == "conflict":
            reason = (
                f"branches of '{self.node}': inputs {format_ico_type(self.output)} "
                f"of '{self.left}' and {format_ico_type(self.input)} "
                f"of '{self.right}' have no common type"
            )
            return format_location(
                reason, self.path, chain_relative=self.chain_relative
            )
        # e.g. "at 0.2 (tryon.segment@1 → tryon.render@1): expected Mask, got Image"
        if self.kind == "uninferred":
            reason = "signature could not be inferred, link not checked"
        elif self.kind == "counts_iterator":
            reason = (
                f"progress counts the whole {format_ico_type(self.output)} as one "
                "item; wrap it in a stream to count the items"
            )
        else:
            reason = (
                f"expected {format_ico_type(self.input)}, "
                f"got {format_ico_type(self.output)}"
            )
        return format_location(
            reason,
            self.path,
            (self.left, self.right),
            chain_relative=self.chain_relative,
        )

    def __str__(self) -> str:
        return self.message


class IcoSignatureError(TypeError):
    """Raised in strict mode when composed operators have incompatible types."""

    def __init__(self, issue: SignatureIssue) -> None:
        super().__init__(issue.message)
        self.issue = issue


# ────────────────────────────────────────────────
# Link checks
# ────────────────────────────────────────────────


def check_link(
    left: IcoNodeProtocol,
    right: IcoNodeProtocol,
    *,
    path: TreePathIndex | None = None,
    chain_relative: bool = False,
) -> SignatureIssue | None:
    """Check that ``left``'s output can be fed into ``right``'s input.

    Args:
        left: Upstream node (a chain is checked at its last step).
        right: Downstream node (a chain is checked at its first step).
        path: Location of ``right``, reported in the issue.
        chain_relative: ``path`` is relative to a chain being composed.

    Returns:
        A :class:`SignatureIssue` on mismatch or when a signature is not
        inferred, ``None`` when the link is compatible or undecidable, or
        when one side is an untyped identity node (see :func:`chain_links`).
    """
    left_leaf = _boundary_node(left, side="output")
    right_leaf = _boundary_node(right, side="input")
    if is_transparent(left_leaf) or is_transparent(right_leaf):
        return None  # checked across the identity node, between neighbours
    left_sig = _inferred_signature(left_leaf)
    right_sig = _inferred_signature(right_leaf)
    left_label, right_label = _label(left_leaf), _label(right_leaf)

    if left_sig is None or right_sig is None:
        return SignatureIssue(
            "uninferred",
            left_label,
            right_label,
            None,
            None,
            path=path,
            chain_relative=chain_relative,
        )

    if is_assignable(left_sig.o, right_sig.i) is False:
        return SignatureIssue(
            "mismatch",
            left_label,
            right_label,
            left_sig.o,
            right_sig.i,
            path=path,
            chain_relative=chain_relative,
        )

    return None


def chain_links(steps: Sequence[IcoNodeProtocol]) -> list[tuple[int, int]]:
    """Links to check in a sequence of chain steps, as index pairs.

    Untyped identity nodes are skipped: a link joins the nearest steps
    around them, so ``a | progress | b`` is checked as ``a → b``.

    Example:
        >>> chain_links([a, progress, b])  # [(0, 2)]
    """
    links: list[tuple[int, int]] = []
    previous: int | None = None
    for index, step in enumerate(steps):
        if is_transparent(_boundary_node(step, side="input")):
            continue
        if previous is not None:
            links.append((previous, index))
        previous = index
    return links


def is_transparent(node: IcoNodeProtocol) -> bool:
    """Whether a node is an untyped identity, skipped by link checks."""
    from ico.core.identity import IcoIdentity
    from ico.registry.registry import get_meta

    # Check the metadata before isinstance narrows the node to a generic identity.
    if get_meta(node) is not None or _inferred_signature(node) is not None:
        return False
    return isinstance(node, IcoIdentity)


def chain_length(node: IcoNodeProtocol) -> int:
    """Number of steps of a node in a flattened chain (1 for non-chains).

    Registered chains are opaque operators and count as one step.
    """
    from ico.core.chain import IcoChain
    from ico.registry.registry import get_meta

    # Check the metadata before isinstance narrows the node to a generic chain.
    if get_meta(node) is not None or not isinstance(node, IcoChain):
        return 1
    left, right = node.children
    return chain_length(left) + chain_length(right)


def check_branch_inputs(
    branches: Sequence[IcoNodeProtocol],
    node_name: str,
    *,
    path: TreePathIndex | None = None,
) -> SignatureIssue | None:
    """Check that branches fed with the same item accept a common input type.

    Used for fork branches and for a route's key function and branches.

    Branches with uninferred signatures are skipped. Two unrelated classes
    are treated as conflicting even though a common subclass might exist.

    Args:
        branches: The node's children that receive the same item.
        node_name: Name of the fork or route, for messages.
        path: Location of the fork or route; the issue points at the second
            conflicting branch below it.

    Returns:
        A ``"conflict"`` issue for the first incompatible pair, else ``None``.
    """
    typed = [
        (index, branch, sig)
        for index, branch in enumerate(branches)
        if (sig := _inferred_signature(branch)) is not None
    ]
    for pos, (_, a, a_sig) in enumerate(typed):
        for b_index, b, b_sig in typed[pos + 1 :]:
            if (
                is_assignable(a_sig.i, b_sig.i) is False
                and is_assignable(b_sig.i, a_sig.i) is False
            ):
                return SignatureIssue(
                    "conflict",
                    _label(a),
                    _label(b),
                    a_sig.i,
                    b_sig.i,
                    node_name,
                    path=path.add_child(b_index) if path is not None else None,
                )
    return None


def validate_signatures(flow: IcoNodeProtocol) -> list[SignatureIssue]:
    """Check chain links, fork/route branches and process stop conditions.

    Collects all issues in the flow tree.

    Unlike strict mode this never raises: errors and warnings are returned
    so that a caller (e.g. an agent) can see all problems at once.

    Issue paths are in spec-tree coordinates: chains are flattened into
    sequences and registered operators are opaque leaves (not checked
    inside), so ``to_spec(flow).root.at(issue.path)`` is the failing node.
    """
    issues: list[SignatureIssue] = []
    _validate_node(flow, TreePathIndex(), issues)
    return issues


def _validate_node(
    node: IcoNodeProtocol, path: TreePathIndex, issues: list[SignatureIssue]
) -> None:
    from ico.core.chain import IcoChain
    from ico.core.fold import IcoFold, fold_input
    from ico.core.fork import IcoFork
    from ico.core.process import IcoProcess
    from ico.core.route import IcoRoute
    from ico.registry.registry import get_meta

    if get_meta(node) is not None:
        return  # registered operators are opaque, like in a spec

    children = _spec_children(node)
    found: list[SignatureIssue | None] = []
    if isinstance(node, IcoChain):
        for left, right in chain_links(children):
            found.append(
                check_link(children[left], children[right], path=path.add_child(right))
            )
        found.extend(_progress_on_iterators(children, path))
    elif isinstance(node, IcoFork | IcoRoute):
        found.append(check_branch_inputs(node.children, node.name or "", path=path))
    elif isinstance(node, IcoFold):
        # The step takes an item of the source with the state it emits.
        fold = cast(IcoFold[Any, Any], node)
        expected = fold_input(fold)
        step = fold.step
        step_sig = _inferred_signature(_boundary_node(step, side="input"))
        if (
            expected is not None
            and step_sig is not None
            and is_assignable(expected, step_sig.i) is False
        ):
            found.append(
                SignatureIssue(
                    "mismatch",
                    _label(_boundary_node(fold.source, side="output")),
                    _label(_boundary_node(step, side="input")),
                    expected,
                    step_sig.i,
                    path=path.add_child(1),
                )
            )
    elif isinstance(node, IcoProcess) and len(node.children) == 2:
        # Process with a stop condition: the body output feeds `until`.
        body, until = node.children
        found.append(check_link(body, until, path=path.add_child(1)))
    issues.extend(issue for issue in found if issue is not None)

    for index, child in enumerate(children):
        _validate_node(child, path.add_child(index), issues)


def _progress_on_iterators(
    steps: Sequence[IcoNodeProtocol], path: TreePathIndex
) -> list[SignatureIssue | None]:
    """Progress steps of a chain that receive a whole iterator.

    A progress node counts the items that pass it; after a step that
    returns an iterator it sees one item. Its input type is its own when
    typed, else the output of the nearest step before it.
    """
    from ico.core.runtime.progress import IcoProgress

    found: list[SignatureIssue | None] = []
    for index, step in enumerate(steps):
        if not isinstance(cast(object, step), IcoProgress):
            continue
        source = _feeding_step(steps, index)
        own = _inferred_signature(step)
        if own is not None:
            received: Any = own.i
        elif source is not None and (sig := _inferred_signature(source)) is not None:
            received = sig.o
        else:
            continue
        if get_origin(received) in _ITERATORS:
            found.append(
                SignatureIssue(
                    "counts_iterator",
                    _label(source) if source is not None else "",
                    _label(step),
                    received,
                    None,
                    path=path.add_child(index),
                )
            )
    return found


_ITERATORS: tuple[object, ...] = (
    abc.Iterator,
    abc.Iterable,
    abc.Generator,
    abc.AsyncIterator,
    abc.AsyncIterable,
)


def _feeding_step(
    steps: Sequence[IcoNodeProtocol], index: int
) -> IcoNodeProtocol | None:
    """The output boundary of the nearest step before ``index`` that is not
    an untyped identity."""
    for previous in reversed(steps[:index]):
        leaf = _boundary_node(previous, side="output")
        if not is_transparent(leaf):
            return leaf
    return None


def _spec_children(node: IcoNodeProtocol) -> list[IcoNodeProtocol]:
    """Children as exported by ``to_spec``: chains flattened, one worker for
    a factory-based async stream."""
    from ico.core.async_stream import IcoAsyncStream
    from ico.core.chain import IcoChain

    if isinstance(node, IcoChain):
        left, right = node.children
        return [*_chain_steps(left), *_chain_steps(right)]
    if isinstance(node, IcoAsyncStream) and node.pool_from_factory:
        return list(node.children[:1])
    return list(node.children)


def _chain_steps(node: IcoNodeProtocol) -> list[IcoNodeProtocol]:
    from ico.core.chain import IcoChain
    from ico.registry.registry import get_meta

    if get_meta(node) is not None or not isinstance(node, IcoChain):
        return [node]
    left, right = node.children
    return [*_chain_steps(left), *_chain_steps(right)]


def narrowest_type(types_: Sequence[Any]) -> Any | None:
    """Pick the type among ``types_`` that is assignable to all the others.

    Used to compute the input type of a fork: a value of the narrowest type
    is accepted by every branch. ``Any`` never narrows. Undecidable
    comparisons are treated as compatible.

    Returns:
        The narrowest type, or ``None`` if none is compatible with all others.
    """
    known = [t for t in types_ if not is_any_type(t)]
    if not known:
        return types_[0] if types_ else None

    def fits_all(candidate: Any) -> bool:
        return all(is_assignable(candidate, other) is not False for other in known)

    # Prefer a candidate proven compatible with every other type.
    for candidate in known:
        if all(is_assignable(candidate, other) is True for other in known):
            return candidate
    for candidate in known:
        if fits_all(candidate):
            return candidate
    return None


def _boundary_node(
    node: IcoNodeProtocol, side: Literal["input", "output"]
) -> IcoNodeProtocol:
    """Leaf node at the given boundary of a (possibly nested) chain.

    The output of ``a | b`` is the output of ``b`` even if ``a`` is untyped,
    so links are checked between the innermost adjacent leaves.
    """
    # A registered chain is an opaque operator: it is its own boundary.
    chain = _open_chain(node)
    if chain is None:
        return node
    left, right = chain.children
    near, far = (right, left) if side == "output" else (left, right)
    leaf = _boundary_node(near, side)
    # Look past an untyped identity: `a | progress` outputs what `a` does.
    return _boundary_node(far, side) if is_transparent(leaf) else leaf


def _open_chain(node: IcoNodeProtocol) -> IcoChain[Any, Any, Any] | None:
    """The node as a chain to look into, or ``None`` (not a chain, or registered)."""
    from ico.core.chain import IcoChain
    from ico.registry.registry import get_meta

    if get_meta(node) is not None or not isinstance(node, IcoChain):
        return None
    return cast(IcoChain[Any, Any, Any], node)


def _inferred_signature(node: IcoNodeProtocol) -> IcoSignature | None:
    signature: IcoSignature | None = getattr(node, "signature", None)
    if signature is None or not signature.infered:
        return None
    return signature


def _label(node: IcoNodeProtocol) -> str:
    """Node name for messages: registry reference, explicit name, function name
    or class."""
    from ico.registry.registry import get_meta

    meta = get_meta(node)
    if meta is not None:
        return meta.ref
    if node.name:
        return node.name
    fn: object = getattr(node, "_fn", None)
    if cast(object, getattr(fn, "__self__", None)) is node:
        # Structural node implemented by its own method (`_stream_fn`, ...).
        return type(node).__name__
    fn_name: str | None = getattr(fn, "__name__", None)
    return fn_name or str(node)


# ────────────────────────────────────────────────
# Type compatibility
# ────────────────────────────────────────────────


def is_assignable(src: Any, dst: Any) -> bool | None:
    """Check whether a value of type ``src`` can be used where ``dst`` is expected.

    Returns:
        ``True`` if compatible, ``False`` if definitely incompatible and
        ``None`` if it cannot be decided at runtime.

    Note:
        Generic arguments are compared covariantly for all containers.
        This is a deliberate simplification: it catches real mistakes
        (``list[str]`` into ``list[int]``) without false positives for
        read-only usage.

    Example:
        >>> is_assignable(bool, int)  # True
        >>> is_assignable(list[int], Iterable[int])  # True
        >>> is_assignable(str, int)  # False
        >>> is_assignable(Any, int)  # True
    """
    if is_any_type(src) or is_any_type(dst) or dst is object:
        return True

    if isinstance(src, TypeVar) or isinstance(dst, TypeVar):
        return None

    if src == dst:
        return True

    if (src_members := _union_members(src)) is not None:
        return _all(is_assignable(m, dst) for m in src_members)

    if (dst_members := _union_members(dst)) is not None:
        return _any(is_assignable(src, m) for m in dst_members)

    src_origin = get_origin(src) or src
    dst_origin = get_origin(dst) or dst

    # Literal, Annotated, Callable, ... are not checked.
    if not isinstance(src_origin, type) or not isinstance(dst_origin, type):
        return None
    if _is_callable_origin(src_origin) or _is_callable_origin(dst_origin):
        return None

    try:
        if not issubclass(src_origin, dst_origin):
            return False
    except TypeError:
        # Non-runtime protocols and other exotic classes.
        return None

    return _args_assignable(src_origin, get_args(src), dst_origin, get_args(dst))


def _args_assignable(
    src_origin: type,
    src_args: Sequence[Any],
    dst_origin: type,
    dst_args: Sequence[Any],
) -> bool | None:
    if not dst_args:
        return True  # Bare generic target such as `list` or `Iterator`.
    if not src_args:
        return None  # Bare generic source: arguments unknown.

    if src_origin is tuple and dst_origin is tuple:
        return _tuple_args_assignable(src_args, dst_args)

    if len(src_args) != len(dst_args):
        return None  # e.g. dict[K, V] -> Iterable[K]: positions differ.

    return _all(is_assignable(s, d) for s, d in zip(src_args, dst_args, strict=True))


def _tuple_args_assignable(
    src_args: Sequence[Any], dst_args: Sequence[Any]
) -> bool | None:
    src_variadic = len(src_args) == 2 and src_args[1] is Ellipsis
    dst_variadic = len(dst_args) == 2 and dst_args[1] is Ellipsis

    if dst_variadic:
        items = src_args[:1] if src_variadic else src_args
        return _all(is_assignable(s, dst_args[0]) for s in items)
    if src_variadic:
        return None  # Length of the source tuple is unknown.
    if len(src_args) != len(dst_args):
        return False
    return _all(is_assignable(s, d) for s, d in zip(src_args, dst_args, strict=True))


def is_any_type(t: Any) -> bool:
    """Whether a signature type means ``Any``.

    Signatures encode Any as ``Any``, ``type(Any)`` or ``type[Any]``.
    """
    return t is Any or t is type(Any) or t == type[Any]


def _union_members(t: Any) -> tuple[Any, ...] | None:
    origin: object = get_origin(t)
    if origin is Union or origin is types.UnionType:
        return get_args(t)
    return None


def _is_callable_origin(t: object) -> bool:
    return t is Callable


def _all(results: Iterator[bool | None]) -> bool | None:
    """Three-valued AND: any False wins, then any unknown."""
    unknown = False
    for r in results:
        if r is False:
            return False
        if r is None:
            unknown = True
    return None if unknown else True


def _any(results: Iterator[bool | None]) -> bool | None:
    """Three-valued OR: any True wins, then any unknown."""
    unknown = False
    for r in results:
        if r is True:
            return True
        if r is None:
            unknown = True
    return None if unknown else False
