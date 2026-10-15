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

Compatibility is checked conservatively: anything that cannot be decided
(``Any``, TypeVars, Literal, Callable, non-runtime protocols, ...) is treated
as compatible.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, Union, get_args, get_origin

from ico.core.node import IcoNodeProtocol, iterate_nodes
from ico.core.signature import IcoSignature

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

IssueKind = Literal["mismatch", "conflict", "uninferred"]


@dataclass(frozen=True, slots=True)
class SignatureIssue:
    """A signature problem between two nodes of a flow.

    Attributes:
        kind: ``"mismatch"`` (error) when the output type of ``left`` is not
            assignable to the input type of the chained ``right``;
            ``"conflict"`` (error) when two branches ``left`` and ``right``
            of a fork or route accept input types with no common type;
            ``"uninferred"`` (warning) when a signature could not be
            inferred, so the link was not checked.
        left: Name of the upstream node (or the first conflicting branch).
        right: Name of the downstream node (or the second conflicting branch).
        output: Output type of ``left``; for a conflict, its input type
            (``None`` when uninferred).
        input: Input type of ``right`` (``None`` when uninferred).
        node: Name of the fork or route for a conflict, ``None`` otherwise.
    """

    kind: IssueKind
    left: str
    right: str
    output: Any
    input: Any
    node: str | None = None

    @property
    def is_error(self) -> bool:
        return self.kind != "uninferred"

    @property
    def message(self) -> str:
        from ico.core.signature_utils import format_ico_type

        if self.kind == "conflict":
            return (
                f"branches of '{self.node}': inputs {format_ico_type(self.output)} "
                f"of '{self.left}' and {format_ico_type(self.input)} "
                f"of '{self.right}' have no common type"
            )
        link = f"'{self.left}' | '{self.right}'"
        if self.kind == "uninferred":
            return f"{link}: signature could not be inferred, link not checked"
        return (
            f"{link}: output {format_ico_type(self.output)} of '{self.left}' "
            f"is not assignable to input {format_ico_type(self.input)} "
            f"of '{self.right}'"
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


def check_link(left: IcoNodeProtocol, right: IcoNodeProtocol) -> SignatureIssue | None:
    """Check that ``left``'s output can be fed into ``right``'s input.

    Returns:
        A :class:`SignatureIssue` on mismatch or when a signature is not
        inferred, ``None`` when the link is compatible or undecidable.
    """
    left_leaf = _boundary_node(left, side="output")
    right_leaf = _boundary_node(right, side="input")
    left_sig = _inferred_signature(left_leaf)
    right_sig = _inferred_signature(right_leaf)
    left_label, right_label = _label(left_leaf), _label(right_leaf)

    if left_sig is None or right_sig is None:
        return SignatureIssue("uninferred", left_label, right_label, None, None)

    if is_assignable(left_sig.o, right_sig.i) is False:
        return SignatureIssue(
            "mismatch", left_label, right_label, left_sig.o, right_sig.i
        )

    return None


def check_branch_inputs(
    branches: Sequence[IcoNodeProtocol], node_name: str
) -> SignatureIssue | None:
    """Check that branches fed with the same item accept a common input type.

    Used for fork branches and for a route's key function and branches.

    Branches with uninferred signatures are skipped. Two unrelated classes
    are treated as conflicting even though a common subclass might exist.

    Returns:
        A ``"conflict"`` issue for the first incompatible pair, else ``None``.
    """
    typed = [
        (branch, sig)
        for branch in branches
        if (sig := _inferred_signature(branch)) is not None
    ]
    for pos, (a, a_sig) in enumerate(typed):
        for b, b_sig in typed[pos + 1 :]:
            if (
                is_assignable(a_sig.i, b_sig.i) is False
                and is_assignable(b_sig.i, a_sig.i) is False
            ):
                return SignatureIssue(
                    "conflict", _label(a), _label(b), a_sig.i, b_sig.i, node_name
                )
    return None


def validate_signatures(flow: IcoNodeProtocol) -> list[SignatureIssue]:
    """Check chain links, fork/route branches and process stop conditions.

    Collects all issues in the flow tree.

    Unlike strict mode this never raises: errors and warnings are returned
    so that a caller (e.g. an agent) can see all problems at once.
    """
    from ico.core.chain import IcoChain
    from ico.core.fork import IcoFork
    from ico.core.process import IcoProcess
    from ico.core.route import IcoRoute

    issues: list[SignatureIssue] = []
    for node in iterate_nodes(flow):
        issue: SignatureIssue | None = None
        if isinstance(node, IcoChain):
            left, right = node.children
            issue = check_link(left, right)
        elif isinstance(node, IcoFork | IcoRoute):
            issue = check_branch_inputs(node.children, node.name or "")
        elif isinstance(node, IcoProcess) and len(node.children) == 2:
            # Process with a stop condition: the body output feeds `until`.
            body, until = node.children
            issue = check_link(body, until)
        if issue is not None:
            issues.append(issue)
    return issues


def narrowest_type(types_: Sequence[Any]) -> Any | None:
    """Pick the type among ``types_`` that is assignable to all the others.

    Used to compute the input type of a fork: a value of the narrowest type
    is accepted by every branch. ``Any`` never narrows. Undecidable
    comparisons are treated as compatible.

    Returns:
        The narrowest type, or ``None`` if none is compatible with all others.
    """
    known = [t for t in types_ if not _is_any(t)]
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
    from ico.core.chain import IcoChain

    while isinstance(node, IcoChain):
        left, right = node.children
        node = right if side == "output" else left
    return node


def _inferred_signature(node: IcoNodeProtocol) -> IcoSignature | None:
    signature: IcoSignature | None = getattr(node, "signature", None)
    if signature is None or not signature.infered:
        return None
    return signature


def _label(node: IcoNodeProtocol) -> str:
    """Human-readable node name: explicit name, wrapped function name or class."""
    if node.name:
        return node.name
    fn_name: str | None = getattr(getattr(node, "_fn", None), "__name__", None)
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
    if _is_any(src) or _is_any(dst) or dst is object:
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


def _is_any(t: Any) -> bool:
    # Signatures encode Any as `Any`, `type(Any)` or `type[Any]`.
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
