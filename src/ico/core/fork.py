from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar, cast, overload

from ico.core.operator import I, IcoOperator, IcoOperatorProtocol, O, wrap_operator
from ico.core.signature import IcoSignature

O1 = TypeVar("O1")
O2 = TypeVar("O2")
O3 = TypeVar("O3")
O4 = TypeVar("O4")
O5 = TypeVar("O5")
O6 = TypeVar("O6")


class IcoFork(Generic[I, O], IcoOperator[I, O]):
    """Parallel composition: feed one input to several branches, collect a tuple.

    Generic Parameters:
        I: Input type accepted by every branch.
        O: Output type - a tuple of the branch outputs, in branch order.

    ICO signature:
        I → tuple[O1, ..., On]

    Branches run sequentially in declaration order and all receive the same
    input object (it is not copied), so branches must not mutate it.

    Example:
        >>> def mean(xs: list[float]) -> float:
        ...     return sum(xs) / len(xs)
        >>> stats = fork(mean, len)
        >>> stats([1.0, 2.0])  # (1.5, 2)

    Note:
        Prefer the :func:`fork` factory: its overloads give precise static
        types for 2–6 branches.
    """

    __slots__ = "branches"

    branches: Sequence[IcoOperatorProtocol[I, Any]]

    def __init__(
        self,
        *branches: Callable[[I], Any],
        name: str | None = None,
        strict: bool | None = None,
    ):
        """Initialize a fork from two or more branches.

        Args:
            *branches: Callables or operators, each accepting the same input.
                Plain callables are wrapped as IcoOperators.
            name: Optional name for this fork (defaults to "fork").
            strict: Check at runtime that the branch input types have a common
                type. ``None`` follows the current ``strict_signatures()``
                context (off by default).

        Raises:
            ValueError: If fewer than two branches are given.
            IcoSignatureError: In strict mode, if two branches have
                incompatible input types.
        """
        if len(branches) < 2:
            raise ValueError("Fork requires at least two branches.")

        branch_ops = [wrap_operator(branch) for branch in branches]

        from ico.core.signature_check import (
            IcoSignatureError,
            check_branch_inputs,
            is_strict,
        )

        fork_name = name or "fork"
        if strict if strict is not None else is_strict():
            issue = check_branch_inputs(branch_ops, fork_name)
            if issue is not None:
                raise IcoSignatureError(issue)

        super().__init__(fn=self._run_fork, name=fork_name, children=branch_ops)
        self.branches = branch_ops

    def _run_fork(self, item: I) -> O:
        return cast(O, tuple(branch(item) for branch in self.branches))

    def __len__(self) -> int:
        """Return the number of branches."""
        return len(self.branches)

    @property
    def signature(self) -> IcoSignature:
        """Infer the ICO signature: narrowest branch input → tuple of outputs.

        The signature is uninferred if any branch signature is uninferred
        or if the branch inputs have no common type.
        """
        from ico.core.signature_check import narrowest_type

        signatures = [branch.signature for branch in self.branches]
        common_input = narrowest_type([s.i for s in signatures])

        if not all(s.infered for s in signatures) or common_input is None:
            return IcoSignature(i=type[Any], c=None, o=type[Any], infered=False)

        outputs = tuple(s.o for s in signatures)
        return IcoSignature(i=common_input, c=None, o=tuple[outputs])  # type: ignore[valid-type]


# ────────────────────────────────────────────────
# Factory with precise static types
# ────────────────────────────────────────────────


@overload
def fork(
    b1: Callable[[I], O1],
    b2: Callable[[I], O2],
    /,
    *,
    name: str | None = None,
    strict: bool | None = None,
) -> IcoOperatorProtocol[I, tuple[O1, O2]]: ...


@overload
def fork(
    b1: Callable[[I], O1],
    b2: Callable[[I], O2],
    b3: Callable[[I], O3],
    /,
    *,
    name: str | None = None,
    strict: bool | None = None,
) -> IcoOperatorProtocol[I, tuple[O1, O2, O3]]: ...


@overload
def fork(
    b1: Callable[[I], O1],
    b2: Callable[[I], O2],
    b3: Callable[[I], O3],
    b4: Callable[[I], O4],
    /,
    *,
    name: str | None = None,
    strict: bool | None = None,
) -> IcoOperatorProtocol[I, tuple[O1, O2, O3, O4]]: ...


@overload
def fork(
    b1: Callable[[I], O1],
    b2: Callable[[I], O2],
    b3: Callable[[I], O3],
    b4: Callable[[I], O4],
    b5: Callable[[I], O5],
    /,
    *,
    name: str | None = None,
    strict: bool | None = None,
) -> IcoOperatorProtocol[I, tuple[O1, O2, O3, O4, O5]]: ...


@overload
def fork(
    b1: Callable[[I], O1],
    b2: Callable[[I], O2],
    b3: Callable[[I], O3],
    b4: Callable[[I], O4],
    b5: Callable[[I], O5],
    b6: Callable[[I], O6],
    /,
    *,
    name: str | None = None,
    strict: bool | None = None,
) -> IcoOperatorProtocol[I, tuple[O1, O2, O3, O4, O5, O6]]: ...


@overload
def fork(
    *branches: Callable[[I], Any],
    name: str | None = None,
    strict: bool | None = None,
) -> IcoOperatorProtocol[I, tuple[Any, ...]]: ...


def fork(
    *branches: Callable[[I], Any],
    name: str | None = None,
    strict: bool | None = None,
) -> IcoOperatorProtocol[I, tuple[Any, ...]]:
    """Create a fork: apply every branch to the same input, return a tuple.

    Overloads give exact output tuple types for 2–6 branches; with more
    branches the output is typed as ``tuple[Any, ...]``.

    Args:
        *branches: Two or more callables or operators accepting the same input.
        name: Optional name for the fork (defaults to "fork").
        strict: Check branch input compatibility at runtime; ``None`` follows
            the current ``strict_signatures()`` context.

    Example:
        >>> features = fork(IcoOperator(extract_x), IcoOperator(extract_y))
        >>> flow = load | features | train  # train: tuple[X, Y] → Model
    """
    return IcoFork[I, tuple[Any, ...]](*branches, name=name, strict=strict)
