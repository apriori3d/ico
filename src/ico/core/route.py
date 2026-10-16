from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar, Union

from ico.core.operator import I, IcoOperator, IcoOperatorProtocol, O, wrap_operator
from ico.core.signature import IcoSignature, is_supported_type

K = TypeVar("K")


class IcoRouteError(LookupError):
    """Raised when a route key matches no branch and there is no default."""

    def __init__(self, key: object, route_name: str) -> None:
        super().__init__(f"route '{route_name}': no branch for key {key!r}")
        self.key = key
        self.route_name = route_name


class IcoRoute(Generic[I, K, O], IcoOperator[I, O]):
    """Dispatch each item to one branch selected by a key function.

    Generic Parameters:
        I: Input type accepted by the key function and every branch.
        K: Key type (must be hashable).
        O: Output type of the branches.

    ICO signature:
        I → O  (O is the union of branch outputs)

    Example:
        >>> handle = route(
        ...     kind_of,
        ...     {"image": resize, "text": tokenize},
        ...     default=passthrough,
        ... )

    Keys are plain data, so a spec lists the available routes explicitly.

    Attributes:
        key: Operator computing the routing key of an item.
        routes: Branches by key, in declaration order.
        default: Branch for unknown keys, or ``None`` to raise
            :class:`IcoRouteError`.
    """

    __slots__ = ("key", "routes", "default")

    key: IcoOperatorProtocol[I, K]
    routes: dict[K, IcoOperatorProtocol[I, O]]
    default: IcoOperatorProtocol[I, O] | None

    def __init__(
        self,
        key: Callable[[I], K],
        routes: Mapping[K, Callable[[I], O]],
        *,
        default: Callable[[I], O] | None = None,
        name: str | None = None,
        strict: bool | None = None,
    ):
        """Initialize a route.

        Args:
            key: Callable or operator computing the routing key of an item.
            routes: Mapping from key to branch (callable or operator).
            default: Optional branch for keys missing from ``routes``.
            name: Optional name for this route (defaults to "route").
            strict: Check at runtime that the key function and all branches
                accept a common input type. ``None`` follows the current
                ``strict_signatures()`` context (off by default).

        Raises:
            ValueError: If there are no branches at all.
            IcoSignatureError: In strict mode, on incompatible input types.
        """
        if not routes and default is None:
            raise ValueError("Route requires at least one branch or a default.")

        key_op = wrap_operator(key)
        route_ops = {k: wrap_operator(branch) for k, branch in routes.items()}
        default_op = wrap_operator(default) if default is not None else None
        branches = [
            *route_ops.values(),
            *([default_op] if default_op is not None else []),
        ]

        from ico.core.signature_check import (
            IcoSignatureError,
            check_branch_inputs,
            is_strict,
        )

        route_name = name or "route"
        if strict if strict is not None else is_strict():
            issue = check_branch_inputs([key_op, *branches], route_name)
            if issue is not None:
                raise IcoSignatureError(issue)

        super().__init__(
            fn=self._route_fn,
            name=route_name,
            children=[key_op, *branches],
        )
        self.key = key_op
        self.routes = route_ops
        self.default = default_op

    def _route_fn(self, item: I) -> O:
        key = self.key(item)
        branch = self.routes.get(key, self.default)
        if branch is None:
            raise IcoRouteError(key, self.name or "route")
        return branch(item)

    @property
    def branches(self) -> list[IcoOperatorProtocol[I, O]]:
        """Routed branches in declaration order, followed by the default."""
        default = [self.default] if self.default is not None else []
        return [*self.routes.values(), *default]

    @property
    def signature(self) -> IcoSignature:
        """Infer the signature: narrowest common input → union of branch outputs.

        Uninferred if the key or any branch signature is uninferred, or if
        their inputs have no common type.
        """
        from ico.core.signature_check import narrowest_type

        signatures = [self.key.signature, *(b.signature for b in self.branches)]
        common_input = narrowest_type([s.i for s in signatures])
        output: Any = Union[tuple(s.o for s in signatures[1:])]  # noqa: UP007

        if (
            not all(s.infered for s in signatures)
            or common_input is None
            or not is_supported_type(output)
        ):
            return IcoSignature(i=type[Any], c=None, o=type[Any], infered=False)

        return IcoSignature(i=common_input, c=None, o=output)


def route(
    key: Callable[[I], K],
    routes: Mapping[K, Callable[[I], O]],
    *,
    default: Callable[[I], O] | None = None,
    name: str | None = None,
    strict: bool | None = None,
) -> IcoOperatorProtocol[I, O]:
    """Create a route: send each item to ``routes[key(item)]`` (or ``default``).

    Args:
        key: Computes the routing key of an item.
        routes: Branch per key; keys should be JSON scalars to be exportable
            as a spec.
        default: Branch for unknown keys; without it they raise
            :class:`IcoRouteError`.
        name: Optional name for the route (defaults to "route").
        strict: Check input compatibility at runtime; ``None`` follows the
            current ``strict_signatures()`` context.

    Example:
        >>> handle = route(kind_of, {"image": resize, "text": tokenize})
    """
    return IcoRoute(key, routes, default=default, name=name, strict=strict)
