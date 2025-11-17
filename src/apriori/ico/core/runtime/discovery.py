from collections.abc import Iterator

from apriori.ico.core.runtime.types import ConnectedToIcoRuntime, IcoRuntimeTreeProtocol
from apriori.ico.core.types import IcoTreeProtocol

# ────────────────────────────────────────────────
# Runtime Discovery and Connection Utilities
# ────────────────────────────────────────────────


def discover_runtime(flow: IcoTreeProtocol) -> Iterator[IcoRuntimeTreeProtocol]:
    """Discover all runtime hosts within the given closure."""
    yield from _discover_runtime_deep(flow)


def _discover_runtime_deep(
    operator: IcoTreeProtocol,
    in_runtime_scope: bool = False,
) -> Iterator[IcoRuntimeTreeProtocol]:
    """Discover all runtime hosts within the given closure."""

    if isinstance(operator, ConnectedToIcoRuntime):
        # If we are already in a runtime scope, do not yield nested hosts
        if in_runtime_scope:
            return
        if operator.runtime is not None:
            yield operator.runtime
        in_runtime_scope = True

    for child in operator.children:
        yield from _discover_runtime_deep(child, in_runtime_scope)


def discover_and_connect_runtimes(
    runtime: IcoRuntimeTreeProtocol,
    flow: IcoTreeProtocol,
) -> None:
    """Discover and connect all runtime hosts within the given closure."""
    for nested_runtime in discover_runtime(flow):
        runtime.connect_runtime(nested_runtime)


def disconnect_all_runtimes(runtime: IcoRuntimeTreeProtocol) -> None:
    """Disconnect from all connected runtime hosts."""
    for nested_runtime in runtime.runtime_children:
        runtime.disconnect_runtime(nested_runtime)
