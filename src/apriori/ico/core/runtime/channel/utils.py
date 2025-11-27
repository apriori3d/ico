from queue import Empty
from typing import TypeVar

from apriori.ico.core.runtime.channel.channel import IcoReceiveEndpoint
from apriori.ico.core.runtime.command import IcoRuntimeCommand, IcoRuntimeCommandType
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode, IcoRuntimeState

T = TypeVar("T")


def wait_for_item(
    *,
    endpoint: IcoReceiveEndpoint[T],
    runtime_node: IcoRuntimeNode | None = None,
    accept_commands: bool = True,
    accept_events: bool = True,
    ignore_timeouts: bool = True,
) -> T | None:
    """Blocking wait for an incoming item."""
    while True:
        if runtime_node is not None:
            runtime_node.state = IcoRuntimeState.waiting

        try:
            input = endpoint.receive()
        except (TimeoutError, Empty):
            if ignore_timeouts:
                continue
            else:
                raise

        # Process runtime commands/events
        if isinstance(input, IcoRuntimeCommand):
            if not accept_commands:
                raise RuntimeError(f"Runtime commands ({input}) can are not accepted.")

            if runtime_node is not None:
                runtime_node.on_command(input)

            # Exit loop on deactivate command
            if input.type == IcoRuntimeCommandType.deactivate:
                return None

            continue  # Wait for actual data item

        if isinstance(input, IcoRuntimeEvent):
            if not accept_events:
                raise RuntimeError(f"Runtime events ({input}) can are not accepted.")

            if runtime_node is not None:
                runtime_node.bubble_event(input)

            continue  # Wait for actual data item

        break  # Exit loop on data item

    # Ready with data item
    if runtime_node is not None:
        runtime_node.state = IcoRuntimeState.ready

    return input
