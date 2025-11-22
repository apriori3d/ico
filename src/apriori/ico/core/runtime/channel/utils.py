from typing import TypeVar

from apriori.ico.core.runtime.channel.channel import IcoReceiveEndpoint
from apriori.ico.core.runtime.command import IcoRuntimeCommand, IcoRuntimeCommandType
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode, IcoRuntimeState

T = TypeVar("T")


def wait_for_output(
    runtime: IcoRuntimeNode,
    endpoint: IcoReceiveEndpoint[T],
    accept_commands: bool = True,
    accept_events: bool = True,
) -> T | None:
    """Blocking wait for an incoming item."""
    while True:
        runtime.state = IcoRuntimeState.waiting
        input = endpoint.receive()

        # Process runtime commands/events
        if isinstance(input, IcoRuntimeCommand):
            if not accept_commands:
                raise RuntimeError(f"Runtime commands ({input}) can are not accepted.")

            runtime.on_command(input)

            # Exit loop on deactivate command
            if input.type == IcoRuntimeCommandType.deactivate:
                return None

            continue  # Wait for actual data item

        if isinstance(input, IcoRuntimeEvent):
            if not accept_events:
                raise RuntimeError(f"Runtime events ({input}) can are not accepted.")

            runtime.bubble_event(input)

            continue  # Wait for actual data item

        break  # Exit loop on data item

    # Ready with data item
    runtime.state = IcoRuntimeState.ready

    return input
