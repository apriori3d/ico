from __future__ import annotations

from collections.abc import Callable
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import final

from typing_extensions import Self

from apriori.ico.channels.mp_queue.channel import MPQueueChannel
from apriori.ico.core.runtime.channels.types import IcoRuntimeChannelProtocol
from apriori.ico.core.runtime.contour import IcoRuntimeContour
from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.exceptions import IcoStopExecutionSignal
from apriori.ico.core.runtime.progress.mixin import ProgressMixin
from apriori.ico.core.runtime.types import IcoRuntimeCommandType
from apriori.ico.core.types import I, IcoOperatorProtocol, O


@final
class MPProcessAgent(
    IcoRuntimeContour,
    ProgressMixin,
):
    def __init__(
        self,
        *,
        channel: IcoRuntimeChannelProtocol[I, O],
        flow_factory: Callable[[], IcoOperatorProtocol[I, O]],
        name: str | None = None,
    ) -> None:
        flow = flow_factory()
        closure = channel.receive | flow | channel.send

        super().__init__(closure=closure)
        self.name = name or f"MPProcessAgent-{id(self)}"

    def run_loop(self) -> Self:
        """
        Main execution loop of the process agent.

        The agent runs a blocking loop, continuously executing its runtime contour.
        Each iteration processes one input payload received from the input channel
        and produces one output payload via the output channel.

        The loop terminates on runtime commands:
            - stop
            - reset
            - deactivate
        """
        while True:
            try:
                # Execute the contour (receive → flow → send)
                # Blocks internally until new input arrives in the input channel.
                self.run()

            except IcoStopExecutionSignal:
                # Flow has completed naturally via runtime command deactivate
                break

            except Exception as e:
                # Report runtime errors downstream to output channel and terminate
                self.bubble_event(IcoRuntimeEvent.exception(e))
                break

        return self

    def on_command(self, command: IcoRuntimeCommandType) -> None:
        super().on_command(command)

        if command == IcoRuntimeCommandType.deactivate:
            # Stop the main execution loop
            raise IcoStopExecutionSignal()

    @staticmethod
    def spawn(
        *,
        mp_context: SpawnContext,
        channel: MPQueueChannel[I, O],
        flow_factory: Callable[[], IcoOperatorProtocol[I, O]],
        name: str | None = None,
        relay_progress: bool = True,
    ) -> SpawnProcess:
        process = mp_context.Process(
            target=MPProcessAgent._process_fn,
            args=(channel, flow_factory, name, relay_progress),
        )
        # TODO: relay_progress
        process.start()
        return process

    @staticmethod
    def _process_fn(
        channel: IcoRuntimeChannelProtocol[I, O],
        flow_factory: Callable[[], IcoOperatorProtocol[I, O]],
        name: str | None = None,
        relay_progress: bool = True,
    ) -> None:
        agent = MPProcessAgent(
            channel=channel,
            flow_factory=flow_factory,
            name=name,
        )
        # Run agent to start receiving and processing commands and items
        agent.run_loop()
