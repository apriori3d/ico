from __future__ import annotations

from collections.abc import Callable
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Generic, final

from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.runtime.channel.utils import wait_for_output
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode, IcoRuntimeState
from apriori.ico.core.runtime.progress.mixin import ProgressMixin
from apriori.ico.runtime.channels.mp_queue.channel import MPQueueChannel


@final
class MPProcessAgent(
    IcoRuntimeNode,
    Generic[I, O],
    ProgressMixin,
):
    _flow: IcoOperator[I, O]
    _channel: MPQueueChannel[O, I]

    def __init__(
        self,
        *,
        channel: MPQueueChannel[O, I],
        flow_factory: Callable[[], IcoOperator[I, O]],
        name: str | None = None,
    ) -> None:
        IcoRuntimeNode.__init__(
            self,
            name=name or "mp_process_agent",
        )
        ProgressMixin.__init__(self)

        self._channel = channel
        self._flow = flow_factory()

        # Connect to runtime port to enable command and event handling remote runtime
        channel.input.runtime_port = self

    def _run_loop(self) -> None:
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
                # Blocks internally until new input arrives in the input channel.

                result = wait_for_output(
                    self,
                    self._channel.input,
                    accept_events=False,
                )
                if result is None:
                    break  # Exit loop on deactivate command

                # Process input item through flow
                self.state = IcoRuntimeState.running
                output = self._flow(result)

                # Send output item upstream
                self.state = IcoRuntimeState.sending
                self._channel.output.send(output)

                # Ready for the next item
                self.state = IcoRuntimeState.ready

            except Exception as e:
                # Report runtime errors downstream to output channel and terminate
                self.state = IcoRuntimeState.fault
                self.bubble_event(IcoRuntimeEvent.exception(e))
                break

    def on_event(self, event: IcoRuntimeEvent) -> None:
        super().on_event(event)

        # Send event to upstream runtime
        self._channel.output.send(event)

    @staticmethod
    def spawn(
        channel: MPQueueChannel[I, O],
        flow_factory: Callable[[], IcoOperator[I, O]],
        *,
        mp_context: SpawnContext,
        name: str | None = None,
    ) -> SpawnProcess:
        process = mp_context.Process(
            target=MPProcessAgent[I, O]._process_fn,
            args=(channel, flow_factory, name),
        )
        process.start()
        return process

    @staticmethod
    def _process_fn(
        channel: MPQueueChannel[O, I],
        flow_factory: Callable[[], IcoOperator[I, O]],
        *,
        name: str | None = None,
    ) -> None:
        agent = MPProcessAgent[I, O](
            channel=channel,
            flow_factory=flow_factory,
            name=name,
        )
        # Run agent to start receiving and processing commands and items
        agent._run_loop()
