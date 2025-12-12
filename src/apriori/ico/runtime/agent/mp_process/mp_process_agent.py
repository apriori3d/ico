from __future__ import annotations

from collections.abc import Callable
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Generic, final

from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.runtime.agent import AgentStateModel, IcoAgentNode
from apriori.ico.core.runtime.channel.channel import IcoChannel
from apriori.ico.core.runtime.contour import discover_and_connect_runtime_subtrees
from apriori.ico.core.runtime.event import IcoFaultEvent, IcoRuntimeEvent


@final
class MPProcessAgent(
    IcoAgentNode,
    Generic[I, O],
):
    _flow: IcoOperator[I, O]
    _channel: IcoChannel[O, I]

    def __init__(
        self,
        *,
        channel: IcoChannel[O, I],
        flow_factory: Callable[[], IcoOperator[I, O]],
        name: str | None = None,
    ) -> None:
        IcoAgentNode.__init__(self, name=name)
        self._channel = channel
        self._flow = flow_factory()

        discover_and_connect_runtime_subtrees(self, self._flow)

        # Connect to runtime port to enable command and event handling for remote runtime
        channel.runtime_port = self

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
        assert isinstance(self.state_model, AgentStateModel)

        # Agent is now pending activation
        self.state_model.pending()

        while True:
            try:
                # Blocks internally until new input arrives in the input channel.
                input_item = self._channel.wait_for_item()

                if input_item is None:
                    self.state_model.idle()
                    break  # Exit loop on deactivate command

                # Process input item through flow
                self.state_model.running()
                output = self._flow(input_item)

                # Send output item upstream
                self.state_model.sending()
                self._channel.send(output)

                # Wait for the next item
                self.state_model.waiting()

            except Exception as e:
                # Report runtime errors downstream to output channel and terminate
                self.state_model.fault()
                self.bubble_event(IcoFaultEvent.exception(e))
                continue

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        # Send event to upstream runtime
        self._channel.send_event(event)
        return super().on_event(event)

    @staticmethod
    def spawn(
        channel: IcoChannel[O, I],
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
        channel: IcoChannel[O, I],
        flow_factory: Callable[[], IcoOperator[I, O]],
        name: str | None = None,
    ) -> None:
        agent = MPProcessAgent[I, O](
            channel=channel,
            flow_factory=flow_factory,
            name=name,
        )
        # Run agent to start receiving and processing commands and items
        agent._run_loop()
