from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Generic

from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.runtime.channel.channel import IcoChannel
from apriori.ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
    IcoRuntimeCommand,
)
from apriori.ico.core.runtime.event import IcoFaultEvent, IcoRuntimeEvent
from apriori.ico.core.runtime.exceptions import IcoRuntimeError
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.state import (
    BaseStateModel,
    IcoRuntimeState,
    ReadyState,
)


@dataclass(slots=True, frozen=True)
class PendingState(IcoRuntimeState):
    name: ClassVar[str] = "Pending"


@dataclass(slots=True, frozen=True)
class WaitingState(ReadyState):
    name: ClassVar[str] = "Waiting"


@dataclass(slots=True, frozen=True)
class SendingState(ReadyState):
    name: ClassVar[str] = "Sending"


class AgentStateModel(BaseStateModel):
    """State model for runtime agents."""

    def waiting(self) -> None:
        if not self.state.is_ready():
            raise RuntimeError(
                "Cannot transition to Waiting state from non-Ready state."
            )
        self.state = WaitingState()

    def sending(self) -> None:
        if not self.state.is_ready():
            raise RuntimeError(
                "Cannot transition to Sending state from non-Ready state."
            )
        self.state = SendingState()


class AgentWorkerStateModel(AgentStateModel):
    """State model for runtime agent workers."""

    def pending(self) -> None:
        if self.state.is_ready():
            raise RuntimeError("Cannot transition to Pending state from Ready state.")
        self.state = PendingState()


class IcoAgent(
    Generic[I, O],
    IcoOperator[I, O],
    IcoRuntimeNode,
    ABC,
):
    channel: IcoChannel[I, O] | None
    subflow_factory: Callable[[], IcoOperator[I, O]]

    def __init__(
        self,
        *,
        channel: IcoChannel[I, O] | None = None,
        subflow_factory: Callable[[], IcoOperator[I, O]],
        name: str | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
        state_model: BaseStateModel | None = None,
    ) -> None:
        # Note: pylance cannot infer IcoOperator.__init__ from Generic inheritance, but mypy can.
        IcoOperator.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, fn=self._portal_fn, name=name
        )
        IcoRuntimeNode.__init__(
            self,
            runtime_name=name,
            runtime_children=runtime_children,
            state_model=state_model or AgentStateModel(),
        )
        self.channel = channel
        self.subflow_factory = subflow_factory

    @abstractmethod
    def worker_factory(self) -> Callable[[], IcoAgentWorker[I, O]]: ...

    @abstractmethod
    def _activate_worker(self) -> None: ...

    @abstractmethod
    def _deactivate_worker(self) -> None: ...

    def _portal_fn(self, input: I) -> O:
        assert self.channel is not None
        assert isinstance(self.state_model, AgentStateModel)

        try:
            # Send item to agent process
            self.state_model.sending()
            self.channel.send(input)

            # Wait for result from agent process
            self.state_model.waiting()
            output = self.channel.wait_for_item()

            self.state_model.ready()

            # MPProcess should always receive an output here
            assert output is not None
            return output

        except Exception:
            self.state_model.fault()
            raise

    def on_command(self, command: IcoRuntimeCommand) -> None:
        if command.broadcast_order == "pre":
            super().on_command(command)

        match command:
            case IcoActivateCommand():
                assert command.broadcast_order == "pre"
                # Spawn an agent in pre-order, before sending a command downstream
                assert self.channel is None
                self._activate_worker()
                assert self.channel is not None

                self.channel.send_command(command)

            case IcoDeactivateCommand():
                assert command.broadcast_order == "post"
                assert self.channel is not None

                self.channel.send_command(command)

                # Deactivate worker in post-order, to ensure proper shutdown
                self._deactivate_worker()

                # Close channel queues
                self.channel.close()
                self.channel = None

            case _:
                assert self.channel is not None
                self.channel.send_command(command)

        if command.broadcast_order == "post":
            super().on_command(command)

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        if isinstance(event, IcoFaultEvent):
            # Raise exception received from agent process
            raise IcoRuntimeError(
                f"Agent event fault received: {event.info['message']}"
            )
        return super().on_event(event)


class IcoAgentWorker(
    Generic[I, O],
    IcoRuntimeNode,
    ABC,
):
    flow: IcoOperator[I, O]
    channel: IcoChannel[O, I]

    def __init__(
        self,
        *,
        channel: IcoChannel[O, I],
        flow_factory: Callable[[], IcoOperator[I, O]],
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
        state_model: BaseStateModel | None = None,
        name: str | None = None,
    ) -> None:
        IcoRuntimeNode.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            runtime_name=name,
            runtime_parent=runtime_parent,
            runtime_children=runtime_children,
            state_model=state_model or AgentStateModel(),
        )
        self.flow = flow_factory()
        self.channel = channel

        # Connect to runtime port to enable command and event handling for remote runtime
        channel.runtime_port = self

    def run_loop(self) -> None:
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
        assert isinstance(self.state_model, AgentWorkerStateModel)

        # Agent is now pending activation
        self.state_model.pending()

        while True:
            try:
                # Blocks internally until new input arrives in the input channel.
                input_item = self.channel.wait_for_item()

                if input_item is None:
                    self.state_model.idle()
                    break  # Exit loop on deactivate command

                # Process input item through flow
                self.state_model.running()
                output = self.flow(input_item)

                # Send output item upstream
                self.state_model.sending()
                self.channel.send(output)

                # Wait for the next item
                self.state_model.waiting()

            except Exception as e:
                # Report runtime errors downstream to output channel and terminate
                self.state_model.fault()
                self.bubble_event(IcoFaultEvent.create(e), from_child=self)
                continue

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        # Send event to upstream runtime
        self.channel.send_event(event)
        return super().on_event(event)
