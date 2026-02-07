from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Generic

from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.runtime.channel.channel import IcoChannel
from apriori.ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
    IcoRuntimeCommand,
)
from apriori.ico.core.runtime.event import IcoFaultEvent, IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRemotePlaceholderNode, IcoRuntimeNode
from apriori.ico.core.runtime.state import (
    BaseStateModel,
    IcoRuntimeState,
    ReadyState,
)
from apriori.ico.core.runtime.utils import discover_and_connect_runtime_nodes
from apriori.ico.core.signature import IcoSignature
from apriori.ico.core.signature_utils import infer_from_flow_factory


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
        if not self.state.is_ready:
            raise RuntimeError(
                "Cannot transition to Waiting state from non-Ready state."
            )
        self.update_state(WaitingState())

    def sending(self) -> None:
        if not self.state.is_ready:
            raise RuntimeError(
                "Cannot transition to Sending state from non-Ready state."
            )
        self.update_state(SendingState())


class AgentWorkerStateModel(AgentStateModel):
    """State model for runtime agent workers."""

    def pending(self) -> None:
        if self.state.is_ready:
            raise RuntimeError("Cannot transition to Pending state from Ready state.")
        self.update_state(PendingState())


class IcoAgent(
    Generic[I, O],
    IcoOperator[I, O],
    IcoRuntimeNode,
    ABC,
):
    channel: IcoChannel[I, O] | None
    flow_factory: Callable[[], IcoOperator[I, O]]
    subtree_factory: Callable[[], IcoRuntimeNode]

    # Placeholder for worker in runtime tree (actual worker exists in separate process).
    _worker_placeholder: IcoRuntimeNode

    def __init__(
        self,
        *,
        channel: IcoChannel[I, O] | None = None,
        flow_factory: Callable[[], IcoOperator[I, O]],
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
        self.flow_factory = flow_factory
        self._worker_placeholder = IcoRemotePlaceholderNode()
        self._runtime_children.append(self._worker_placeholder)

    # ──────── Worker management ────────

    def get_remote_flow_factory(self) -> Callable[[], IcoOperator[I, O]]:
        return self.flow_factory

    @abstractmethod
    def get_remote_runtime_factory(self) -> Callable[[], IcoAgentWorker[I, O]]: ...

    def _activate_worker(self) -> None:
        pass

    def _deactivate_worker(self) -> None:
        pass

    # ──────── Data flow function ────────

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

    # ──────── Channel message handling ────────

    def on_channel_command(self, command: IcoRuntimeCommand) -> None:
        raise Exception("Channel commands should not be sent to the agent runtime.")

    def on_channel_event(self, event: IcoRuntimeEvent) -> None:
        # Ensure events from the worker are bubbled correctly in the runtime tree
        self.bubble_event(event, from_child=self._worker_placeholder)

    # ─────── Runtime API ────────

    def on_command(self, command: IcoRuntimeCommand) -> None:
        is_ready = self.state.is_ready
        if command.broadcast_order == "pre":
            super().on_command(command)

        match command:
            case IcoActivateCommand():
                assert command.broadcast_order == "pre"

                if not is_ready:
                    # Spawn an agent in pre-order, before sending a command downstream
                    # assert self.channel is None
                    self._activate_worker()

                assert self.channel is not None
                self.channel.send_command(command)

            case IcoDeactivateCommand():
                if self.state.is_ready or self.state.is_fault:
                    assert command.broadcast_order == "post"
                    assert self.channel is not None

                    self.channel.send_command(command)

                    # Deactivate worker in post-order, to ensure proper shutdown
                    self._deactivate_worker()

                    # Close channel queues
                    self.channel.close()
                    # self.channel = None

            case _:
                if is_ready:
                    assert self.channel is not None
                    self.channel.send_command(command)

        if command.broadcast_order == "post":
            super().on_command(command)

    @property
    def signature(self) -> IcoSignature:
        signature = super().signature

        if not signature.infered:
            signature = infer_from_flow_factory(self.flow_factory)

            if signature is None:
                signature = IcoSignature(i=Any, c=None, o=Any, infered=False)
        return signature


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
            state_model=state_model or AgentWorkerStateModel(),
        )
        self.flow = flow_factory()
        self.channel = channel

        # Connect to runtime port to enable command and event handling for remote runtime
        channel.runtime_port = self

        discover_and_connect_runtime_nodes(self, self.flow)

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
                self.bubble_event(IcoFaultEvent.create(e))
                continue

    # ──────── Channel message handling ────────

    def on_channel_command(self, command: IcoRuntimeCommand) -> None:
        self.broadcast_command(command)

    def on_channel_event(self, event: IcoRuntimeEvent) -> None:
        raise Exception(
            "Channel events should not be sent to the agent worker runtime."
        )

    # ─────── Runtime API ────────

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        # Send event to upstream runtime
        self.channel.send_event(event)
        return super().on_event(event)
