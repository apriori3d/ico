from __future__ import annotations

from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Generic, final

from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.runtime.channel.channel import IcoChannel
from apriori.ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
    IcoRuntimeCommand,
)
from apriori.ico.core.runtime.event import (
    IcoFaultEvent,
    IcoRuntimeEvent,
)
from apriori.ico.core.runtime.exceptions import IcoRuntimeError
from apriori.ico.core.runtime.node import IcoRuntimeNode, IcoRuntimeState
from apriori.ico.runtime.agent.mp_process.mp_process_agent import MPProcessAgent
from apriori.ico.runtime.channel.mp_queue.channel import (
    MPChannel,
)
from apriori.ico.tools.printer.node import IcoPrinter


@final
class MPProcess(
    Generic[I, O],
    IcoOperator[I, O],
    IcoRuntimeNode,
):
    type_name: str = "agent"

    flow_factory: Callable[[], IcoOperator[I, O]]

    _agent_process: SpawnProcess | None
    _channel: IcoChannel[I, O] | None
    _mp_context: SpawnContext
    _print: IcoPrinter

    def __init__(
        self,
        flow_factory: Callable[[], IcoOperator[I, O]],
        *,
        name: str | None = None,
    ) -> None:
        name = name or "mp_process"
        printer = IcoPrinter()

        IcoRuntimeNode.__init__(
            self,
            runtime_name=name,
            runtime_children=[printer],
        )

        # Note: pylance cannot infer IcoOperator.__init__ from Generic inheritance, but mypy can.
        IcoOperator.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            fn=self._portal_fn,
            ico_form_target=flow_factory,
            name=name,
        )

        self.flow_factory = flow_factory
        self._mp_context = get_context("spawn")
        self._channel = None
        self._agent_process = None
        self._print = printer

    @property
    def is_alive(self) -> bool:
        """Check if the agent process is alive."""
        return self._agent_process.is_alive() if self._agent_process else False

    def _portal_fn(self, input: I) -> O:
        assert self._state is IcoRuntimeState.ready
        assert self._channel is not None

        try:
            # Send item to agent process
            self._set_state(IcoRuntimeState.running)
            self._channel.send(input)

            # Wait for result from agent process
            self._set_state(IcoRuntimeState.waiting)
            output = self._channel.wait_for_item()
            self._set_state(IcoRuntimeState.ready)

            # MPProcess should always receive an output here
            assert output is not None
            return output

        except Exception:
            self._set_state(IcoRuntimeState.fault)
            raise

    def on_command(self, command: IcoRuntimeCommand) -> IcoRuntimeCommand:
        match command:
            case IcoActivateCommand():
                assert self._channel is None

                self._channel = MPChannel[I, O](
                    mp_context=self._mp_context,
                    runtime_port=self,
                    accept_commands=False,
                    accept_events=True,
                    strict_accept=True,
                )
                # Spawn agent before sending a command downstream
                self._spawn_agent()
                next_command = self._channel.send_command(command)

            case IcoDeactivateCommand():
                assert self._channel is not None
                # Use post-order propagation to ensure children are deactivated before parents
                next_command = self._channel.send_command(command)
                self._shutdown_agent()

                # Close channel queues
                self._channel.close()
                self._channel = None

            case _:
                assert self._channel is not None
                next_command = self._channel.send_command(command)

        return super().on_command(next_command)

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        if isinstance(event, IcoFaultEvent):
            # Raise exception received from agent process
            raise IcoRuntimeError(
                f"Agent event fault received: {event.info['message']}"
            )
        return super().on_event(event)

    # ─── Agent process management ───

    def _spawn_agent(self) -> None:
        assert self._channel is not None and isinstance(self._channel, MPChannel)
        # Invert channel for agent

        self._agent_process = MPProcessAgent[I, O].spawn(
            channel=self._channel.make_agent_channel(),
            flow_factory=self.flow_factory,
            mp_context=self._mp_context,
            name=f"{self.name}_agent",
        )

    def _shutdown_agent(self) -> None:
        assert self._agent_process is not None

        try:
            # Gracefully join the worker process
            if self._agent_process.is_alive():
                # Notify agent to shutdown befor closing agent process and channels queues
                # self.channel.send.send_command(IcoRuntimeCommandType.deactivate)

                # Wait for agent process to exit
                self._agent_process.join(timeout=5)

                # Note: Queues will be closed by the channels themselves after command propagation downstream
                print(f"Process Agent {self.name} worker joined.")

                # Check if worker exited properly
                if self._agent_process.exitcode is None:
                    self._print("⚠️ Worker did not exit (possibly stuck)")

                elif self._agent_process.exitcode != 0:
                    self._print(
                        f"⚠️ Process Agent {self.name} worker exited with code {self._agent_process.exitcode}."
                    )
        except Exception as e:
            self._print(f"❌ Error while stopping agent {self.name}: {e}")
            self.bubble_event(IcoFaultEvent.exception(e))

        finally:
            if self._agent_process.is_alive():
                self._print(
                    f"⚠️ Process Agent {self.name} did not terminate worker gracefully."
                )
                self._agent_process.terminate()
