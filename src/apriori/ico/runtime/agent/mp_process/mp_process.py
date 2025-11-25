from __future__ import annotations

from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Generic, final

from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.runtime.channel.utils import wait_for_item
from apriori.ico.core.runtime.command import IcoRuntimeCommand, IcoRuntimeCommandType
from apriori.ico.core.runtime.event import IcoRuntimeEvent, IcoRuntimeEventType
from apriori.ico.core.runtime.exceptions import IcoRuntimeError
from apriori.ico.core.runtime.node import IcoRuntimeNode, IcoRuntimeState
from apriori.ico.core.runtime.progress.mixin import ProgressMixin
from apriori.ico.runtime.agent.mp_process.mp_process_agent import MPProcessAgent
from apriori.ico.runtime.channel.mp_queue.channel import MPQueueChannel


@final
class MPProcess(
    Generic[I, O],
    IcoOperator[I, O],
    IcoRuntimeNode,
    ProgressMixin,
):
    flow_factory: Callable[[], IcoOperator[O, I]]

    _agent_process: SpawnProcess | None
    _channel: MPQueueChannel[I, O] | None
    _mp_context: SpawnContext

    def __init__(
        self,
        flow_factory: Callable[[], IcoOperator[O, I]],
        *,
        name: str | None = None,
    ) -> None:
        name = name or "mp_process"
        IcoRuntimeNode.__init__(self, name=name)
        # Note: pylance cannot infer IcoOperator.__init__ from Generic inheritance, but mypy can.
        IcoOperator.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            fn=self._portal_fn,
            name=name,
        )
        ProgressMixin.__init__(self)

        self.flow_factory = flow_factory
        self._mp_context = get_context("spawn")
        self._channel = None
        self._agent_process = None

    @property
    def is_alive(self) -> bool:
        """Check if the agent process is alive."""
        return self._agent_process.is_alive() if self._agent_process else False

    def _portal_fn(self, input: I) -> O:
        assert self._state is IcoRuntimeState.ready
        assert self._channel is not None

        try:
            # Send item to agent process
            self.state = IcoRuntimeState.running
            self._channel.output.send(input)

            # Wait for result from agent process
            output = wait_for_item(
                endpoint=self._channel.input,
                runtime_node=self,
                accept_commands=False,
            )
            # MPProcess should always receive an output here
            assert output is not None
            return output

        except Exception:
            self.state = IcoRuntimeState.fault
            raise

    def on_command(self, command: IcoRuntimeCommand) -> None:
        super().on_command(command)

        match command.type:
            case IcoRuntimeCommandType.activate:
                assert self._channel is None

                self._channel = MPQueueChannel[I, O](self._mp_context)
                # Connect to runtime port to enable event handling from downstream remote runtime
                self._channel.input.runtime_port = self

                # Spawn agent before sending a command downstream
                self._spawn_agent()
                self._channel.output.send(command)

            case IcoRuntimeCommandType.deactivate:
                assert self._channel is not None

                # Send deactivate command downstream before shutting down an agent
                self._channel.output.send(command)
                self._shutdown_agent()

                # Close channel queues
                self._channel.close()
                self._channel = None

            case _:
                pass

    def on_event(self, event: IcoRuntimeEvent) -> None:
        if event.type == IcoRuntimeEventType.fault:
            # Raise exception received from agent process
            raise IcoRuntimeError(
                f"Agent event fault received: {event.meta['message']}"
            )

    # ─── Agent process management ───

    def _spawn_agent(self) -> None:
        assert self._channel is not None

        self._agent_process = MPProcessAgent[O, I].spawn(
            channel=self._channel.make_pair(),  # Invert channel for agent
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
                    self.progress.print("⚠️ Worker did not exit (possibly stuck)")

                elif self._agent_process.exitcode != 0:
                    self.progress.print(
                        f"⚠️ Process Agent {self.name} worker exited with code {self._agent_process.exitcode}."
                    )
        except Exception as e:
            self.progress.print(f"❌ Error while stopping agent {self.name}: {e}")
            self.bubble_event(IcoRuntimeEvent.exception(e))

        finally:
            if self._agent_process.is_alive():
                # self.progress.print(
                #     f"⚠️ Process Agent {self.name} did not terminate worker gracefully."
                # )
                self._agent_process.terminate()
