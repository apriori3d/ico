from __future__ import annotations

from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Generic, final

from apriori.ico.agents.mp_process.mp_process_agent import MPProcessAgent
from apriori.ico.channels.mp_queue.channel import MPQueueChannel
from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.runtime.command import IcoRuntimeCommand, IcoRuntimeCommandType
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode, IcoRuntimeState
from apriori.ico.core.runtime.progress.mixin import ProgressMixin


@final
class MPProcess(
    Generic[I, O],
    IcoOperator[I, O],
    IcoRuntimeNode,
    ProgressMixin,
):
    flow_factory: Callable[[], IcoOperator[O, I]]
    _channel: MPQueueChannel[I, O]
    _mp_context: SpawnContext
    _agent_process: SpawnProcess | None

    def __init__(
        self,
        flow_factory: Callable[[], IcoOperator[O, I]],
        *,
        name: str | None = None,
    ) -> None:
        mp_context = get_context("spawn")

        channel = MPQueueChannel[I, O](mp_context)

        IcoRuntimeNode.__init__(self, name=name or "mp_process")
        IcoOperator[I, O].__init__(
            self,
            fn=self._portal_fn,
            name=name,
            children=[],
        )
        ProgressMixin.__init__(self)

        self.flow_factory = flow_factory
        self._channel = channel
        self._mp_context = mp_context
        self._agent_process = None

        # Connect to runtime port to enable event handling from downstream remote runtime
        self._channel.input.runtime_port = self

    def _portal_fn(self, item: I) -> O:
        if self._state != IcoRuntimeState.ready:
            raise RuntimeError(
                "MPProcess agent is not ready. Call activate on runtime."
            )

        try:
            # Send item to agent process
            self._set_state(IcoRuntimeState.running)
            self._channel.output.send_item(item)

            # Wait for result from agent process
            self._set_state(IcoRuntimeState.waiting)
            while True:
                result = self._channel.input.receive()

                # Process runtime commands/events
                if isinstance(result, IcoRuntimeCommand):
                    raise RuntimeError(
                        f"Runtime commands ({result}) can not be send to agent host ({self.name}) runtime."
                    )
                if isinstance(result, IcoRuntimeEvent):
                    self.bubble_event(result)
                    continue  # Wait for actual result item

                break  # Exit loop on valid result item

            self._set_state(IcoRuntimeState.ready)
            return result

        except Exception:
            self._set_state(IcoRuntimeState.error)
            raise

    def on_command(self, command: IcoRuntimeCommand) -> None:
        super().on_command(command)

        match command.type:
            case IcoRuntimeCommandType.activate:
                self._spawn_agent()

            case IcoRuntimeCommandType.deactivate:
                self._shutdown_agent()
            case _:
                pass

    # ─── Agent process management ───

    def _spawn_agent(self) -> None:
        self._agent_process = MPProcessAgent[O, I].spawn(
            channel=self._channel.make_pair(),  # Invert channel for agent
            flow_factory=self.flow_factory,
            mp_context=self._mp_context,
        )

    def _shutdown_agent(self) -> None:
        if self._agent_process is None:
            return
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
