from __future__ import annotations

from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Generic, final

from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.runtime.agent import IcoAgent, IcoAgentWorker
from apriori.ico.core.runtime.event import (
    IcoFaultEvent,
)
from apriori.ico.runtime.agent.mp.mp_channel import (
    MPChannel,
)
from apriori.ico.tools.printer.node import IcoPrinter


@final
class MPAgent(Generic[I, O], IcoAgent[I, O]):
    _agent_process: SpawnProcess | None
    _mp_context: SpawnContext
    _print: IcoPrinter

    def __init__(
        self,
        subflow_factory: Callable[[], IcoOperator[I, O]],
        *,
        name: str | None = None,
    ) -> None:
        printer = IcoPrinter()

        IcoAgent.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            channel=None,
            subflow_factory=subflow_factory,
            name=name,
            runtime_children=[printer],
        )

        self._mp_context = get_context("spawn")
        self._agent_process = None
        self._print = printer

    @property
    def is_alive(self) -> bool:
        """Check if the agent process is alive."""
        return self._agent_process.is_alive() if self._agent_process else False

    # ─── Agent process management ───

    def worker_factory(self) -> Callable[[], IcoAgentWorker[I, O]]:
        assert self.channel is not None

        worker_channel = self.channel.invert()  # Invert channel for worker
        flow_factory = self.subflow_factory
        name = self.name

        def _create_worker():
            return IcoAgentWorker[I, O](
                channel=worker_channel,
                flow_factory=flow_factory,
                name=name,
            )

        return _create_worker

    def _activate_worker(self) -> None:
        self.channel = MPChannel[I, O](
            mp_context=self._mp_context,
            runtime_port=self,
            accept_commands=False,
            accept_events=True,
            strict_accept=True,
        )
        self._agent_process = self.spawn_worker()

    def _deactivate_worker(self) -> None:
        assert self._agent_process is not None

        try:
            # Gracefully join the worker process
            if self._agent_process.is_alive():
                # Notify agent to shutdown befor closing agent process and channels queues
                # self.channel.send.send_command(IcoRuntimeCommandType.deactivate)

                # Wait for agent process to exit
                self._agent_process.join(timeout=5)

                # Note: Queues will be closed by the channels themselves after command propagation downstream
                print(f"Process Agent {self} worker joined.")

                # Check if worker exited properly
                if self._agent_process.exitcode is None:
                    self._print("⚠️ Worker did not exit (possibly stuck)")

                elif self._agent_process.exitcode != 0:
                    self._print(
                        f"⚠️ Process Agent {self} worker exited with code {self._agent_process.exitcode}."
                    )
        except Exception as e:
            self._print(f"❌ Error while stopping agent {self}: {e}")
            self.bubble_event(IcoFaultEvent.create(e))

        finally:
            if self._agent_process.is_alive():
                self._print(
                    f"⚠️ Process Agent {self} did not terminate worker gracefully."
                )
                self._agent_process.terminate()

    def spawn_worker(self) -> SpawnProcess:
        process = self._mp_context.Process(
            target=MPAgent[I, O]._start_worker_process,
            args=(self.worker_factory(),),
        )
        process.start()
        return process

    @staticmethod
    def _start_worker_process(
        worker_factory: Callable[[], IcoAgentWorker[I, O]],
    ) -> None:
        worker = worker_factory()
        # Run agent to start receiving and processing commands and items
        worker.run_loop()
