from __future__ import annotations

from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Generic, final

from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.runtime.agent import IcoAgent, IcoAgentWorker
from apriori.ico.core.runtime.channel.channel import IcoChannel
from apriori.ico.core.runtime.event import (
    IcoFaultEvent,
)
from apriori.ico.core.signature import IcoSignature
from apriori.ico.core.signature_utils import infer_from_flow_factory
from apriori.ico.runtime.agent.mp.mp_channel import (
    MPChannel,
)

# from apriori.ico.tools.printer.node import IcoPrinter


@final
class MPAgent(Generic[I, O], IcoAgent[I, O]):
    _agent_process: SpawnProcess | None
    _mp_context: SpawnContext
    # _print: IcoPrinter

    def __init__(
        self,
        flow_factory: Callable[[], IcoOperator[I, O]],
        *,
        name: str | None = None,
    ) -> None:
        # printer = IcoPrinter()

        IcoAgent.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            channel=None,
            flow_factory=flow_factory,
            name=name,
            # runtime_children=[printer],
        )

        self._mp_context = get_context("spawn")
        # Create channel to ensure subtree_factory can access channel
        self.channel = self._create_channel()

        self._agent_process = None
        # self._print = printer

    @property
    def is_alive(self) -> bool:
        """Check if the agent process is alive."""
        return self._agent_process.is_alive() if self._agent_process else False

    # ─── Agent process management ───

    def get_remote_runtime_factory(self) -> Callable[[], IcoAgentWorker[I, O]]:
        assert self.channel is not None

        worker_channel = self.channel.invert()  # Invert channel for worker
        flow_factory = self.flow_factory
        name = self.name

        # Return a pickleable factory that preserves generic typing
        return _WorkerFactory[I, O](
            worker_channel=worker_channel,
            flow_factory=flow_factory,
            name=name,
        )

    def _create_channel(self) -> MPChannel[I, O]:
        assert self._mp_context is not None
        return MPChannel[I, O](
            mp_context=self._mp_context,
            runtime_port=self,
            accept_commands=False,
            accept_events=True,
            strict_accept=True,
        )

    def _activate_worker(self) -> None:
        # Recreate channel for agent process to ensure closed channels are not reused
        self.channel = self._create_channel()
        self._agent_process = self.spawn_worker()

        super()._activate_worker()

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
                # self._print(f"Process Agent {self} worker joined.")

                # Check if worker exited properly
                # if self._agent_process.exitcode is None:
                #     self._print("⚠️ Worker did not exit (possibly stuck)")

                # elif self._agent_process.exitcode != 0:
                #     self._print(
                #         f"⚠️ Process Agent {self} worker exited with code {self._agent_process.exitcode}."
                #     )
        except Exception as e:
            # self._print(f"❌ Error while stopping agent {self}: {e}")
            self.bubble_event(IcoFaultEvent.create(e))

        finally:
            if self._agent_process.is_alive():
                # self._print(
                #     f"⚠️ Process Agent {self} did not terminate worker gracefully."
                # )
                self._agent_process.terminate()

            super()._deactivate_worker()

    def spawn_worker(self) -> SpawnProcess:
        worker_factory = self.get_remote_runtime_factory()
        process = self._mp_context.Process(
            target=MPAgent[I, O]._start_worker_process,
            args=(worker_factory,),
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

    # ────── Signature API ──────

    @property
    def signature(self) -> IcoSignature:
        signature = super().signature

        if signature.infered:
            signature = infer_from_flow_factory(self.flow_factory)

        return IcoSignature(
            i=signature.c,
            c=None,
            o=signature.c,
        )


@final
class _WorkerFactory(Generic[I, O]):
    """Pickleable factory for creating MPAgentWorker instances."""

    def __init__(
        self,
        worker_channel: IcoChannel[O, I],
        flow_factory: Callable[[], IcoOperator[I, O]],
        name: str | None,
    ) -> None:
        self.worker_channel = worker_channel
        self.flow_factory = flow_factory
        self.name = name

    def __call__(self) -> IcoAgentWorker[I, O]:
        return IcoAgentWorker[I, O](
            channel=self.worker_channel,
            flow_factory=self.flow_factory,
            name=self.name,
        )
