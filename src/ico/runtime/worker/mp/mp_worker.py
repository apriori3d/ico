from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Any, Generic, final

from ico.core.operator import I, IcoOperatorProtocol, O
from ico.core.plan_keys import remote_flow_key
from ico.core.probe import IcoRunnerProbe
from ico.core.runtime.channel.channel import IcoChannel
from ico.core.runtime.event import (
    IcoFaultEvent,
)
from ico.core.runtime.worker import IcoRunner, IcoWorker, IcoWorkerDied
from ico.core.signature import IcoSignature
from ico.core.signature_utils import infer_from_flow_factory
from ico.runtime.worker.mp.mp_channel import (
    MPChannel,
)

# ────────────────────────────────────────────────────────────────────────────
# Multiprocessing-based distributed worker implementation
# ────────────────────────────────────────────────────────────────────────────


DEFAULT_START_TIMEOUT = 300.0
"""Seconds a runner has to start by default: enough to import heavy
frameworks and load a model, short of an endless hang."""
ABORT_GRACE = 5.0
"""Seconds a runner that did not start has to exit after ``terminate``."""


@final
class MPWorker(Generic[I, O], IcoWorker[I, O]):
    """Multiprocessing-based distributed worker for ICO computation execution.

    MPWorker implements distributed computation by spawning separate runner processes
    and coordinating with them through MPChannel-based inter-process communication.
    It provides transparent distributed execution while maintaining full integration
    with ICO runtime tree and computation flow architecture.

    Distributed Architecture:
        • **Process Separation**: Worker (portal) and Runner (computation) in separate processes
        • **Channel Communication**: Bidirectional MPChannel for data and runtime coordination
        • **Factory Pattern**: Flow factory for creating computation logic in runner process
        • **Process Lifecycle**: Complete runner process lifecycle management

    Worker/Runner Collaboration:
        **Worker Process (Portal)**:
        • Acts as computation node in main compute flow
        • Forwards data items to remote runner process
        • Receives computed results from runner
        • Manages runner process lifecycle and error handling

        **Runner Process (Remote)**:
        • Executes actual computation flow in isolated process
        • Participates in distributed runtime tree
        • Sends progress/status events back to worker
        • Independent memory space and Python interpreter

    Flow Factory Integration:
        MPWorker uses factory pattern for runner process initialization:
        • Flow factory creates computation logic in runner process
        • Factory must be pickleable for cross-process transmission
        • Runner process reconstructs computation flow from factory
        • Full computation isolation between worker and runner

    Runtime Tree Distribution:
        ```
        Main Process Runtime Tree          Runner Process Runtime Tree
        ┌─────────────────────┐           ┌─────────────────────┐
        │ Runtime             │           │ IcoRunner           │
        │ └── MPWorker(portal)│ ←─────→   │ └── Flow (actual)   │
        │     └── Channel     │  Channel  │     ├── Operators   │
        │                     │    IPC    │     └── Progress    │
        └─────────────────────┘           └─────────────────────┘
        ```

    Process Management:
        • **Spawn Context**: Uses 'spawn' multiprocessing for cross-platform compatibility
        • **Process Creation**: Runner process spawned during activation phase
        • **Graceful Shutdown**: Coordinated runner termination during deactivation
        • **Error Handling**: Runner exceptions propagated to worker via events
        • **Resource Cleanup**: Automatic channel and process resource cleanup

    Channel Configuration:
        MPWorker configures channels for worker/runner coordination:
        • **accept_commands=False**: Worker receives commands from runtime
        • **accept_events=True**: Worker accepts events from runner
        • **strict_accept=True**: Enforces strict message type validation
        • **Channel Inversion**: Runner gets inverted channel automatically

    Computation Transparency:
        From computation flow perspective, MPWorker appears as regular operator:
        • **Input**: Receives items from previous operators
        • **Output**: Produces results to next operators
        • **Type Safety**: Maintains Generic[I, O] type parameters
        • **Runtime Integration**: Full runtime tree participation

    Process Safety Features:
        • **Isolated Execution**: Runner process completely isolated from worker
        • **Exception Propagation**: Runner exceptions bubble to worker via IcoFaultEvent
        • **Resource Management**: Automatic cleanup of processes and channels
        • **Timeout Handling**: Configurable timeouts for runner operations
        • **Process Monitoring**: Runner process health monitoring and lifecycle tracking

    Usage Integration:
        MPWorker integrates seamlessly into ICO computation flows:
        • Can be used anywhere IcoOperator is expected
        • Supports method chaining (.stream()) and standard ICO flow composition (|)
        • Compatible with all ICO runtime tools and monitoring
        • Transparent to upstream and downstream operators

    Example - Distributed Computation Flow:
        ```python
        from ico.core import IcoSource, IcoSink, IcoOperator
        from ico.core.runtime.progress import IcoProgress
        from ico.core.runtime.runtime import IcoRuntime
        from ico.runtime.worker.mp.mp_worker import MPWorker
        from ico.tools.progress.rich_progress_tool import RichProgressTool
        import time

        # Define computation that will run in runner process
        def expensive_computation(item: int) -> int:
            time.sleep(0.1)  # Simulate expensive work
            return item ** 2

        # Factory for creating computation flow in runner process
        def create_worker_flow() -> IcoOperator[int, int]:
            # This flow runs in isolated runner process
            return IcoOperator(expensive_computation, name="compute_square")

        # Build main computation flow with distributed processing
        source = IcoSource(lambda: range(20), name="data_source")
        progress = IcoProgress[int](total=20, name="input_progress")

        # MPWorker runs expensive_computation in separate process
        distributed_worker = MPWorker[int, int](
            flow_factory=create_worker_flow,
            name="square_worker"
        )

        result_sink = IcoSink(
            lambda results: [print(f"Result: {r}") for r in results],
            name="result_sink"
        )

        # Compose distributed computation flow
        flow = source | progress | distributed_worker.stream() | result_sink

        # Execute with runtime coordination and progress monitoring
        runtime = IcoRuntime(flow, tools=[RichProgressTool()], name="distributed_runtime")
        runtime.activate().run().deactivate()

        # Output: Real-time progress bar + computed squares
        # Runner process runs expensive_computation in isolation
        # Results: 0, 1, 4, 9, 16, 25, 36, 49, 64, ...
        ```

    Performance Considerations:
        • **Process Overhead**: Multiprocessing has startup cost vs threading
        • **Serialization Cost**: Data serialization for cross-process communication
        • **Memory Isolation**: Runner process uses separate memory space
        • **GIL Avoidance**: True parallelism by avoiding Python GIL limitations

    Error Handling Strategy:
        • **Runner Exceptions**: Captured and forwarded as IcoFaultEvent
        • **Process Failures**: Detected during deactivation and reported
        • **Channel Errors**: Communication failures handled gracefully
        • **Timeout Conditions**: Configurable timeout with cleanup procedures

    Distributed Coordination:
        MPWorker coordinates distributed execution through:
        • **Runtime Commands**: Lifecycle commands (prepare/reset) sent to runner
        • **Progress Events**: Runner progress events bubbled to worker runtime tree
        • **State Synchronization**: Runner and worker state coordination
        • **Exception Reporting**: Runner exceptions reported to worker tools
    """

    _runner_process: SpawnProcess | None  # Runner process instance (None when inactive)
    _mp_context: SpawnContext  # Multiprocessing context for process/queue creation
    # _print: IcoPrinter                   # Optional printer for debugging (currently disabled)

    def __init__(
        self,
        flow_factory: Callable[[], IcoOperatorProtocol[I, O]],
        *,
        name: str | None = None,
        start_timeout: float | None = DEFAULT_START_TIMEOUT,
    ) -> None:
        """Initialize multiprocessing worker with computation flow factory.

        Args:
            flow_factory: Factory function creating computation flow in runner process.
                         Must be pickleable for cross-process transmission.
            name: Optional worker identifier for debugging and monitoring.
            start_timeout: Seconds the runner process has to start (import
                modules, build its flow, acknowledge activation); ``None``
                for no limit. A runner that hangs is stopped and activation
                raises ``IcoWorkerStartTimeout``.

        Initialization Process:
            1. **Parent Initialization**: Call IcoWorker.__init__ with None channel
            2. **Context Creation**: Create 'spawn' multiprocessing context
            3. **Channel Setup**: Create initial MPChannel for worker/runner communication
            4. **Process Preparation**: Initialize process tracking (None until activation)

        Flow Factory Requirements:
            • **Pickleable**: Factory must be serializable for process transmission
            • **Stateless**: Factory should not capture non-pickleable state
            • **Complete Flow**: Factory must create complete I → O computation
            • **Type Safety**: Factory return type must match Generic[I, O] parameters

        Multiprocessing Context:
            Uses 'spawn' context for maximum compatibility:
            • **Cross-Platform**: Works on Windows, macOS, and Linux
            • **Clean State**: Runner process starts with fresh interpreter
            • **Import Safety**: Avoids import-related multiprocessing issues
            • **Resource Isolation**: Complete memory and resource isolation

        Channel Pre-Creation:
            Channel created during initialization (before activation) to enable:
            • **Factory Access**: Runner factory can access channel configuration
            • **Runtime Discovery**: Runtime tree can discover channel relationships
            • **Configuration Completion**: All worker configuration available immediately
        """
        # printer = IcoPrinter()

        IcoWorker.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            channel=None,
            flow_factory=flow_factory,
            name=name,
            # runtime_children=[printer],
        )

        self.start_timeout = start_timeout
        self._mp_context = get_context("spawn")
        # Create channel to ensure subtree_factory can access channel
        self.channel = self._create_channel()

        self._runner_process = None
        # self._print = printer

    @property
    def is_alive(self) -> bool:
        """Check if runner process is currently running.

        Returns:
            True if runner process exists and is alive, False otherwise.

        Process Health Check:
            • **None Check**: Returns False if no process spawned yet
            • **Alive Check**: Uses multiprocessing.Process.is_alive() for actual status
            • **Real-time Status**: Reflects current runner process state

        Usage:
            Used for monitoring runner health and coordinating lifecycle operations.
            Worker tools can check runner status for debugging and error handling.
        """
        return self._runner_process.is_alive() if self._runner_process else False

    # ────────────────────────────────────────────────
    # Runner process lifecycle management
    # ────────────────────────────────────────────────

    def get_remote_runtime_factory(self) -> Callable[[], IcoRunner[I, O]]:
        """Create pickleable factory for runner process runtime initialization.

        Returns:
            Factory function that creates IcoRunner in runner process.

        Factory Creation Process:
            1. **Channel Inversion**: Invert worker channel for runner perspective
            2. **Factory Capture**: Capture flow_factory and configuration
            3. **Pickleable Wrapper**: Wrap in _WorkerFactory for serialization
            4. **Type Preservation**: Maintain Generic[I, O] typing through factory

        Cross-Process Serialization:
            Factory is designed for multiprocessing transmission:
            • **Pickleable Design**: _WorkerFactory implements __call__ for serialization
            • **State Capture**: Captures all necessary runner initialization data
            • **Channel Inversion**: Provides inverted channel to runner process
            • **Configuration Transfer**: Transfers name and flow_factory to runner

        Runner Process Integration:
            Factory creates fully configured IcoRunner in remote process:
            • **Runtime Tree**: Runner becomes root of distributed runtime tree
            • **Channel Assignment**: Runner gets inverted channel for worker communication
            • **Flow Creation**: Runner uses flow_factory to create computation logic
            • **Name Preservation**: Runner inherits worker name for debugging
        """
        assert self.channel is not None

        runner_channel = self.channel.invert()  # Invert channel for runner
        flow_factory = self.flow_factory
        name = self.name

        # Return a pickleable factory that preserves generic typing
        return _WorkerFactory[I, O](
            runner_channel=runner_channel,
            flow_factory=flow_factory,
            name=name,
            probe=self.runner_probe,
            flow_key=(
                remote_flow_key(self.plan_key, self)
                if self.plan_key is not None
                else None
            ),
        )

    def _create_channel(self) -> MPChannel[I, O]:
        """Create configured MPChannel for worker/runner communication.

        Returns:
            Configured MPChannel with worker-specific settings.

        Channel Configuration:
            • **Runtime Port**: Worker acts as runtime port for channel integration
            • **Command Handling**: accept_commands=False (worker receives from runtime)
            • **Event Handling**: accept_events=True (worker accepts runner events)
            • **Strict Validation**: strict_accept=True for robust message handling

        Worker-Specific Settings:
            Channel configured for worker role in distributed architecture:
            • **Worker → Runner**: Data messages flow from worker to runner
            • **Runner → Worker**: Result messages and events flow back
            • **Runtime Commands**: Flow from main runtime through worker to runner
            • **Runner Events**: Bubble from runner through worker to main runtime
        """
        assert self._mp_context is not None
        return MPChannel[I, O](
            mp_context=self._mp_context,
            runtime_port=self,
            accept_commands=False,
            accept_events=True,
            strict_accept=True,
        )

    def _activate_runner(self) -> None:
        """Initialize and start runner process for distributed computation.

        Activation Process:
            1. **Channel Recreation**: Create fresh channel to avoid reusing closed channels
            2. **Runner Spawn**: Create and start new runner process
            3. **Parent Activation**: Call parent class activation for runtime coordination

        Channel Recreation Strategy:
            Fresh channel created on each activation to ensure:
            • **Clean State**: No residual state from previous activations
            • **Resource Safety**: Avoid using closed/corrupted channels
            • **Process Isolation**: Each activation gets independent communication

        Process Creation:
            Runner process spawned through spawn_runner() with:
            • **Factory Injection**: Runner factory passed to process
            • **Process Start**: Runner process started immediately
            • **Resource Tracking**: Process reference stored for lifecycle management
        """
        # Recreate channel for worker process to ensure closed channels are not reused
        self.channel = self._create_channel()
        self._runner_process = self.spawn_runner()
        # A crashed runner ends every wait for it with IcoWorkerDied.
        self.channel.check_peer = self._check_runner

        super()._activate_runner()

    def _check_runner(self) -> None:
        """Raise ``IcoWorkerDied`` if the runner process has exited."""
        process = self._runner_process
        if process is not None and not process.is_alive():
            raise IcoWorkerDied(process.exitcode, self.name)

    def runner_alive(self) -> bool:
        return self.is_alive

    def _abort_runner(self) -> None:
        """Stop a runner process that did not start in time."""
        process = self._runner_process
        if process is None or not process.is_alive():
            return
        process.terminate()
        _wait_for_exit(process, timeout=ABORT_GRACE)
        if process.is_alive():
            process.kill()
            _wait_for_exit(process, timeout=ABORT_GRACE)

    def _deactivate_runner(self) -> None:
        """Gracefully shutdown runner process and cleanup resources.

        Deactivation Process:
            1. **Graceful Join**: Wait for runner process to complete
            2. **Timeout Handling**: Use configurable timeout for join operation
            3. **Exception Handling**: Capture and report runner shutdown issues
            4. **Forced Termination**: Terminate runner if graceful shutdown fails
            5. **Parent Deactivation**: Call parent cleanup for runtime coordination

        Graceful Shutdown Strategy:
            • **Join Timeout**: 5-second timeout for runner process completion
            • **Exit Code Check**: Monitor runner exit status for error detection
            • **Exception Propagation**: Runner exceptions bubbled as IcoFaultEvent
            • **Resource Cleanup**: Channels and process resources cleaned automatically

        Error Handling:
            • **Join Failures**: Exceptions during runner join reported as fault events
            • **Process Monitoring**: Runner process health monitored during shutdown
            • **Forced Cleanup**: Terminate() called if graceful shutdown fails
            • **Resource Safety**: Cleanup guaranteed even with exceptions

        Channel Lifecycle:
            Channels are closed automatically during deactivation:
            • **Command Propagation**: Commands sent before channel closure
            • **Event Collection**: Final events received before shutdown
            • **Queue Cleanup**: Underlying multiprocessing queues cleaned up
        """
        assert self._runner_process is not None

        try:
            # Wait for the runner process to exit
            _wait_for_exit(self._runner_process, timeout=5)

        except Exception as e:
            self.bubble_event(IcoFaultEvent.create(e))

        finally:
            if self._runner_process.is_alive():
                self._runner_process.terminate()

            super()._deactivate_runner()

    def spawn_runner(self) -> SpawnProcess:
        """Create and start runner process with runtime factory.

        Returns:
            Started SpawnProcess instance running runner computation.

        Process Creation:
            1. **Factory Generation**: Get remote runtime factory for runner initialization
            2. **Process Configuration**: Create process with runner entry point
            3. **Process Start**: Start process immediately after creation

        Runner Entry Point:
            Runner process executes _start_runner_process static method:
            • **Factory Execution**: Runner factory called to create IcoRunner
            • **Runtime Loop**: Runner enters run_loop() for continuous operation
            • **Process Isolation**: Complete isolation from worker process

        Multiprocessing Context:
            Uses configured spawn context for process creation:
            • **Cross-Platform**: Spawn context works on all platforms
            • **Clean Interpreter**: Runner starts with fresh Python interpreter
            • **Resource Isolation**: Independent memory space and file handles
        """
        worker_factory = self.get_remote_runtime_factory()
        process = self._mp_context.Process(
            target=MPWorker[I, O]._start_runner_process,
            args=(worker_factory,),
        )
        process.start()
        return process

    @staticmethod
    def _start_runner_process(
        worker_factory: Callable[[], IcoRunner[I, O]],
    ) -> None:
        """Static entry point for runner process execution.

        Args:
            worker_factory: Factory function for creating IcoRunner instance.

        Runner Process Lifecycle:
            1. **Factory Execution**: Create IcoRunner from factory
            2. **Runtime Loop**: Enter continuous operation loop
            3. **Command Processing**: Process commands and data from worker
            4. **Event Generation**: Send progress and status events to worker

        Process Isolation:
            Runner process operates independently:
            • **Separate Runtime Tree**: Runner has own runtime hierarchy
            • **Independent Tools**: Runner can have own monitoring tools
            • **Isolated Memory**: No shared memory with worker process
            • **Exception Isolation**: Runner exceptions don't crash worker

        Communication Flow:
            Runner communicates with worker through inverted channel:
            • **Receives**: Data items and runtime commands from worker
            • **Sends**: Computed results and runtime events to worker
            • **Bidirectional**: Full bidirectional communication capability
        """
        runner = worker_factory()
        # An orphaned runner (its worker's process was killed) exits instead
        # of waiting forever for the next item.
        runner.channel.check_peer = _exit_if_parent_gone
        # Run worker to start receiving and processing commands and items
        runner.run_loop()

    # ────────────────────────────────────────────────
    # Type signature and computation interface
    # ────────────────────────────────────────────────

    @property
    def signature(self) -> IcoSignature:
        """Compute worker signature for computation flow integration.

        Returns:
            IcoSignature representing worker's computation interface.

        Signature Computation:
            1. **Parent Signature**: Get base signature from IcoWorker
            2. **Inference**: Use flow_factory for signature inference if needed
            3. **Worker Transformation**: Convert to worker-specific signature format

        Worker Signature Format:
            Worker signature differs from standard operator signature:
            • **Input (i)**: Accepts items of type from parent signature (c)
            • **Context (c)**: None (worker doesn't expose internal context)
            • **Output (o)**: Produces items of type from parent signature (c)

        Signature Inference:
            When parent signature is inferred:
            • **Flow Factory Analysis**: Analyze flow_factory return type
            • **Type Extraction**: Extract I and O types from IcoOperator[I, O]
            • **Signature Construction**: Build proper IcoSignature from types

        Computation Transparency:
            Worker signature hides distributed implementation details:
            • **Input Interface**: Matches expected input type I
            • **Output Interface**: Matches expected output type O
            • **Process Abstraction**: Internal multiprocessing hidden from signature
            • **Type Safety**: Maintains Generic[I, O] type constraints
        """
        signature = super().signature

        if signature.infered:
            return signature

        factory_signature = infer_from_flow_factory(self.flow_factory)

        return factory_signature or IcoSignature(
            i=type(Any), o=type(Any), infered=False
        )


# ────────────────────────────────────────────────────────────────────────────
# Pickleable runner factory for cross-process initialization
# ────────────────────────────────────────────────────────────────────────────


def _wait_for_exit(process: SpawnProcess, timeout: float) -> None:
    """Wait until ``process`` exits or ``timeout`` seconds pass.

    Polls ``waitpid`` (``is_alive``) instead of ``Process.join``: ``join``
    waits for the process sentinel, a pipe whose write end the runner
    inherits, and so does every child the runner starts without closing
    its fds (e.g. ``torch_shm_manager``). Such a child outlives the runner
    and would make ``join`` wait for the whole timeout.
    """
    deadline = time.monotonic() + timeout
    delay = 0.001
    while process.is_alive() and time.monotonic() < deadline:
        time.sleep(delay)
        delay = min(delay * 2, 0.05)


def _exit_if_parent_gone() -> None:
    """In a runner process: exit when the parent process is gone."""
    parent = multiprocessing.parent_process()
    if parent is not None and not parent.is_alive():
        raise SystemExit(0)  # not an Exception: the runner loop does not catch it


@final
class _WorkerFactory(Generic[I, O]):
    """Pickleable factory for creating IcoRunner instances in runner processes.

    _WorkerFactory enables cross-process transmission of runner initialization
    data by implementing a pickleable callable that reconstructs IcoRunner
    with proper configuration in the target runner process.

    Pickleable Design:
        • **Serializable State**: All attributes are pickleable (channel, factory, name)
        • **Callable Interface**: Implements __call__ for factory pattern
        • **Generic Preservation**: Maintains Generic[I, O] typing through serialization
        • **Cross-Process Safe**: Safe for multiprocessing transmission

    Factory Encapsulation:
        Factory encapsulates all runner initialization requirements:
        • **Inverted Channel**: Pre-configured channel for runner communication
        • **Flow Factory**: Computation logic factory for runner process
        • **Configuration**: Name and other runner configuration data
        • **Type Safety**: Generic parameters preserved through factory

    Cross-Process Workflow:
        1. **Worker Process**: MPWorker creates _WorkerFactory with runner config
        2. **Serialization**: Factory pickled for transmission to runner process
        3. **Runner Process**: Factory unpickled and called to create IcoRunner
        4. **Runner Initialization**: IcoRunner created with proper configuration

    Channel Inversion Pattern:
        Factory receives inverted channel from worker:
        • **Worker Perspective**: MPChannel[I, O] for sending I, receiving O
        • **Runner Perspective**: MPChannel[O, I] for receiving I, sending O
        • **Automatic Inversion**: Channel inversion handled by worker before factory creation
        • **Bidirectional Communication**: Full communication capability in runner
    """

    def __init__(
        self,
        runner_channel: IcoChannel[O, I],
        flow_factory: Callable[[], IcoOperatorProtocol[I, O]],
        name: str | None,
        probe: IcoRunnerProbe | None = None,
        flow_key: str | None = None,
    ) -> None:
        """Initialize runner factory with runner configuration.

        Args:
            runner_channel: Pre-configured inverted channel for runner communication.
            flow_factory: Factory function for creating computation flow in runner.
            name: Optional runner name for debugging and identification.
            probe: The worker's ``runner_probe``, passed to the runner.
            flow_key: Plan key of the worker's remote flow, passed to the
                runner.

        Factory State:
            All arguments stored as instance attributes for pickling:
            • **runner_channel**: Channel configured for runner perspective
            • **flow_factory**: Computation logic factory for runner process
            • **name**: Runner identification for runtime tree integration

        Pickling Requirements:
            All stored objects must be pickleable:
            • **Channel**: MPChannel implements proper pickling support
            • **Flow Factory**: User-provided factory must be pickleable
            • **Name**: String or None (always pickleable)
        """
        self.runner_channel = runner_channel
        self.flow_factory = flow_factory
        self.name = name
        self.probe = probe
        self.flow_key = flow_key

    def __call__(self) -> IcoRunner[I, O]:
        """Create and return configured IcoRunner instance.

        Returns:
            Fully configured IcoRunner ready for distributed execution.

        Runner Creation Process:
            1. **IcoRunner Instantiation**: Create runner with stored configuration
            2. **Channel Assignment**: Assign inverted channel for worker communication
            3. **Flow Factory**: Pass flow_factory for computation logic creation
            4. **Name Assignment**: Set runner name for runtime tree identification

        Runner Configuration:
            Created runner has complete configuration:
            • **Communication**: Inverted channel for bidirectional worker communication
            • **Computation**: Flow factory for creating actual computation logic
            • **Runtime Integration**: Name and configuration for runtime tree participation
            • **Type Safety**: Generic[I, O] parameters preserved from factory

        Process Isolation:
            Runner created in isolated runner process:
            • **Independent Runtime Tree**: Runner becomes root of runner runtime hierarchy
            • **Separate Memory Space**: Complete isolation from worker process
            • **Communication Bridge**: Only connection to worker through channel
        """
        return IcoRunner[I, O](
            channel=self.runner_channel,
            flow_factory=self.flow_factory,
            name=self.name,
            probe=self.probe,
            flow_key=self.flow_key,
        )
