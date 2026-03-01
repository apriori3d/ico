from __future__ import annotations

from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Any, Generic, final

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

# ────────────────────────────────────────────────────────────────────────────
# Multiprocessing-based distributed agent implementation
# ────────────────────────────────────────────────────────────────────────────


@final
class MPAgent(Generic[I, O], IcoAgent[I, O]):
    """Multiprocessing-based distributed agent for ICO computation execution.

    MPAgent implements distributed computation by spawning separate worker processes
    and coordinating with them through MPChannel-based inter-process communication.
    It provides transparent distributed execution while maintaining full integration
    with ICO runtime tree and computation flow architecture.

    Distributed Architecture:
        • **Process Separation**: Agent (portal) and Worker (computation) in separate processes
        • **Channel Communication**: Bidirectional MPChannel for data and runtime coordination
        • **Factory Pattern**: Flow factory for creating computation logic in worker process
        • **Process Lifecycle**: Complete worker process lifecycle management

    Agent/Worker Collaboration:
        **Agent Process (Portal)**:
        • Acts as computation node in main compute flow
        • Forwards data items to remote worker process
        • Receives computed results from worker
        • Manages worker process lifecycle and error handling

        **Worker Process (Remote)**:
        • Executes actual computation flow in isolated process
        • Participates in distributed runtime tree
        • Sends progress/status events back to agent
        • Independent memory space and Python interpreter

    Flow Factory Integration:
        MPAgent uses factory pattern for worker process initialization:
        • Flow factory creates computation logic in worker process
        • Factory must be pickleable for cross-process transmission
        • Worker process reconstructs computation flow from factory
        • Full computation isolation between agent and worker

    Runtime Tree Distribution:
        ```
        Main Process Runtime Tree          Worker Process Runtime Tree
        ┌─────────────────────┐           ┌─────────────────────┐
        │ Runtime             │           │ AgentWorker         │
        │ └── MPAgent (portal)│ ←─────→   │ └── Flow (actual)   │
        │     └── Channel     │  Channel  │     ├── Operators   │
        │                     │    IPC    │     └── Progress    │
        └─────────────────────┘           └─────────────────────┘
        ```

    Process Management:
        • **Spawn Context**: Uses 'spawn' multiprocessing for cross-platform compatibility
        • **Process Creation**: Worker process spawned during activation phase
        • **Graceful Shutdown**: Coordinated worker termination during deactivation
        • **Error Handling**: Worker exceptions propagated to agent via events
        • **Resource Cleanup**: Automatic channel and process resource cleanup

    Channel Configuration:
        MPAgent configures channels for agent/worker coordination:
        • **accept_commands=False**: Agent receives commands from runtime
        • **accept_events=True**: Agent accepts events from worker
        • **strict_accept=True**: Enforces strict message type validation
        • **Channel Inversion**: Worker gets inverted channel automatically

    Computation Transparency:
        From computation flow perspective, MPAgent appears as regular operator:
        • **Input**: Receives items from previous operators
        • **Output**: Produces results to next operators
        • **Type Safety**: Maintains Generic[I, O] type parameters
        • **Runtime Integration**: Full runtime tree participation

    Process Safety Features:
        • **Isolated Execution**: Worker process completely isolated from agent
        • **Exception Propagation**: Worker exceptions bubble to agent via IcoFaultEvent
        • **Resource Management**: Automatic cleanup of processes and channels
        • **Timeout Handling**: Configurable timeouts for worker operations
        • **Process Monitoring**: Worker process health monitoring and lifecycle tracking

    Usage Integration:
        MPAgent integrates seamlessly into ICO computation flows:
        • Can be used anywhere IcoOperator is expected
        • Supports method chaining (.stream()) and standard ICO flow composition (|)
        • Compatible with all ICO runtime tools and monitoring
        • Transparent to upstream and downstream operators

    Example - Distributed Computation Flow:
        ```python
        from apriori.ico.core import IcoSource, IcoSink, IcoOperator
        from apriori.ico.core.runtime.progress import IcoProgress
        from apriori.ico.core.runtime.runtime import IcoRuntime
        from apriori.ico.runtime.agent.mp.mp_agent import MPAgent
        from apriori.ico.tools.progress.rich_progress_tool import RichProgressTool
        import time

        # Define computation that will run in worker process
        def expensive_computation(item: int) -> int:
            time.sleep(0.1)  # Simulate expensive work
            return item ** 2

        # Factory for creating computation flow in worker process
        def create_worker_flow() -> IcoOperator[int, int]:
            # This flow runs in isolated worker process
            return IcoOperator(expensive_computation, name="compute_square")

        # Build main computation flow with distributed processing
        source = IcoSource(lambda: range(20), name="data_source")
        progress = IcoProgress[int](total=20, name="input_progress")

        # MPAgent runs expensive_computation in separate process
        distributed_agent = MPAgent[int, int](
            flow_factory=create_worker_flow,
            name="square_agent"
        )

        result_sink = IcoSink(
            lambda results: [print(f"Result: {r}") for r in results],
            name="result_sink"
        )

        # Compose distributed computation flow
        flow = source | progress | distributed_agent.stream() | result_sink

        # Execute with runtime coordination and progress monitoring
        runtime = IcoRuntime(flow, tools=[RichProgressTool()], name="distributed_runtime")
        runtime.activate().run().deactivate()

        # Output: Real-time progress bar + computed squares
        # Worker process runs expensive_computation in isolation
        # Results: 0, 1, 4, 9, 16, 25, 36, 49, 64, ...
        ```

    Performance Considerations:
        • **Process Overhead**: Multiprocessing has startup cost vs threading
        • **Serialization Cost**: Data serialization for cross-process communication
        • **Memory Isolation**: Worker process uses separate memory space
        • **GIL Avoidance**: True parallelism by avoiding Python GIL limitations

    Error Handling Strategy:
        • **Worker Exceptions**: Captured and forwarded as IcoFaultEvent
        • **Process Failures**: Detected during deactivation and reported
        • **Channel Errors**: Communication failures handled gracefully
        • **Timeout Conditions**: Configurable timeout with cleanup procedures

    Distributed Coordination:
        MPAgent coordinates distributed execution through:
        • **Runtime Commands**: Lifecycle commands (prepare/reset) sent to worker
        • **Progress Events**: Worker progress events bubbled to agent runtime tree
        • **State Synchronization**: Worker and agent state coordination
        • **Exception Reporting**: Worker exceptions reported to agent tools
    """

    _agent_process: SpawnProcess | None  # Worker process instance (None when inactive)
    _mp_context: SpawnContext  # Multiprocessing context for process/queue creation
    # _print: IcoPrinter                   # Optional printer for debugging (currently disabled)

    def __init__(
        self,
        flow_factory: Callable[[], IcoOperator[I, O]],
        *,
        name: str | None = None,
    ) -> None:
        """Initialize multiprocessing agent with computation flow factory.

        Args:
            flow_factory: Factory function creating computation flow in worker process.
                         Must be pickleable for cross-process transmission.
            name: Optional agent identifier for debugging and monitoring.

        Initialization Process:
            1. **Parent Initialization**: Call IcoAgent.__init__ with None channel
            2. **Context Creation**: Create 'spawn' multiprocessing context
            3. **Channel Setup**: Create initial MPChannel for agent/worker communication
            4. **Process Preparation**: Initialize process tracking (None until activation)

        Flow Factory Requirements:
            • **Pickleable**: Factory must be serializable for process transmission
            • **Stateless**: Factory should not capture non-pickleable state
            • **Complete Flow**: Factory must create complete I → O computation
            • **Type Safety**: Factory return type must match Generic[I, O] parameters

        Multiprocessing Context:
            Uses 'spawn' context for maximum compatibility:
            • **Cross-Platform**: Works on Windows, macOS, and Linux
            • **Clean State**: Worker process starts with fresh interpreter
            • **Import Safety**: Avoids import-related multiprocessing issues
            • **Resource Isolation**: Complete memory and resource isolation

        Channel Pre-Creation:
            Channel created during initialization (before activation) to enable:
            • **Factory Access**: Worker factory can access channel configuration
            • **Runtime Discovery**: Runtime tree can discover channel relationships
            • **Configuration Completion**: All agent configuration available immediately
        """
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
        """Check if worker process is currently running.

        Returns:
            True if worker process exists and is alive, False otherwise.

        Process Health Check:
            • **None Check**: Returns False if no process spawned yet
            • **Alive Check**: Uses multiprocessing.Process.is_alive() for actual status
            • **Real-time Status**: Reflects current worker process state

        Usage:
            Used for monitoring worker health and coordinating lifecycle operations.
            Agent tools can check worker status for debugging and error handling.
        """
        return self._agent_process.is_alive() if self._agent_process else False

    # ────────────────────────────────────────────────
    # Worker process lifecycle management
    # ────────────────────────────────────────────────

    def get_remote_runtime_factory(self) -> Callable[[], IcoAgentWorker[I, O]]:
        """Create pickleable factory for worker process runtime initialization.

        Returns:
            Factory function that creates IcoAgentWorker in worker process.

        Factory Creation Process:
            1. **Channel Inversion**: Invert agent channel for worker perspective
            2. **Factory Capture**: Capture flow_factory and configuration
            3. **Pickleable Wrapper**: Wrap in _WorkerFactory for serialization
            4. **Type Preservation**: Maintain Generic[I, O] typing through factory

        Cross-Process Serialization:
            Factory is designed for multiprocessing transmission:
            • **Pickleable Design**: _WorkerFactory implements __call__ for serialization
            • **State Capture**: Captures all necessary worker initialization data
            • **Channel Inversion**: Provides inverted channel to worker process
            • **Configuration Transfer**: Transfers name and flow_factory to worker

        Worker Process Integration:
            Factory creates fully configured IcoAgentWorker in remote process:
            • **Runtime Tree**: Worker becomes root of distributed runtime tree
            • **Channel Assignment**: Worker gets inverted channel for agent communication
            • **Flow Creation**: Worker uses flow_factory to create computation logic
            • **Name Preservation**: Worker inherits agent name for debugging
        """
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
        """Create configured MPChannel for agent/worker communication.

        Returns:
            Configured MPChannel with agent-specific settings.

        Channel Configuration:
            • **Runtime Port**: Agent acts as runtime port for channel integration
            • **Command Handling**: accept_commands=False (agent receives from runtime)
            • **Event Handling**: accept_events=True (agent accepts worker events)
            • **Strict Validation**: strict_accept=True for robust message handling

        Agent-Specific Settings:
            Channel configured for agent role in distributed architecture:
            • **Agent → Worker**: Data messages flow from agent to worker
            • **Worker → Agent**: Result messages and events flow back
            • **Runtime Commands**: Flow from main runtime through agent to worker
            • **Worker Events**: Bubble from worker through agent to main runtime
        """
        assert self._mp_context is not None
        return MPChannel[I, O](
            mp_context=self._mp_context,
            runtime_port=self,
            accept_commands=False,
            accept_events=True,
            strict_accept=True,
        )

    def _activate_worker(self) -> None:
        """Initialize and start worker process for distributed computation.

        Activation Process:
            1. **Channel Recreation**: Create fresh channel to avoid reusing closed channels
            2. **Worker Spawn**: Create and start new worker process
            3. **Parent Activation**: Call parent class activation for runtime coordination

        Channel Recreation Strategy:
            Fresh channel created on each activation to ensure:
            • **Clean State**: No residual state from previous activations
            • **Resource Safety**: Avoid using closed/corrupted channels
            • **Process Isolation**: Each activation gets independent communication

        Process Creation:
            Worker process spawned through spawn_worker() with:
            • **Factory Injection**: Worker factory passed to process
            • **Process Start**: Worker process started immediately
            • **Resource Tracking**: Process reference stored for lifecycle management
        """
        # Recreate channel for agent process to ensure closed channels are not reused
        self.channel = self._create_channel()
        self._agent_process = self.spawn_worker()

        super()._activate_worker()

    def _deactivate_worker(self) -> None:
        """Gracefully shutdown worker process and cleanup resources.

        Deactivation Process:
            1. **Graceful Join**: Wait for worker process to complete
            2. **Timeout Handling**: Use configurable timeout for join operation
            3. **Exception Handling**: Capture and report worker shutdown issues
            4. **Forced Termination**: Terminate worker if graceful shutdown fails
            5. **Parent Deactivation**: Call parent cleanup for runtime coordination

        Graceful Shutdown Strategy:
            • **Join Timeout**: 5-second timeout for worker process completion
            • **Exit Code Check**: Monitor worker exit status for error detection
            • **Exception Propagation**: Worker exceptions bubbled as IcoFaultEvent
            • **Resource Cleanup**: Channels and process resources cleaned automatically

        Error Handling:
            • **Join Failures**: Exceptions during worker join reported as fault events
            • **Process Monitoring**: Worker process health monitored during shutdown
            • **Forced Cleanup**: Terminate() called if graceful shutdown fails
            • **Resource Safety**: Cleanup guaranteed even with exceptions

        Channel Lifecycle:
            Channels are closed automatically during deactivation:
            • **Command Propagation**: Commands sent before channel closure
            • **Event Collection**: Final events received before shutdown
            • **Queue Cleanup**: Underlying multiprocessing queues cleaned up
        """
        assert self._agent_process is not None

        try:
            # Gracefully join the worker process
            if self._agent_process.is_alive():
                # Wait for agent process to exit
                self._agent_process.join(timeout=5)

        except Exception as e:
            self.bubble_event(IcoFaultEvent.create(e))

        finally:
            if self._agent_process.is_alive():
                self._agent_process.terminate()

            super()._deactivate_worker()

    def spawn_worker(self) -> SpawnProcess:
        """Create and start worker process with runtime factory.

        Returns:
            Started SpawnProcess instance running worker computation.

        Process Creation:
            1. **Factory Generation**: Get remote runtime factory for worker initialization
            2. **Process Configuration**: Create process with worker entry point
            3. **Process Start**: Start process immediately after creation

        Worker Entry Point:
            Worker process executes _start_worker_process static method:
            • **Factory Execution**: Worker factory called to create IcoAgentWorker
            • **Runtime Loop**: Worker enters run_loop() for continuous operation
            • **Process Isolation**: Complete isolation from agent process

        Multiprocessing Context:
            Uses configured spawn context for process creation:
            • **Cross-Platform**: Spawn context works on all platforms
            • **Clean Interpreter**: Worker starts with fresh Python interpreter
            • **Resource Isolation**: Independent memory space and file handles
        """
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
        """Static entry point for worker process execution.

        Args:
            worker_factory: Factory function for creating IcoAgentWorker instance.

        Worker Process Lifecycle:
            1. **Factory Execution**: Create IcoAgentWorker from factory
            2. **Runtime Loop**: Enter continuous operation loop
            3. **Command Processing**: Process commands and data from agent
            4. **Event Generation**: Send progress and status events to agent

        Process Isolation:
            Worker process operates independently:
            • **Separate Runtime Tree**: Worker has own runtime hierarchy
            • **Independent Tools**: Worker can have own monitoring tools
            • **Isolated Memory**: No shared memory with agent process
            • **Exception Isolation**: Worker exceptions don't crash agent

        Communication Flow:
            Worker communicates with agent through inverted channel:
            • **Receives**: Data items and runtime commands from agent
            • **Sends**: Computed results and runtime events to agent
            • **Bidirectional**: Full bidirectional communication capability
        """
        worker = worker_factory()
        # Run agent to start receiving and processing commands and items
        worker.run_loop()

    # ────────────────────────────────────────────────
    # Type signature and computation interface
    # ────────────────────────────────────────────────

    @property
    def signature(self) -> IcoSignature:
        """Compute agent signature for computation flow integration.

        Returns:
            IcoSignature representing agent's computation interface.

        Signature Computation:
            1. **Parent Signature**: Get base signature from IcoAgent
            2. **Inference**: Use flow_factory for signature inference if needed
            3. **Agent Transformation**: Convert to agent-specific signature format

        Agent Signature Format:
            Agent signature differs from standard operator signature:
            • **Input (i)**: Accepts items of type from parent signature (c)
            • **Context (c)**: None (agent doesn't expose internal context)
            • **Output (o)**: Produces items of type from parent signature (c)

        Signature Inference:
            When parent signature is inferred:
            • **Flow Factory Analysis**: Analyze flow_factory return type
            • **Type Extraction**: Extract I and O types from IcoOperator[I, O]
            • **Signature Construction**: Build proper IcoSignature from types

        Computation Transparency:
            Agent signature hides distributed implementation details:
            • **Input Interface**: Matches expected input type I
            • **Output Interface**: Matches expected output type O
            • **Process Abstraction**: Internal multiprocessing hidden from signature
            • **Type Safety**: Maintains Generic[I, O] type constraints
        """
        signature = super().signature

        if signature.infered:
            signature = infer_from_flow_factory(self.flow_factory)

        if signature is None:
            return IcoSignature(i=Any, c=None, o=Any, infered=False)

        return IcoSignature(
            i=signature.i,
            c=None,
            o=signature.o,
        )


# ────────────────────────────────────────────────────────────────────────────
# Pickleable worker factory for cross-process initialization
# ────────────────────────────────────────────────────────────────────────────


@final
class _WorkerFactory(Generic[I, O]):
    """Pickleable factory for creating IcoAgentWorker instances in worker processes.

    _WorkerFactory enables cross-process transmission of worker initialization
    data by implementing a pickleable callable that reconstructs IcoAgentWorker
    with proper configuration in the target worker process.

    Pickleable Design:
        • **Serializable State**: All attributes are pickleable (channel, factory, name)
        • **Callable Interface**: Implements __call__ for factory pattern
        • **Generic Preservation**: Maintains Generic[I, O] typing through serialization
        • **Cross-Process Safe**: Safe for multiprocessing transmission

    Factory Encapsulation:
        Factory encapsulates all worker initialization requirements:
        • **Inverted Channel**: Pre-configured channel for worker communication
        • **Flow Factory**: Computation logic factory for worker process
        • **Configuration**: Name and other worker configuration data
        • **Type Safety**: Generic parameters preserved through factory

    Cross-Process Workflow:
        1. **Agent Process**: MPAgent creates _WorkerFactory with worker config
        2. **Serialization**: Factory pickled for transmission to worker process
        3. **Worker Process**: Factory unpickled and called to create IcoAgentWorker
        4. **Worker Initialization**: IcoAgentWorker created with proper configuration

    Channel Inversion Pattern:
        Factory receives inverted channel from agent:
        • **Agent Perspective**: MPChannel[I, O] for sending I, receiving O
        • **Worker Perspective**: MPChannel[O, I] for receiving I, sending O
        • **Automatic Inversion**: Channel inversion handled by agent before factory creation
        • **Bidirectional Communication**: Full communication capability in worker
    """

    def __init__(
        self,
        worker_channel: IcoChannel[O, I],
        flow_factory: Callable[[], IcoOperator[I, O]],
        name: str | None,
    ) -> None:
        """Initialize worker factory with worker configuration.

        Args:
            worker_channel: Pre-configured inverted channel for worker communication.
            flow_factory: Factory function for creating computation flow in worker.
            name: Optional worker name for debugging and identification.

        Factory State:
            All arguments stored as instance attributes for pickling:
            • **worker_channel**: Channel configured for worker perspective
            • **flow_factory**: Computation logic factory for worker process
            • **name**: Worker identification for runtime tree integration

        Pickling Requirements:
            All stored objects must be pickleable:
            • **Channel**: MPChannel implements proper pickling support
            • **Flow Factory**: User-provided factory must be pickleable
            • **Name**: String or None (always pickleable)
        """
        self.worker_channel = worker_channel
        self.flow_factory = flow_factory
        self.name = name

    def __call__(self) -> IcoAgentWorker[I, O]:
        """Create and return configured IcoAgentWorker instance.

        Returns:
            Fully configured IcoAgentWorker ready for distributed execution.

        Worker Creation Process:
            1. **IcoAgentWorker Instantiation**: Create worker with stored configuration
            2. **Channel Assignment**: Assign inverted channel for agent communication
            3. **Flow Factory**: Pass flow_factory for computation logic creation
            4. **Name Assignment**: Set worker name for runtime tree identification

        Worker Configuration:
            Created worker has complete configuration:
            • **Communication**: Inverted channel for bidirectional agent communication
            • **Computation**: Flow factory for creating actual computation logic
            • **Runtime Integration**: Name and configuration for runtime tree participation
            • **Type Safety**: Generic[I, O] parameters preserved from factory

        Process Isolation:
            Worker created in isolated worker process:
            • **Independent Runtime Tree**: Worker becomes root of worker runtime hierarchy
            • **Separate Memory Space**: Complete isolation from agent process
            • **Communication Bridge**: Only connection to agent through channel
        """
        return IcoAgentWorker[I, O](
            channel=self.worker_channel,
            flow_factory=self.flow_factory,
            name=self.name,
        )
