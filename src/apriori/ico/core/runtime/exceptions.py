class IcoRuntimeError(Exception):
    """Base class for runtime execution infrastructure errors.

    IcoRuntimeError represents failures in the runtime execution infrastructure
    that are distinct from computation logic errors. These errors relate to
    execution management, resource allocation, and runtime tree coordination.

    Runtime Error Categories:
        • State corruption: Invalid state transitions or corrupted state models
        • Resource failures: Memory allocation, handle exhaustion, connection issues
        • IPC failures: Inter-process communication errors in distributed execution
        • Tree coordination: Runtime tree structure or synchronization problems
        • Agent failures: Remote execution node failures or timeouts

    Error Handling Integration:
        • Runtime errors automatically trigger IcoFaultEvent generation
        • Fault events bubble up the runtime tree for centralized handling
        • Parent nodes can implement recovery strategies for child failures
        • Error information preserved for debugging and analysis

    Usage:
        Raise IcoRuntimeError subclasses when runtime infrastructure fails.
        Computation logic errors should use regular Python exceptions.
        Runtime errors integrate with the fault event system for recovery.

    """

    pass
