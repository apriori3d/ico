class IcoRuntimeError(Exception):
    """Base class for actual runtime errors (state corruption, IPC failure, etc.)."""

    pass


class IcoRuntimeSignal(Exception):
    """
    Base class for runtime control signals.

    These are not errors — they propagate upward through contours
    to control execution flow (e.g. graceful stop, teardown).
    """

    pass


class IcoStopExecutionSignal(IcoRuntimeSignal):
    """Signal to stop the current runtime flow or contour."""

    pass


class IcoResetExecutionSignal(IcoRuntimeSignal):
    """Signal to reset the runtime flow without teardown."""

    pass


class IcoPauseExecutionSignal(IcoRuntimeSignal):
    """Signal to pause the runtime flow without teardown."""

    pass


class DeactivateExecutionSignal(IcoRuntimeSignal):
    """Signal to deactivate and release runtime resources."""

    pass
