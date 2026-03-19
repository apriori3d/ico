from typing import Generic

from ico.core.operator import I, IcoOperator


class IcoIdentity(Generic[I], IcoOperator[I, I]):
    """Identity operator that passes input through unchanged.

    IcoIdentity represents a no-op transformation in ICO pipelines, useful
    for structural purposes, debugging, or as placeholder in compositions.
    It simply returns whatever input it receives without modification.

    This is primarily a logical node designed for inheritance by more
    specialized operators that need identity behavior as their base.

    Generic Parameters:
        I: Input/Output type - the type of data that passes through unchanged.

    ICO signature:
        I → I

    Example:
        >>> identity = IcoIdentity[int]()
        >>> result = identity(42)
        >>> assert result == 42

        >>> # Used as base class for monitoring operators
        >>> # IcoMonitor inherits from IcoIdentity + IcoRuntimeNode
        >>> # This allows monitors to pass data through while tracking runtime state
        >>> monitor = IcoMonitor[str](name="data_monitor")
        >>> result = monitor("hello")  # Passes through + updates runtime state
        >>> assert result == "hello"

    Note:
        Identity operators are particularly useful for:
        - Base class for specialized operators (IcoMonitor, IcoProgress)
        - Debugging pipeline stages
        - Structural composition patterns
        - No-op placeholders during development
    """

    def __init__(self, *, name: str | None = None) -> None:
        """Initialize an identity operator.

        Args:
            name: Optional name for this identity operator (useful for debugging).

        Note:
            Identity operators require no additional configuration since they
            simply pass through their input unchanged.
        """
        super().__init__(fn=self._identity_fn, name=name)

    def _identity_fn(self, item: I) -> I:
        """Internal implementation that returns the input unchanged.

        Args:
            item: Input value of type I.

        Returns:
            The same input value, unmodified.

        Note:
            This is the function used by __call__. It's the simplest possible
            transformation: f(x) = x.
        """
        return item
