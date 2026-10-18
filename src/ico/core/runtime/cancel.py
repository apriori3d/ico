"""Cooperative cancellation of a run.

A run is cancelled through a token, from any thread (a request handler, a
signal handler). The loops that pull items (streams, sources, sinks, the
scheduler of an async stream, epochs and processes) check it between items
and raise ``IcoCancelled``: the item in progress finishes, no new one
starts. The runtime makes its token current while it runs (a context
variable, so the threads and tasks of async streams see it too); a flow
called outside a runtime has no token and checks nothing.

Cancelling does not use the command bus: the bus reads the workers'
channels, which the workers' portals read during a run.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import final


class IcoCancelled(RuntimeError):
    """A run was cancelled; raised between items by the loops of the flow."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"the run was cancelled: {reason}")
        self.reason = reason


@final
class IcoCancelToken:
    """A thread-safe request to stop a run, with its reason."""

    __slots__ = ("_event", "reason")

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason: str | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        """Request the run to stop; a second call keeps the first reason."""
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


_current: ContextVar[IcoCancelToken | None] = ContextVar(
    "ico_cancel_token", default=None
)


def check_cancelled() -> None:
    """Raise ``IcoCancelled`` if the current run was cancelled."""
    token = _current.get()
    if token is not None and token.cancelled:
        raise IcoCancelled(token.reason or "cancelled")


@contextmanager
def cancellable(token: IcoCancelToken) -> Iterator[None]:
    """Make ``token`` the current one while the block runs."""
    reset = _current.set(token)
    try:
        yield
    finally:
        _current.reset(reset)
