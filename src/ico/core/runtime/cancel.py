"""Cooperative cancellation of a run.

A run is cancelled through a token, from any thread (a request handler, a
signal handler). The loops that pull items (streams, sources, sinks, the
scheduler of an async stream, epochs and processes) check it between items
and raise ``IcoCancelled``: the item in progress finishes, no new one
starts. The runtime makes its token current while it runs (a context
variable, so the threads and tasks of async streams see it too); a flow
called outside a runtime has no token and checks nothing.

The runtime also sends an ``IcoCancelCommand`` to the runners of its
workers, through their control lanes (see ``IcoChannel.send_control``):
a runner cancels its own token, so the item it runs stops between its
inner items too, and the worker raises ``IcoCancelled`` for that item.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import ClassVar, final

from ico.core.runtime.command import IcoRuntimeCommand


class IcoCancelled(RuntimeError):
    """A run was cancelled; raised between items by the loops of the flow."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"the run was cancelled: {reason}")
        self.reason = reason


@final
class IcoCancelToken:
    """A thread-safe request to stop a run, with its reason."""

    __slots__ = ("_event", "_lock", "_listeners", "reason")

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._listeners: list[Callable[[str], None]] = []
        self.reason: str | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        """Request the run to stop; a second call keeps the first reason.
        Listeners are called once, in the calling thread."""
        with self._lock:
            if self._event.is_set():
                return
            self.reason = reason
            self._event.set()
            listeners = list(self._listeners)
        for listener in listeners:
            listener(reason)

    def add_listener(self, listener: Callable[[str], None]) -> None:
        """Call ``listener(reason)`` when the token is cancelled (at once if
        it already is)."""
        with self._lock:
            if not self._event.is_set():
                self._listeners.append(listener)
                return
        listener(self.reason or "cancelled")

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


_current: ContextVar[IcoCancelToken | None] = ContextVar(
    "ico_cancel_token", default=None
)


@final
@dataclass(slots=True, frozen=True)
class IcoCancelCommand(IcoRuntimeCommand):
    """Cancel the runs of the runners below (sent by ``IcoRuntime`` when its
    token is cancelled)."""

    acknowledged: ClassVar[bool] = False

    reason: str


def cancelled_reason() -> str | None:
    """The reason the current run was cancelled, or ``None``."""
    token = _current.get()
    return token.reason if token is not None and token.cancelled else None


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
