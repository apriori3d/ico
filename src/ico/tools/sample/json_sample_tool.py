"""Runtime tool that passes samples (``IcoRuntime.sample``) to a callback
as JSON."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict, final

from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.runtime.sample import IcoSampleEvent
from ico.core.runtime.tool import IcoRuntimeTool


class SampleSnapshot(TypedDict):
    """One sample: ``node`` is the plan key of the operator, ``index`` the
    number of the sample in its process, ``type`` and ``preview`` describe
    the result."""

    node: str
    index: int
    type: str
    preview: str
    recent: bool
    """A recent result (``IcoRuntime.keep_recent``), not one asked for."""


@final
class JsonSampleTool(IcoRuntimeTool):
    """Pass every sample of the runtime to ``on_sample``::

        tool = JsonSampleTool(print)
        runtime = IcoRuntime(flow, tools=[tool]).activate()
        runtime.sample("0.1", count=3)  # from any thread, also during run()
        runtime.run().deactivate()

    ``on_sample`` may be called from the threads of the flow.
    """

    __slots__ = ("on_sample",)

    def __init__(self, on_sample: Callable[[SampleSnapshot], None]) -> None:
        super().__init__()
        self.on_sample = on_sample

    def on_forward_event(self, event: IcoRuntimeEvent) -> None:
        super().on_forward_event(event)
        if isinstance(event, IcoSampleEvent) and event.node is not None:
            self.on_sample(
                {
                    "node": event.node,
                    "index": event.index,
                    "type": event.type,
                    "preview": event.preview,
                    "recent": event.recent,
                }
            )
