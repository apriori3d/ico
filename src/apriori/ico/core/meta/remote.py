from dataclasses import dataclass

from typing_extensions import Protocol, final

from apriori.ico.core.meta.flow_meta import IcoFlowMeta
from apriori.ico.core.runtime.command import IcoRuntimeCommand


class HasRemoteMeta(Protocol):
    """Protocol for ICO nodes that can provide remote metadata, like agents."""

    def provide_remote_meta(self) -> IcoFlowMeta: ...


@final
@dataclass(slots=True, frozen=True)
class IcoFlowMetaRequest(IcoRuntimeCommand):
    """Request to obtain the flow metadata from a remote ICO node."""

    pass
