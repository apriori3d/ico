from dataclasses import dataclass
from typing import final

from apriori.ico.core.runtime.command import IcoRuntimeCommand


@final
@dataclass(slots=True, frozen=True)
class IcoRuntimeMetaRequest(IcoRuntimeCommand):
    """Request to obtain the flow metadata from a remote ICO node."""

    pass
