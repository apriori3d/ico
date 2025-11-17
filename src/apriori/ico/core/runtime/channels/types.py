from __future__ import annotations

from typing import Protocol

from apriori.ico.core.runtime.types import (
    ConnectedToIcoRuntime,
    IcoRuntimeFlowProtocol,
    IcoRuntimeOperatorProtocol,
)
from apriori.ico.core.types import I, IcoOperatorProtocol, O


class IcoSendEndpointProtocol(
    IcoRuntimeFlowProtocol,
    IcoOperatorProtocol[I, None],
    Protocol[I],
):
    """Operator responsible for pushing data and runtime events downstream."""

    ...


class IcoReceiveEndpointProtocol(
    ConnectedToIcoRuntime,
    IcoRuntimeFlowProtocol,
    IcoOperatorProtocol[None, O],
    Protocol[O],
):
    """Operator responsible for pulling data and runtime events."""

    ...


class IcoRuntimeChannelProtocol(IcoRuntimeOperatorProtocol, Protocol[I, O]):
    send: IcoSendEndpointProtocol[I]
    receive: IcoReceiveEndpointProtocol[O]
