from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from apriori.ico.core.operator import IcoOperator


@runtime_checkable
class HasSubflowFactory(Protocol):
    subflow_factory: Callable[[], IcoOperator[Any, Any]]
