from typing import Generic

from apriori.ico.core.operator import I, IcoOperator, O


class IcoFlowOperator(Generic[I, O], IcoOperator[I, O]):
    pass
