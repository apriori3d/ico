from __future__ import annotations

import abc
from typing import (
    Generic,
    Literal,
    Protocol,
    overload,
    runtime_checkable,
)

from ico.core.chain import O3, IcoChain
from ico.core.operator import O2, I, IContra, IcoOperator, IcoOperatorProtocol, O
from ico.core.pipeline import IcoPipeline

SKMode = Literal["fit", "predict"]


@runtime_checkable
class SKOperatorProtocol(IcoOperatorProtocol[IContra, O], Protocol[IContra, O]):
    mode: SKMode

    def fit_mode(self) -> None: ...

    def predict_mode(self) -> None: ...


class SKOperator(Generic[I, O], IcoOperator[I, O], SKOperatorProtocol[I, O], abc.ABC):
    mode: SKMode = "fit"

    def __init__(
        self,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(self._estimator_fn, name=name)

    @abc.abstractmethod
    def _estimator_fn(self, input: I) -> O: ...

    def fit_mode(self) -> None:
        self.mode = "fit"

        for child in self.children:
            if isinstance(child, SKOperatorProtocol):
                child.fit_mode()

    def predict_mode(self) -> None:
        self.mode = "predict"

        for child in self.children:
            if isinstance(child, SKOperatorProtocol):
                child.predict_mode()

    @overload
    def __or__(self, other: SKOperatorProtocol[O, O2]) -> SKOperatorProtocol[I, O2]: ...

    @overload
    def __or__(
        self, other: IcoOperatorProtocol[O, O2]
    ) -> SKOperatorProtocol[I, O2]: ...

    def __or__(self, other: IcoOperatorProtocol[O, O2]) -> SKOperatorProtocol[I, O2]:
        return SKChain(self, other)


class SKChain(Generic[I, O, O2], IcoChain[I, O, O2], SKOperatorProtocol[I, O2]):
    mode: SKMode = "fit"

    def fit_mode(self) -> None:
        self.mode = "fit"

        if isinstance(self._left, SKOperatorProtocol):
            self._left.fit_mode()
        if isinstance(self._right, SKOperatorProtocol):
            self._right.fit_mode()

    def predict_mode(self) -> None:
        self.mode = "predict"

        if isinstance(self._left, SKOperatorProtocol):
            self._left.predict_mode()
        if isinstance(self._right, SKOperatorProtocol):
            self._right.predict_mode()

    @overload
    def __or__(self, other: SKOperatorProtocol[O, O2]) -> SKOperatorProtocol[I, O2]: ...

    @overload
    def __or__(
        self, other: IcoOperatorProtocol[O, O2]
    ) -> SKOperatorProtocol[I, O2]: ...

    def __or__(self, other: IcoOperatorProtocol[O2, O3]) -> IcoOperatorProtocol[I, O3]:
        return SKChain(self, other)


class SKPipeline(IcoPipeline[I], Generic[I], SKOperatorProtocol[I, I]):
    mode: SKMode = "fit"

    def fit_mode(self) -> None:
        self.mode = "fit"

        for op in self.body:
            if isinstance(op, SKOperatorProtocol):
                op.fit_mode()

    def predict_mode(self) -> None:
        self.mode = "predict"

        for op in self.body:
            if isinstance(op, SKOperatorProtocol):
                op.predict_mode()
