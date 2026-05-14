from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import (
    Generic,
    Literal,
    Protocol,
    overload,
    runtime_checkable,
)

from ico.core.chain import O3, IcoChain
from ico.core.context_operator import (
    C,
    # CContra,
    IcoContextOperator,
    IcoContextOperatorProtocol,
    OCovariant,
)
from ico.core.node import IcoNodeProtocol
from ico.core.operator import O2, I, IcoOperator, IcoOperatorProtocol, IInv, O
from ico.core.pipeline import IcoPipeline

SKMode = Literal["fit", "predict"]


@runtime_checkable
class SKProtocol(Protocol):
    mode: SKMode

    def fit_mode(self) -> None: ...

    def predict_mode(self) -> None: ...


class SKMixin(SKProtocol):
    mode: SKMode = "fit"
    fitted: bool = False

    def fit_mode(self) -> None:
        self.mode = "fit"

        if isinstance(self, IcoNodeProtocol):
            for child in self.children:
                if isinstance(child, SKOperatorProtocol | SKContextOperatorProtocol):
                    child.fit_mode()

    def predict_mode(self) -> None:
        self.mode = "predict"

        if isinstance(self, IcoNodeProtocol):
            for child in self.children:
                if isinstance(child, SKOperatorProtocol | SKContextOperatorProtocol):
                    child.predict_mode()


@runtime_checkable
class SKOperatorProtocol(
    SKProtocol,
    IcoOperatorProtocol[I, O],
    Protocol[I, O],
):
    @overload
    def __or__(self, other: SKOperatorProtocol[O, O2]) -> SKOperatorProtocol[I, O2]: ...

    @overload
    def __or__(
        self, other: IcoOperatorProtocol[O, O2]
    ) -> SKOperatorProtocol[I, O2]: ...

    def __or__(
        self, other: IcoOperatorProtocol[O, O2]
    ) -> SKOperatorProtocol[I, O2]: ...

    @overload
    def __ior__(
        self, other: SKOperatorProtocol[O, O2]
    ) -> SKOperatorProtocol[I, O2]: ...

    @overload
    def __ior__(
        self, other: IcoOperatorProtocol[O, O2]
    ) -> SKOperatorProtocol[I, O2]: ...

    def __ior__(
        self, other: IcoOperatorProtocol[O, O2]
    ) -> SKOperatorProtocol[I, O2]: ...


@runtime_checkable
class SKContextOperatorProtocol(
    SKProtocol,
    IcoContextOperatorProtocol[I, C, OCovariant],
    Protocol[I, C, OCovariant],
):
    pass


class SKOperator(
    Generic[I, O],
    IcoOperator[I, O],
    SKOperatorProtocol[I, O],
    SKMixin,
    abc.ABC,
):
    def __init__(
        self,
        *,
        name: str | None = None,
        children: Sequence[IcoNodeProtocol] | None = None,
    ) -> None:
        super().__init__(
            self._estimator_fn,
            name=name,
            children=children,
        )

    @abc.abstractmethod
    def _estimator_fn(self, input: I) -> O: ...

    @overload
    def __or__(self, other: SKOperatorProtocol[O, O2]) -> SKOperatorProtocol[I, O2]: ...

    @overload
    def __or__(
        self, other: IcoOperatorProtocol[O, O2]
    ) -> SKOperatorProtocol[I, O2]: ...

    def __or__(self, other: IcoOperatorProtocol[O, O2]) -> SKOperatorProtocol[I, O2]:
        return SKChain(self, other)

    @overload
    def __ior__(
        self, other: SKOperatorProtocol[O, O2]
    ) -> SKOperatorProtocol[I, O2]: ...

    @overload
    def __ior__(
        self, other: IcoOperatorProtocol[O, O2]
    ) -> SKOperatorProtocol[I, O2]: ...

    def __ior__(self, other: IcoOperatorProtocol[O, O2]) -> SKOperatorProtocol[I, O2]:
        return SKChain(self, other)


class SKChain(
    Generic[I, O, O2],
    IcoChain[I, O, O2],
    SKOperatorProtocol[I, O2],
    SKMixin,
):
    @overload
    def __or__(
        self, other: SKOperatorProtocol[O2, O3]
    ) -> SKOperatorProtocol[I, O3]: ...

    @overload
    def __or__(
        self, other: IcoOperatorProtocol[O2, O3]
    ) -> SKOperatorProtocol[I, O3]: ...

    def __or__(self, other: IcoOperatorProtocol[O2, O3]) -> SKOperatorProtocol[I, O3]:
        return SKChain(self, other)

    @overload
    def __ior__(
        self, other: SKOperatorProtocol[O2, O3]
    ) -> SKOperatorProtocol[I, O3]: ...

    @overload
    def __ior__(
        self, other: IcoOperatorProtocol[O2, O3]
    ) -> SKOperatorProtocol[I, O3]: ...

    def __ior__(self, other: IcoOperatorProtocol[O2, O3]) -> SKOperatorProtocol[I, O3]:
        return SKChain(self, other)


class SKPipeline(
    IcoPipeline[IInv],
    Generic[IInv],
    SKOperatorProtocol[IInv, IInv],
    SKMixin,
):
    mode: SKMode = "fit"

    @overload
    def __or__(
        self, other: SKOperatorProtocol[IInv, O]
    ) -> SKOperatorProtocol[IInv, O]: ...

    @overload
    def __or__(
        self, other: IcoOperatorProtocol[IInv, O]
    ) -> SKOperatorProtocol[IInv, O]: ...

    def __or__(
        self, other: IcoOperatorProtocol[IInv, O]
    ) -> SKOperatorProtocol[IInv, O]:
        return SKChain(self, other)

    @overload
    def __ior__(
        self, other: SKOperatorProtocol[IInv, O]
    ) -> SKOperatorProtocol[IInv, O]: ...

    @overload
    def __ior__(
        self, other: IcoOperatorProtocol[IInv, O]
    ) -> SKOperatorProtocol[IInv, O]: ...

    def __ior__(
        self, other: IcoOperatorProtocol[IInv, O]
    ) -> SKOperatorProtocol[IInv, O]:
        return SKChain(self, other)


class SKContextOperator(
    Generic[I, C, O],
    IcoContextOperator[I, C, O],
    SKContextOperatorProtocol[I, C, O],
    SKMixin,
):
    pass
