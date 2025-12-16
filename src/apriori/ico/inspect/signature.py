# In this module me infer ICO forms for possibly untyped callables,
# and have to disable categories of errors related to using Any type in inspect api.
# mypy: disable-error-code=misc
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import FunctionType, GenericAlias
from typing import (
    Any,
    Literal,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.chain import IcoChain
from apriori.ico.core.context_pipeline import IcoContextPipeline
from apriori.ico.core.epoch import IcoEpoch
from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.process import IcoProcess
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.source import IcoSource
from apriori.ico.core.stream import IcoStream
from apriori.ico.runtime.agent.mp_process.mp_process import MPProcess
from apriori.ico.utils.data.batcher import IcoBatcher

# ────────────────────────────────────────────────
# Signature descriptions
# ────────────────────────────────────────────────


@dataclass(slots=True)
class IcoSignature:
    """
    A representation of the ICO form of an operator in DSL.
    Possible types are type[Any] or typing generics."""

    i: object
    c: object | None
    o: object

    def format(self) -> str:
        if self.c is None or self.c is type(None):
            return f"{format_ico_type(self.i)} → {format_ico_type(self.o)}"
        return f"{format_ico_type(self.i)}, {format_ico_type(self.c)} → {format_ico_type(self.o)}"

    @property
    def name(self) -> str:
        return self.format()

    @property
    def has_input(self) -> bool:
        return self.i is not None and self.i is not type(None)

    @property
    def has_context(self) -> bool:
        return self.c is not None and self.c is not type(None)

    @property
    def has_output(self) -> bool:
        return self.o is not None and self.o is not type(None)


# ────────────────────────────────────────────────
# Inference strategy dispatcher
# ────────────────────────────────────────────────


def infer_signature(obj: object) -> IcoSignature:
    signature: IcoSignature | None = None

    for strategy in _ALL_STRATEGIES:
        signature = strategy(obj)

        if signature is not None:
            return signature

    # Fallback to Any → Any
    return IcoSignature(Any, None, Any)


# ─── Strategy: Node-specific inference ───


def infer_by_node_type(obj: object) -> IcoSignature | None:
    # if signature is not None:
    #    return signature

    if not isinstance(obj, IcoNode):
        return None

    match obj:
        case IcoSource():
            source = cast(IcoSource[Any], obj)
            provider_form = infer_from_callable(source.provider)

            if provider_form is None:
                return IcoSignature(
                    i=type(None),
                    c=type(None),
                    o=Iterator[Any],
                )

            # Provider returns an Iterable[T], we need to convert it to Iterator[T]
            # to match IcoSource signature
            if isinstance(provider_form.o, GenericAlias):
                o_args = get_args(provider_form.o)

                return IcoSignature(
                    i=type(None),
                    c=type(None),
                    o=Iterator[o_args],
                )

            return None

        case IcoSink():
            sink = cast(IcoSink[Any], obj)
            body_form = infer_signature(sink.consumer)
            return IcoSignature(
                i=_wrap_iterator(body_form.i),
                c=type(None),
                o=type(None),
            )

        case IcoStream() | IcoAsyncStream():
            assert len(obj.children) == 1
            body_form = infer_signature(obj.children[0])
            return IcoSignature(
                i=_wrap_iterator(body_form.i),
                c=_wrap_iterator(body_form.c),
                o=_wrap_iterator(body_form.o),
            )

        case IcoChain():
            assert len(obj.children) == 2
            a = infer_signature(obj.children[0])
            b = infer_signature(obj.children[1])
            return IcoSignature(a.i, None, b.o)

        case IcoPipeline():
            assert len(obj.children) >= 1
            return infer_signature(obj.children[0])

        case IcoProcess():
            assert len(obj.children) == 1
            body = infer_signature(obj.children[0])
            return IcoSignature(body.i, None, body.o)

        case IcoContextPipeline():
            assert len(obj.children) >= 1
            return infer_signature(obj.children[0])

        case IcoEpoch():
            assert len(obj.children) == 2
            source_form = infer_signature(obj.children[0])
            context_form = infer_signature(obj.children[1])
            return IcoSignature(
                source_form.o,
                context_form.c,
                context_form.o,
            )

        case MPProcess():
            process = cast(MPProcess[Any, Any], obj)
            return infer_from_flow_factory(process.flow_factory)

        case IcoBatcher():
            # Generic parameter for batcher dosn't include Iterators
            batcher = cast(IcoBatcher[Any], obj)
            signature = infer_from_generic(batcher)

            if signature is not None:
                return IcoSignature(
                    i=_wrap_iterator(signature.i),
                    c=None,
                    o=_wrap_iterator(_wrap_iterator(signature.o)),
                )
        case _:
            pass

    return None


# ─── Strategy: from __orig_class__ ───


def infer_from_generic(obj: object) -> IcoSignature | None:
    args = get_args(getattr(obj, "__orig_class__", None))
    if not args:
        return None

    if any((type(a) is TypeVar) for a in args):
        return None

    match len(args):
        case 1:
            return IcoSignature(args[0], None, args[0])
        case 2:
            return IcoSignature(args[0], None, args[1])
        case 3:
            return IcoSignature(args[0], args[1], args[2])
        case _:
            pass
    return None


def _wrap_iterator(tp: object) -> object:
    if tp is None or tp is type(None):
        return tp
    return Iterator[tp]  # type: ignore


def infer_from_flow_factory(fn: object) -> IcoSignature | None:
    if not callable(fn):
        return None

    hints = get_type_hints(fn, globalns=getattr(fn, "__globals__", {}))
    return_hint = hints.get("return", None)
    i, o = get_args(return_hint)

    return IcoSignature(i, None, o)


# ─── Strategy: from function annotations ───


def infer_from_callable(obj: object) -> IcoSignature | None:
    if not callable(obj):
        return None

    if isinstance(obj, IcoOperator):
        fn = cast(Callable[[Any], Any], obj.fn)  # type: ignore
    elif type(obj) is not FunctionType:
        fn = obj.__call__
    else:
        fn = obj

    # try:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn, globalns=getattr(fn, "__globals__", {}))

    params = list(sig.parameters.values())
    i = hints.get(params[0].name, None) if params else None
    c = hints.get(params[1].name, None) if len(params) > 1 else None
    o = hints.get("return", None)

    # No hints
    if i is None and o is None:
        return None

    # Both input and output are unresolved
    if (
        i is not None
        and isinstance(i, TypeVar)
        and o is not None
        and isinstance(o, TypeVar)
    ):
        return None

    # Special case: method can be a flow factory.
    # In this case we try to extract ico form from return value annotation.
    o_origin = get_origin(o)

    if o_origin is not None and issubclass(o_origin, IcoOperator):
        o_args = get_args(o)

        match len(o_args):
            case 1:
                i = o_args[0]
                o = o_args[0]
            case 2:
                i = o_args[0]
                o = o_args[1]
            case 3:
                i = o_args[0]
                c = o_args[1]
                o = o_args[2]
            case _:
                pass
        return IcoSignature(i, c, o)

    # Special case: item type may be inferered from generic ICO form, but origin, like Iterator, is lost.
    # Yet it may be restored here from function hints.
    # In order to get complete ico form here both sources are combined.

    # if ico_form is not None:
    #     if i is not None and ico_form.i is not None:
    #         i = replace_deepest_typevar(i, I, ico_form.i)

    #     if o is not None and ico_form.o is not None:
    #         o = replace_deepest_typevar(o, O, ico_form.o)

    #     return IcoSignature(i, ico_form.c, o)

    return IcoSignature(i, c, o)

    # return None

    # except Exception:
    #    return ico_form


# def replace_deepest_typevar(
#     type: GenericAlias,
#     target: TypeVar,
#     replacement: object,
# ) -> object:
#     args = get_args(type)
#     if not args:
#         return replacement
#     else:
#         origin = get_origin(type)
#         new_args = tuple(
#             replace_deepest_typevar(arg, target, replacement) for arg in args
#         )
#         return origin[new_args]  # type: ignore


# ─── Strategy list ───

_ALL_STRATEGIES = [
    infer_by_node_type,
    infer_from_generic,
    infer_from_callable,
]


# ─── Type formatting ───


def format_ico_type(tp: object) -> str:
    if tp is Any or tp is object:
        return "Any"

    origin = get_origin(tp)
    args = get_args(tp)

    if tp is None or tp is type(None):
        return "None"

    if origin is Union:
        args_ = [a for a in args if a is not type(None)]
        if len(args_) == 1:
            return f"Optional[{format_ico_type(args_[0])}]"
        return " | ".join(format_ico_type(a) for a in args_)

    if origin is Literal:
        name = getattr(origin, "__name__", str(origin))
        if args:
            return f"{name}[{', '.join(format_ico_type(a) for a in args)}]"
        return name

    if origin is Iterator:
        return f"Iterator[{format_ico_type(args[0])}]"

    if isinstance(tp, type):
        return tp.__name__

    return str(tp)
