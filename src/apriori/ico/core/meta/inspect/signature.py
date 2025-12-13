# In this module me infer ICO forms for possibly untyped callables,
# and have to disable categories of errors related to using Any type in inspect api.
# mypy: disable-error-code=misc
from __future__ import annotations

import inspect
from collections.abc import Iterator
from types import GenericAlias
from typing import (
    Any,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.chain import IcoChainOperator
from apriori.ico.core.context_pipeline import IcoContextPipeline
from apriori.ico.core.epoch import IcoEpoch
from apriori.ico.core.meta.meta import IcoSignature
from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.process import IcoProcess
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.stream import IcoStream

# ────────────────────────────────────────────────
# Inference strategy dispatcher
# ────────────────────────────────────────────────


def infer_signature(obj: object) -> IcoSignature:
    ico_form: IcoSignature | None = None

    for strategy in _ALL_STRATEGIES:
        ico_form = strategy(obj, ico_form)

    if ico_form is not None:
        return ico_form

    # Fallback to Any → Any
    return IcoSignature(Any, None, Any)


# ─── Strategy: from __orig_class__ ───


def infer_from_generic(
    obj: object, ico_form: IcoSignature | None
) -> IcoSignature | None:
    if ico_form is not None:
        return ico_form

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


# ─── Strategy: structural node decomposition ───


def infer_from_node_structure(
    obj: object, ico_form: IcoSignature | None
) -> IcoSignature | None:
    if ico_form is not None:
        return ico_form

    if not isinstance(obj, IcoNode):
        return None

    match obj:
        case IcoSink():
            body_form = infer_signature(obj.original_fn)
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

        case IcoChainOperator():
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
            return IcoSignature(source_form.o, context_form.c, context_form.o)

        case _:
            pass

    return None


def _wrap_iterator(tp: object) -> object:
    if tp is None or tp is type(None):
        return tp
    return Iterator[tp]  # type: ignore


# ─── Strategy: from operator function ───


def infer_from_ico_target(
    obj: object, ico_form: IcoSignature | None
) -> IcoSignature | None:
    if ico_form is not None:
        return ico_form

    if not isinstance(obj, IcoNode):
        return ico_form

    fn = obj.original_fn
    if fn is None:
        return ico_form

    if isinstance(fn, IcoOperator):
        return infer_signature(fn)  # pyright: ignore[reportUnknownArgumentType]

    if not callable(fn):
        return ico_form

    if inspect.isfunction(fn) or inspect.ismethod(fn):
        return infer_from_callable(fn, ico_form)

    return infer_from_callable(fn.__call__, ico_form)


# ─── Strategy: from function annotations ───


def infer_from_callable(
    fn: object, ico_form: IcoSignature | None
) -> IcoSignature | None:
    if ico_form is not None:
        return ico_form

    if not callable(fn):
        return ico_form

    try:
        sig = inspect.signature(fn)
        hints = get_type_hints(fn, globalns=getattr(fn, "__globals__", {}))

        params = list(sig.parameters.values())
        i = hints.get(params[0].name, None) if params else None
        c = hints.get(params[1].name, None) if len(params) > 1 else None
        o = hints.get("return", None)

        # No hints, return existing ico form.
        if i is None and o is None:
            return ico_form

        # Both input and output are unresolved, return existing ico form.
        if (
            i is not None
            and isinstance(i, TypeVar)
            and o is not None
            and isinstance(o, TypeVar)
        ):
            return ico_form

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

        if ico_form is not None:
            if i is not None and ico_form.i is not None:
                i = replace_deepest_typevar(i, I, ico_form.i)

            if o is not None and ico_form.o is not None:
                o = replace_deepest_typevar(o, O, ico_form.o)

            return IcoSignature(i, ico_form.c, o)

        return IcoSignature(i, c, o)

    except Exception:
        return ico_form


def replace_deepest_typevar(
    type: GenericAlias,
    target: TypeVar,
    replacement: object,
) -> object:
    args = get_args(type)
    if not args:
        return replacement
    else:
        origin = get_origin(type)
        new_args = tuple(
            replace_deepest_typevar(arg, target, replacement) for arg in args
        )
        return origin[new_args]  # type: ignore


# ─── Strategy list ───

_ALL_STRATEGIES = [
    infer_from_generic,
    infer_from_node_structure,
    infer_from_ico_target,
    infer_from_callable,
]
