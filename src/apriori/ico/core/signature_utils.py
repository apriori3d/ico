# In this module me infer ICO forms for possibly untyped callables,
# and have to disable categories of errors related to using Any type in inspect api.
# mypy: disable-error-code=misc
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from types import FunctionType
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

from apriori.ico.core.context_operator import IcoContextOperator
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.signature import IcoSignature


def infer_from_generic(obj: object) -> IcoSignature | None:
    args = get_generic_args(obj)
    if args is None:
        return None

    match len(args):
        case 2:
            return IcoSignature(args[0], None, args[1])
        case 3:
            return IcoSignature(args[0], args[1], args[2])
        case _:
            pass
    return None


def get_generic_args(obj: object) -> tuple[object, ...] | None:
    args = get_args(getattr(obj, "__orig_class__", None))
    if not args:
        return None

    if any((type(a) is TypeVar) for a in args):
        return None

    return args


def wrap_iterator_or_none(tp: object) -> object:
    if tp is None or tp is type(None):
        return tp
    return Iterator[tp]  # type: ignore


def infer_from_flow_factory(fn: object) -> IcoSignature | None:
    if not callable(fn):
        return None

    if type(fn) is not FunctionType:
        fn = fn.__call__

    hints = get_type_hints(fn, globalns=getattr(fn, "__globals__", {}))
    return_hint = hints.get("return", None)
    i, o = get_args(return_hint)

    return IcoSignature(i, None, o)


def infer_from_callable(obj: object) -> IcoSignature | None:
    if not callable(obj):
        return None

    if isinstance(obj, IcoOperator):
        fn = cast(Callable[[Any], Any], obj.fn)  # type: ignore
    elif isinstance(obj, IcoContextOperator):
        fn = cast(Callable[[Any, Any], Any], obj.fn)  # type: ignore
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

    return IcoSignature(i, c, o)


# ─── Type formatting ───


def format_ico_type(tp: object) -> str:
    if tp is Any or tp is object:
        return "Any"

    origin = get_origin(tp)
    args = get_args(tp)

    if tp is None or tp is type(None):
        return "()"

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
