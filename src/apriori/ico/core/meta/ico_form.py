from __future__ import annotations

import inspect
from collections.abc import Iterator
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.chain import IcoChainOperator
from apriori.ico.core.iterate import IcoIterateOperator
from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.process import IcoProcess
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.source import IcoSource
from apriori.ico.core.stream import IcoStream

# In this module me infer ICO forms for possibly untyped callables,
# and have to disable categories of errors related to using Any type in inspect api.
# mypy: disable-error-code=misc

# ─── Typed IcoForm ───


@dataclass(frozen=True, slots=True)
class IcoForm:
    """
    A representation of the ICO form of an operator in DSL.
    Possible types are type[Any] or typing generics."""

    i: object
    c: object | None
    o: object

    def format(self) -> str:
        if self.c is None:
            return f"{_format_type(self.i)} → {_format_type(self.o)}"
        return (
            f"{_format_type(self.i)} → {_format_type(self.c)} → {_format_type(self.o)}"
        )

    @property
    def name(self) -> str:
        return self.format()


# ─── Inference dispatcher ───


def infer_ico_form(obj: object) -> IcoForm:
    ico_form: IcoForm | None = None

    for strategy in _ALL_STRATEGIES:
        ico_form = strategy(obj, ico_form)

    if ico_form is not None:
        return ico_form

    # Fallback to Any → Any
    return IcoForm(Any, None, Any)


# ─── Strategy: from __orig_class__ ───


def infer_from_generic(obj: object, ico_form: IcoForm | None) -> IcoForm | None:
    if ico_form is not None:
        return ico_form

    args = get_args(getattr(obj, "__orig_class__", None))
    if not args:
        return None

    if any((type(a) is TypeVar) for a in args):
        return None

    match len(args):
        case 1:
            return IcoForm(args[0], None, args[0])
        case 2:
            return IcoForm(args[0], None, args[1])
        case 3:
            return IcoForm(args[0], args[1], args[2])
        case _:
            pass
    return None


# ─── Strategy: structural node decomposition ───


def infer_from_node_structure(obj: object, ico_form: IcoForm | None) -> IcoForm | None:
    if ico_form is not None:
        return ico_form

    if not isinstance(obj, IcoNode):
        return None

    match obj:
        case IcoStream():
            return infer_ico_form(obj.children[0])

        case IcoAsyncStream():
            return infer_ico_form(obj.children[0])

        case IcoIterateOperator():
            return infer_ico_form(obj.children[0])

        case IcoChainOperator():
            A = infer_ico_form(obj.children[0])
            B = infer_ico_form(obj.children[1])
            return IcoForm(A.i, None, B.o)

        case IcoPipeline():
            ctx = infer_ico_form(obj.children[0])
            body = infer_ico_form(obj.children[1])
            out = infer_ico_form(obj.children[-1])
            return IcoForm(ctx.i, body.o, out.o)

        case IcoProcess():
            body = infer_ico_form(obj.children[0])
            return IcoForm(body.i, None, body.o)

        case _:
            pass

    return None


def annotate_iterator_nodes(obj: object, ico_form: IcoForm | None) -> IcoForm | None:
    if ico_form is None:
        return None

    if not isinstance(ico_form.i, type) or not isinstance(ico_form.o, type):
        return ico_form

    ico_form_i = ico_form.i
    ico_form_o = ico_form.o

    if get_origin(ico_form.i) is not Iterator:
        ico_form_i = Iterator[ico_form.i]  # type: ignore

    if get_origin(ico_form.o) is not Iterator:
        ico_form_o = Iterator[ico_form.o]  # type: ignore

    match obj:
        case IcoStream() | IcoAsyncStream() | IcoIterateOperator():
            return IcoForm(ico_form_i, None, ico_form_o)
        case IcoSource():
            return IcoForm(type(None), None, ico_form_o)

        case IcoSink():
            return IcoForm(ico_form_i, None, type(None))

        case _:
            pass

    return ico_form


# ─── Strategy: from operator function ───


def infer_from_function(obj: object, ico_form: IcoForm | None) -> IcoForm | None:
    if ico_form is not None:
        return ico_form

    fn = getattr(obj, "fn", None)
    if fn is not None:
        if isinstance(fn, IcoOperator):
            return infer_ico_form(fn)  # pyright: ignore[reportUnknownArgumentType]

        return infer_from_callable(fn, ico_form)

    return None


# ─── Strategy: from function annotations ───


def infer_from_callable(fn: object, ico_form: IcoForm | None) -> IcoForm | None:
    if ico_form is not None:
        return ico_form

    if not callable(fn):
        return None

    try:
        sig = inspect.signature(fn)
        hints = get_type_hints(fn, globalns=getattr(fn, "__globals__", {}))

        params = list(sig.parameters.values())
        i = hints.get(params[0].name, object) if params else object
        o = hints.get("return", object)

        return IcoForm(i, None, o)

    except Exception:
        return None


# ─── Strategy list ───

_ALL_STRATEGIES = [
    infer_from_generic,
    infer_from_node_structure,
    infer_from_function,
    infer_from_callable,
    annotate_iterator_nodes,
]

# ─── Type formatting ───


def _format_type(tp: object) -> str:
    if tp is Any or tp is object:
        return "Any"

    origin = get_origin(tp)
    args = get_args(tp)

    if tp is None or tp is type(None):
        return "()"

    if origin is Union:
        args_ = [a for a in args if a is not type(None)]
        if len(args_) == 1:
            return f"Optional[{_format_type(args_[0])}]"
        return " | ".join(_format_type(a) for a in args_)

    if origin is Literal:
        name = getattr(origin, "__name__", str(origin))
        if args:
            return f"{name}[{', '.join(_format_type(a) for a in args)}]"
        return name

    if origin is Iterator:
        return f"Iterator[{_format_type(args[0])}]"

    if isinstance(tp, type):
        return tp.__name__

    return str(tp)
