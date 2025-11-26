from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import (
    Any,
    Literal,
    TypeVar,
    Union,
    final,
    get_args,
    get_origin,
    get_type_hints,
)

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


# ─── IcoForm dataclass ───


@final
class IcoForm:
    __slots__ = ("i", "c", "o")

    i: str
    c: str | None
    o: str

    def __init__(self, i: str, c: str | None, o: str) -> None:
        self.i = i
        self.c = c
        self.o = o

    @property
    def name(self) -> str:
        if self.c is None:
            return f"{self.i} → {self.o}"
        else:
            return f"{self.i} → {self.c} → {self.o}"

    @staticmethod
    def from_node(node: IcoNode) -> IcoForm:
        return infer_ico_form(node)


# ──── ICO form inference ────


def infer_ico_form(obj: object) -> IcoForm:
    """
    Infer (I, C, O) form of an ICO operator using, in order:

      1. Generic annotations (`__orig_class__`)
      2. Function type hints (for IcoOperator only)
      3. Fallback to ("Any", None, "Any")
    """

    # 1) Try match by structural node type first (map, stream, pipeline, etc.)
    if isinstance(obj, IcoNode):
        ico_form = infer_ico_form_by_node_type(obj)
        if ico_form:
            return ico_form

    # 2) Try infer from generics (__orig_class__)
    generic_form = try_infer_ico_form_from_generic(obj)
    if generic_form:
        return generic_form

    # 3) try infer using function annotations via inspect
    if isinstance(obj, IcoOperator):
        function_form = try_infer_ico_form_from_function(obj.fn)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        if function_form:
            return function_form

    if callable(obj):  # pyright: ignore[reportUnknownArgumentType]
        function_form = try_infer_ico_form_from_function(obj)  # pyright: ignore[reportUnknownArgumentType]
        if function_form:
            return function_form

    # 4) Final fallback
    return IcoForm("Unknown", None, "Unknown")


# ──── Generic inference helper ────


def infer_ico_form_by_node_type(node: IcoNode) -> IcoForm | None:
    generic_form = try_infer_ico_form_from_generic(node)

    match node:
        # Structural composition nodes

        case IcoChainOperator():
            if generic_form:
                return generic_form

            assert len(node.children) == 2
            left = infer_ico_form(node.children[0])
            right = infer_ico_form(node.children[1])
            return IcoForm(left.i, None, right.o)

        case IcoStream() | IcoIterateOperator():
            if generic_form:
                return IcoForm(
                    f"Iterator[{generic_form.i}]",
                    None,
                    f"Iterator[{generic_form.o}]",
                )

            assert len(node.children) == 1
            body_form = infer_ico_form(node.children[0])
            assert body_form is not None

            # Body ICO form may be infered from generics of fn itself.
            # But generic does not provide Iterator annotation.
            if body_form.i.startswith("Iterator[") and body_form.o.startswith(
                "Iterator["
            ):
                return body_form

            return IcoForm(
                f"Iterator[{body_form.i}]",
                None,
                f"Iterator[{body_form.o}]",
            )

        case IcoPipeline():
            if generic_form:
                return generic_form

            assert len(node.children) >= 3

            # Infer from context, body, output as child operators
            context = infer_ico_form(node.children[0])
            body = infer_ico_form(node.children[1])
            output = infer_ico_form(node.children[-1])
            return IcoForm(context.i, body.o, output.o)

        case IcoProcess():
            if generic_form:
                return generic_form

            assert len(node.children) == 1
            body_form = infer_ico_form(node.children[0])
            return IcoForm(body_form.i, None, body_form.o)

        # Note: ICO Form should have at least two types (I, O) to be defined.
        # Some nodes only have one type each,
        # so ICO Form must be defined depending on node type.

        case IcoSource():
            # Source has only one generic type: Output
            if generic_form:
                return IcoForm("()", None, f"Iterator[{generic_form.o}]")

            return infer_ico_form(node.fn)  # pyright: ignore[reportUnknownArgumentType]

        case IcoSink():
            # Sink has only one generic type: Input
            if generic_form:
                return IcoForm(f"Iterator[{generic_form.i}]", None, "()")

            return infer_ico_form(node.fn)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        case _:
            pass

    return None


def get_generic_type_names(operator: Any) -> list[str] | None:
    args = get_args(getattr(operator, "__orig_class__", None))
    if not args:
        return None
    return [_type_name(a) for a in args]


def try_infer_ico_form_from_generic(operator: Any) -> IcoForm | None:
    type_names = get_generic_type_names(operator)

    if not type_names:
        return None

    if len(type_names) == 1:
        c = type_names[0]
        if c == "Any":
            return None
        return IcoForm(c, None, c)

    if len(type_names) == 2:
        i, o = type_names
        if i == "Any" or o == "Any":
            return None
        return IcoForm(i, None, o)

    if len(type_names) == 3:
        i, c, o = type_names
        if i == "Any" or c == "Any" or o == "Any":
            return None
        return IcoForm(i, c, o)

    return None


def try_infer_ico_form_from_function(fn: Callable[[Any], Any]) -> IcoForm | None:
    try:
        input_type_name = "Any"
        output_type_name = "Any"

        signature = inspect.signature(fn)  # pyright: ignore[reportArgumentType]
        hints = get_type_hints(fn)

        # extract input type (first annotated parameter)
        params = list(signature.parameters.values())
        if params and params[0].annotation is not inspect._empty:  # pyright: ignore[reportPrivateUsage]
            input_type = hints.get(params[0].name, None)
            if input_type is not None:
                input_type_name = _type_name(input_type)

        # extract return type
        if signature.return_annotation is not inspect._empty:  # pyright: ignore[reportPrivateUsage]
            output_type = hints.get("return", None)
            if output_type is not None:
                output_type_name = _type_name(output_type)

        return IcoForm(input_type_name, None, output_type_name)

    except Exception:
        return None


# ──── Formatter ────


def _type_name(tp: Any) -> str:
    origin = get_origin(tp)
    args = get_args(tp)

    if origin is not None:
        origin_name = getattr(origin, "__name__", str(origin))

        if isinstance(origin, type) and origin == Union:
            if len(args) == 2 and type(None) in args:
                non_none = next(a for a in args if a is not type(None))
                return f"Optional[{_type_name(non_none)}]"
            return " | ".join(_type_name(a) for a in args)

        if isinstance(origin, type) and origin == Literal:
            return f"Literal[{', '.join(repr(a) for a in args)}]"

        if args:
            return f"{origin_name}[{', '.join(_type_name(a) for a in args)}]"

        return origin_name

    # Base cases
    if tp in (None, type(None)):
        return "()"
    if isinstance(tp, type):
        return tp.__name__
    if tp is Any or isinstance(tp, TypeVar):
        return "Any"

    return str(tp)
