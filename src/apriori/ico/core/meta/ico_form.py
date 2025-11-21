from __future__ import annotations

import inspect
from dataclasses import dataclass
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

from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.process import IcoProcess
from apriori.ico.core.types import (
    IcoNode,
    IcoNodeType,
)

# ─── IcoForm dataclass ───


@final
@dataclass(slots=True)
class IcoForm:
    i: str
    c: str | None
    o: str

    @property
    def name(self) -> str:
        if self.c is None:
            return f"{self.i} → {self.o}"
        else:
            return f"{self.i} → {self.c} → {self.o}"

    @staticmethod
    def from_operator(operator: IcoNode) -> IcoForm:
        return infer_ico_form(operator)


# ──── ICO form inference ────


def infer_ico_form(obj: Any) -> IcoForm:
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
        function_form = try_infer_ico_form_from_function(obj.fn)  # pyright: ignore[reportUnknownMemberType]
        if function_form:
            return function_form

    function_form = try_infer_ico_form_from_function(obj)  # pyright: ignore[reportUnknownVariableType]
    if function_form:
        return function_form

    # 4) Final fallback
    return IcoForm("Any", None, "Any")


# ──── Generic inference helper ────


def infer_ico_form_by_node_type(operator: IcoNode) -> IcoForm | None:
    match operator.node_type:
        # Structural composition nodes

        case IcoNodeType.chain if len(operator.children) == 2:
            left = infer_ico_form(operator.children[0])
            right = infer_ico_form(operator.children[1])
            return IcoForm(left.i, None, right.o)

        case IcoNodeType.stream | IcoNodeType.map if len(operator.children) == 1:
            body = infer_ico_form(operator.children[0])
            return IcoForm(
                f"Iterator[{body.i}]",
                None,
                f"Iterator[{body.o}]",
            )

        case IcoNodeType.pipeline:
            generic_form = try_infer_ico_form_from_generic(operator)
            if generic_form:
                return generic_form

            # Infer from context, body, output as child operators
            if len(operator.children) >= 3:
                context = infer_ico_form(operator.children[0])
                body = infer_ico_form(operator.children[1])
                output = infer_ico_form(operator.children[-1])
                return IcoForm(context.i, body.o, output.o)

            # Infer from context, body, output as functions
            if isinstance(operator, IcoPipeline) and len(operator.body) >= 1:  # type: ignore
                context_fn_form = infer_ico_form(
                    operator.context  # pyright: ignore[reportUnknownMemberType]
                )
                body_fn_form = infer_ico_form(
                    operator.body[0]  # pyright: ignore[reportUnknownMemberType]
                )
                output_fn_form = infer_ico_form(
                    operator.output  # pyright: ignore[reportUnknownMemberType]
                )
                if context_fn_form and body_fn_form and output_fn_form:
                    return IcoForm(
                        context_fn_form.i,
                        body_fn_form.o,
                        output_fn_form.o,
                    )

        case IcoNodeType.process:
            generic_names = get_generic_type_names(operator)
            if generic_names and len(generic_names) == 1:
                return IcoForm(generic_names[0], None, generic_names[0])

            if len(operator.children) == 1:
                body_form = infer_ico_form(operator.children[0])
                return IcoForm(body_form.i, None, body_form.o)

            if isinstance(operator, IcoProcess):
                function_form = infer_ico_form(operator.body)  # pyright: ignore[reportUnknownMemberType]
                if function_form:
                    return function_form

        # Note: ICO Form should have at least two types (I, O) to be defined.
        # Some nodes only have one type each,
        # so ICO Form must be defined depending on node type.

        case IcoNodeType.source:
            # Source has only one generic type: Output
            generic_names = get_generic_type_names(operator)

            if generic_names and len(generic_names) == 1:
                return IcoForm("()", None, f"Iterator[{generic_names[0]}]")

        case IcoNodeType.sink:
            # Sink has only one generic type: Input
            generic_names = get_generic_type_names(operator)

            if generic_names and len(generic_names) == 1:
                return IcoForm(f"Iterator[{generic_names[0]}]", None, "()")

        case IcoNodeType.runtime:
            if not get_generic_type_names(operator):
                return IcoForm("()", None, "()")

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

    if len(type_names) == 2:
        i, o = type_names
        return IcoForm(i, None, o)

    if len(type_names) == 3:
        i, c, o = type_names
        return IcoForm(i, c, o)

    return None


def try_infer_ico_form_from_function(fn: Any) -> IcoForm | None:
    try:
        signature = inspect.signature(fn)  # type: ignore
        hints = get_type_hints(fn)  # type: ignore

        # extract input type (first annotated parameter)
        params = list(signature.parameters.values())
        if params and params[0].annotation is not inspect._empty:  # pyright: ignore[reportPrivateUsage]
            input_type = hints.get(params[0].name, Any)
        else:
            input_type = Any

        # extract return type
        if signature.return_annotation is not inspect._empty:  # pyright: ignore[reportPrivateUsage]
            output_type = hints.get("return", Any)
        else:
            output_type = Any

        return IcoForm(_type_name(input_type), None, _type_name(output_type))

    except Exception:
        return None


# ──── Formatter ────


def _type_name(tp: Any) -> str:
    origin = get_origin(tp)
    args = get_args(tp)

    if origin:
        origin_name = getattr(origin, "__name__", str(origin))

        if origin is Union:
            if len(args) == 2 and type(None) in args:
                non_none = next(a for a in args if a is not type(None))
                return f"Optional[{_type_name(non_none)}]"
            return " | ".join(_type_name(a) for a in args)

        if origin is Literal:
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
