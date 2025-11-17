from __future__ import annotations

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

from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.types import (
    IcoNodeType,
    IcoOperatorProtocol,
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
            # Operator node
            return f"{self.i} → {self.o}"
        else:
            # Pipeline node
            return f"{self.i} → {self.c} → {self.o}"

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def from_operator(operator: IcoOperatorProtocol[Any, Any]) -> IcoForm:
        """Infer IcoForm from an operator."""
        return infer_ico_form(operator)


# ──── ICO form inference ────


def infer_ico_form(operator: IcoOperatorProtocol[Any, Any]) -> IcoForm:
    """
    Infer (I, C, O) form of an ICO operator using:
      1. Generic annotations (`__orig_class__`)
      2. Function type hints for IcoOperator instances
      3. Fallback to ("Any", None, "Any")
    """
    match operator.node_type:
        # ──── Infer types for compositional nodes ────

        case IcoNodeType.chain if len(operator.children) == 2:
            # Infer types from children
            left_form = try_infer_ico_form_from_generic(operator.children[0])
            right_form = try_infer_ico_form_from_generic(operator.children[1])
            if left_form and right_form:
                return IcoForm(left_form.i, None, right_form.o)

        case IcoNodeType.stream | IcoNodeType.map if len(operator.children) == 1:
            # Infer types from body operator
            body_form = try_infer_ico_form_from_generic(operator.children[0])
            if body_form:
                return IcoForm(
                    f"Iterable[{body_form.i}]",
                    None,
                    f"Iterable[{body_form.o}]",
                )

        # ──── Infer types for special type nodes ────

        case IcoNodeType.source:
            form = try_infer_ico_form_from_generic(operator)
            if form:
                return IcoForm("()", None, f"Iterable[{form.o}]")

        case IcoNodeType.sink:
            form = try_infer_ico_form_from_generic(operator)
            if form:
                return IcoForm(f"Iterable[{form.i}]", None, "()")

        case _:
            pass

    # Infer from generics
    ico_form = try_infer_ico_form_from_generic(operator)
    if ico_form:
        return ico_form

    # Fallback to Any
    if not isinstance(operator, IcoOperator):
        return IcoForm("Any", None, "Any")

    # Fallback: infer from function type hints
    try:
        hints = get_type_hints(operator.fn)
        input_type = next(iter(hints.values()), Any)
        output_type = hints.get("return", Any)
        return IcoForm(_type_name(input_type), None, _type_name(output_type))

    except Exception:
        return IcoForm("Any", None, "Any")


def try_infer_ico_form_from_generic(operator: Any) -> IcoForm | None:
    """Infer ICO form from generic annotations, if available."""
    args = get_args(getattr(operator, "__orig_class__", None))
    if not args:
        return None

    num_args = len(args)
    if num_args == 1:
        # Process node: (C) → (C)
        c_name = _type_name(args[0])
        return IcoForm(c_name, None, c_name)

    if num_args == 2:
        # Operator node: (I) → (O)
        i_name, o_name = (_type_name(a) for a in args)
        return IcoForm(i_name, None, o_name)

    if num_args == 3:
        # Pipeline node: (I) → (C) → (O)
        i_name, c_name, o_name = (_type_name(a) for a in args)
        return IcoForm(i_name, c_name, o_name)

    return None


# ──── Type name formatter ────


def _type_name(tp: Any) -> str:
    """Return readable name for a possibly generic type (Iterable[float], tuple[int, str], etc.)."""
    origin = get_origin(tp)
    args = get_args(tp)

    # ---- Generic types (Iterable[float], dict[str, int], etc.) ----
    if origin:
        origin_name = getattr(origin, "__name__", str(origin))

        # Handle Union/Optional explicitly
        if origin is Union:
            # Optional[T] is Union[T, NoneType]
            if len(args) == 2 and type(None) in args:
                non_none = next(a for a in args if a is not type(None))
                return f"Optional[{_type_name(non_none)}]"
            args_str = " | ".join(_type_name(a) for a in args)
            return f"Union[{args_str}]"

        if origin is Literal:
            # Literal[...] is special — represent as Literal[...]
            args_str = ", ".join(repr(a) for a in args)
            return f"Literal[{args_str}]"

        # Regular generics
        if args:
            args_str = ", ".join(_type_name(a) for a in args)
            return f"{origin_name}[{args_str}]"
        return origin_name

    # ---- Base cases ----
    if isinstance(tp, type):
        if isinstance(None, tp):
            return "()"
        return tp.__name__
    if tp is Any or isinstance(tp, TypeVar):
        return "Any"
    if tp is None or tp is type(None):
        return "()"
    return str(tp)
