from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union, final, get_args, get_origin, get_type_hints

from apriori.ico.core.types import IcoOperatorProtocol, NodeType

# ─── IcoForm dataclass ───


@final
@dataclass(slots=True)
class IcoForm:
    i: str
    c: str | None
    o: str

    @property
    def name(self) -> str:
        if self.i is None and self.c is None:
            # Source node
            return f"() → {self.o}"
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
      2. Function type hints
      3. Fallback to ("Any", None, "Any")
    """
    node_type = operator.node_type
    args = get_args(getattr(operator, "__orig_class__", None))

    # ──── Match structural node type ────
    match node_type:
        case NodeType.operator:
            if args and len(args) == 2:
                i_name, o_name = (_type_name(a) for a in args)
                return IcoForm(i_name, None, o_name)

        case NodeType.chain:
            if len(operator.children) >= 2:
                first = infer_ico_form(operator.children[0])
                last = infer_ico_form(operator.children[-1])
                return IcoForm(first.i, None, last.o)

        case NodeType.map | NodeType.stream:
            child = operator.children[0] if operator.children else None
            inner = infer_ico_form(child) if child else IcoForm("Any", None, "Any")
            return IcoForm(f"Iterable[{inner.i}]", None, f"Iterable[{inner.o}]")

        case NodeType.pipeline:
            if args and len(args) == 3:
                i, c, o = (_type_name(a) for a in args)
                return IcoForm(i, c, o)

        case NodeType.process:
            if args and len(args) == 1:
                c = _type_name(args[0])
                return IcoForm(c, None, c)

        case NodeType.source:
            # Example: IcoSource[float] is () → Iterable[float]
            if args:
                o_name = _type_name(args[0])
                return IcoForm("()", None, f"Iterable[{o_name}]")

        case NodeType.sink:
            # Example: IcoSink[float] is Iterable[float] → ()
            if args and len(args) == 1:
                i_name = _type_name(args[0])
                return IcoForm(f"Iterable[{i_name}]", None, "()")

    # ──── Fallback to function type hints ────
    try:
        hints = get_type_hints(operator.fn)
        input_type = next(iter(hints.values()), Any)
        output_type = hints.get("return", Any)
        return IcoForm(_type_name(input_type), None, _type_name(output_type))
    except Exception:
        return IcoForm("Any", None, "Any")


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
    if tp is Any:
        return "Any"
    if tp is None or tp is type(None):
        return "()"

    return str(tp)
