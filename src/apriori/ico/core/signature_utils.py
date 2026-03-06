# In this module me infer ICO signaure for possibly untyped callables,
# and must disable categories of errors related to using Any type in inspect api.
# mypy: disable-error-code=misc
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from types import FunctionType
from typing import (
    Any,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from apriori.ico.core.context_operator import IcoContextOperator
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.signature import IcoSignature, SignatureParamType


def infer_from_generic(obj: object) -> IcoSignature | None:
    """Infer ICO signature from generic type parameters on an object.

    Analyzes the generic type arguments (e.g., MyClass[int, str]) to extract
    input, context, and output types for ICO signature construction.

    Args:
        obj: Object that may have generic type parameters (__orig_class__).

    Returns:
        IcoSignature if generic args can be parsed, None otherwise.

    Example:
        >>> class MyOp(IcoOperator[int, str]): pass
        >>> op = MyOp(lambda x: str(x))
        >>> sig = infer_from_generic(op)
        >>> # Returns IcoSignature(i=int, c=None, o=str)
    """
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


def get_generic_args(obj: object) -> tuple[type, ...] | None:
    """Extract generic type arguments from an object's __orig_class__.

    Retrieves the concrete types used when instantiating a generic class,
    filtering out unresolved TypeVars that can't be used for inference.

    Args:
        obj: Object that may have been instantiated from a generic class.

    Returns:
        Tuple of concrete type arguments, or None if no usable args found.

    Note:
        Returns None if any arguments are TypeVars, indicating incomplete
        type information that can't be used for signature inference.
    """
    args = get_args(getattr(obj, "__orig_class__", None))
    if not args:
        return None

    if any((type(a) is TypeVar) for a in args):
        return None

    return args


def wrap_iterator_or_none(tp: type | None) -> type | None:
    """Wrap a type in Iterator, preserving None values.

    Utility function for transforming types to their iterator equivalents
    while handling None/NoneType specially.

    Args:
        tp: Type to potentially wrap in Iterator.

    Returns:
        Iterator[tp] if tp is not None/NoneType, otherwise tp unchanged.

    Example:
        >>> wrap_iterator_or_none(int)  # Returns Iterator[int]
        >>> wrap_iterator_or_none(None)  # Returns None
    """
    if tp is None or tp is type(None):
        return tp
    return Iterator[tp]  # type: ignore


def infer_from_flow_factory(fn: object) -> IcoSignature | None:
    """Infer ICO signature from a flow factory function's return type.

    Analyzes factory functions that return ICO operators, extracting
    the input/output types from the returned operator's generic parameters.

    Args:
        fn: Callable that should return an ICO operator with type annotations.

    Returns:
        IcoSignature derived from the return type, or None if inference fails.

    Example:
        >>> def create_converter() -> IcoOperator[int, str]:
        ...     return IcoOperator(str)
        >>> sig = infer_from_flow_factory(create_converter)
        >>> # Returns IcoSignature(i=int, c=None, o=str)
    """
    if not callable(fn):
        return None

    if type(fn) is not FunctionType:  # type: ignore[comparison-overlap]
        fn = fn.__call__

    hints = get_type_hints(fn, globalns=getattr(fn, "__globals__", {}))
    return_hint = hints.get("return", None)
    i, o = get_args(return_hint)

    return IcoSignature(i, None, o)


def infer_from_callable(obj: object) -> IcoSignature | None:
    """Infer ICO signature from a callable object's type hints.

    The main signature inference function that analyzes function signatures,
    parameter types, return types, and special cases like flow factories
    to construct an ICO type signature.

    Args:
        obj: Callable object (function, method, or callable class).

    Returns:
        IcoSignature inferred from type hints, or None if inference fails.

    Note:
        Handles special cases:
        - IcoOperator and IcoContextOperator instances
        - Flow factory methods returning typed operators
        - Functions with TypeVar parameters
        - Methods vs standalone functions

    Example:
        >>> def process(data: int) -> str:
        ...     return str(data)
        >>> sig = infer_from_callable(process)
        >>> # Returns IcoSignature(i=int, c=None, o=str)
    """
    if not callable(obj):
        return None

    if isinstance(obj, IcoOperator):
        fn = cast(Callable[[Any], Any], obj.fn)  # type: ignore
    elif isinstance(obj, IcoContextOperator):
        fn = cast(Callable[[Any, Any], Any], obj.fn)  # type: ignore
    elif not (
        inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj)
    ):
        # Assume it's a callable class instance
        fn = obj.__call__
    else:
        fn = obj

    sig = inspect.signature(fn)
    hints = get_type_hints(fn, globalns=getattr(fn, "__globals__", {}))

    params = list(sig.parameters.values())

    # If 'i', 'c' or 'o' is present in params, they will be set to type from hints
    # or to type(None) if hints is not present.
    # 'c' is optional and can be None if not present.

    i = hints.get(params[0].name, type(Any)) if params else type(None)
    c = (
        hints.get(params[1].name, type(Any)) if len(params) > 1 else None
    )  # Context parameter is optional
    o = hints.get("return", type(Any))

    # ──────── Unresolved cases ────────
    # We can't infer a meaningful signature:

    # 1. Both input and output are TypeVars
    if isinstance(i, TypeVar) and isinstance(o, TypeVar):
        return None

    # 2. Generic Alias with TypeVars, like Iterator[O].
    origins_args = get_args(i) if isinstance(i, type) else ()
    origins_args += get_args(c) if isinstance(c, type) else ()
    origins_args += get_args(o) if isinstance(o, type) else ()
    if any((type(a) is TypeVar) for a in origins_args):
        return None

    # ──────── Method is a flow factory ────────
    # In this case we try to extract ico signature from return value annotation.
    if o is not type(None):
        o_origin = get_origin(o)

        if o_origin is not None and issubclass(o_origin, IcoOperator):  # type: ignore[arg-type]
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
                    c = o_args[1] or type(None)  # Context is optional
                    o = o_args[2]
                case _:
                    pass

            i = i or type(None)
            o = o or type(None)

    if isinstance(i, SignatureParamType) and isinstance(o, SignatureParamType):
        return IcoSignature(i, c, o)

    return None


# ─── Type formatting ───


def format_ico_type(tp: SignatureParamType) -> str:
    """Format a type object into a human-readable string for display.

    Converts Python type objects into clean, readable strings suitable
    for ICO signature display, handling generics, unions, and special types.

    Args:
        tp: Type object to format.

    Returns:
        Human-readable string representation of the type.

    Example:
        >>> format_ico_type(int)  # "int"
        >>> format_ico_type(list[str])  # "list[str]"
        >>> format_ico_type(type(None))  # "()"
        >>> format_ico_type(Union[int, str])  # "int | str"

    Note:
        Handles special cases:
        - Any/object types → "Any"
        - None/NoneType → "()"
        - Optional types → "Optional[T]"
        - Union types → "T1 | T2 | ..."
        - Generic containers like Iterator, list
    """
    if tp is type(Any):
        return "Any"

    if tp is type(None):
        return "()"

    origin = get_origin(tp)

    if origin is None:
        return getattr(tp, "__name__", str(tp))

    name = getattr(origin, "__name__", str(origin))
    args = get_args(tp)

    if args:
        return f"{name}[{', '.join(format_ico_type(a) for a in args)}]"
    return name
