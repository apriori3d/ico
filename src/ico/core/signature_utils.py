# In this module me infer ICO signaure for possibly untyped callables,
# and must disable categories of errors related to using Any type in inspect api.
# mypy: disable-error-code=misc
from __future__ import annotations

import inspect
from collections import OrderedDict
from collections.abc import Callable
from types import FunctionType
from typing import (
    Any,
    Generic,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from ico.core.context_operator import IcoContextOperator
from ico.core.operator import IcoOperator
from ico.core.signature import IcoSignature, SignatureParamType, is_supported_type

# ──────── Generic Type Resolution  ────────


def resolve_types_from_generic(
    obj: object,
    base_class: type,
    *vars: TypeVar,
) -> list[SignatureParamType | None]:
    """Resolve concrete types for specified TypeVars from an object's generic bases.
    This function traverses the MRO of the object's class to find generic base classes
    that are subclasses of the specified base_class, and attempts to resolve the provided TypeVars
    to concrete types based on the generic arguments of those bases.
    Args:
        obj: The object whose class hierarchy to inspect for generic type resolution.
        base_class: The base class to look for in the MRO (e.g., IcoOperator).
        *vars: The TypeVars to resolve (e.g., I, O).
    """

    # Collect generic bases in MRO that are subclasses of base_class.
    # Result would be in form :
    #  - ..., Generic[I, O], Operator[I, O], ... for the case of generic type parameters in class definition,
    #  - ..., Generic[I, O], Operator[int, float], ... for the case of concrete type in class definitions.
    # This allows to build a mapping of TypeVars:
    #  - (I, C, O) -> (int, str, etc.) to concrete types
    #  - (I, O) -> (I, I) to other TypeVars
    # and resolve them as we go through the MRO.

    resolve_hint_order = [
        orig
        for c in obj.__class__.__mro__
        for orig in place_generic_first(getattr(c, "__orig_bases__", ()))
        if issubclass(get_origin(orig) or object, base_class | Generic)
    ]

    # If the instance itself has a generic type hint - add it to the list to resolve.
    # Expected
    #  - all generic args are concrete types: for the case of generic type parameters in class definition.
    #  - None for the case of staticly typed classes,
    # otherwise we won't be able to resolve them.

    if instance_class := getattr(obj, "__orig_class__", None):
        resolve_hint_order.insert(0, instance_class)

    if len(resolve_hint_order) < 2:
        return [None] * len(vars)  # No generic bases found, can't resolve.

    resolving_vars = OrderedDict[TypeVar, TypeVar | SignatureParamType](
        (var, var) for var in vars
    )

    resolve_hint_order = list(reversed(resolve_hint_order))
    resolve_generic_hints = resolve_hint_order[::2]
    resolve_class_hints = resolve_hint_order[1::2]

    # Resolve given vars in generic arguments using reverse MRO as we go.
    for generic_hint, cls_hint in zip(
        resolve_generic_hints, resolve_class_hints, strict=False
    ):
        if get_origin(generic_hint) is not Generic or not issubclass(
            get_origin(cls_hint) or object, base_class
        ):
            break  # We expect pairs of Generic and base_class, if not - we can't resolve further.

        # Make pairs of generic and class hints
        generic_args: tuple[Any, ...] = get_args(generic_hint)  # pyright: ignore[reportAssignmentType]
        cls_args: tuple[Any, ...] = get_args(cls_hint)  # pyright: ignore[reportAssignmentType]

        assert len(generic_args) == len(
            cls_args
        ), "Mismatch in number of generic arguments."

        # Resoulute TypeVars in generic args using substitution from previous iterations.
        for var, var_cls in resolving_vars.items():
            for generic_arg, cls_arg in zip(generic_args, cls_args, strict=False):
                var_cls = replace_typevar(var_cls, generic_arg, cls_arg)
            resolving_vars[var] = var_cls

    # Return resolved types for the requested vars, or None if they couldn't be resolved to concrete types.
    return [
        t if type(t) is not TypeVar and not type_contain_any_typevar(t, *vars) else None
        for t in resolving_vars.values()
    ]


# ──────── Generic Type Resolution and Replacement Utilities ────────


def place_generic_first(
    args: tuple[SignatureParamType, ...],
) -> tuple[SignatureParamType, ...]:
    """Utility to reorder generic arguments, placing the first Generic found at the front.
    This is used to ensure that when resolving generic type parameters, we get expected order:
    ..., Generic[I, O], Operator[I,O] instead of ..., Operator[I,O], Generic[I, O]."""

    generic_arg = next(
        (arg for arg in args if get_origin(arg) is Generic),
        None,
    )
    if generic_arg is None:
        return args

    generic_index = args.index(generic_arg)

    if generic_index == 0:
        return args

    return (generic_arg,) + args[:generic_index] + args[generic_index + 1 :]


def type_contain_any_typevar(
    target_type: TypeVar | SignatureParamType | None, *vars: TypeVar
) -> bool:
    """Check if a type annotation contains any of the specified TypeVars.
    Args:
        target_type: The type annotation to check, which may be a TypeVar or a typing generic or concrete type.
        *vars: The TypeVars to look for within the target_type.

    Returns:
        True if any of the specified TypeVars are found within the target_type, False otherwise.
    """
    if target_type is None:
        return False

    if type(target_type) is TypeVar:
        return target_type in vars if len(vars) > 0 else True

    # GenericAlias can contain nested TypeVars, so we check recursively
    return any(
        type_contain_any_typevar(arg, *vars)
        for arg in get_args(target_type)  # pyright: ignore[reportArgumentType]
    )


def replace_typevar(
    value: TypeVar | SignatureParamType,
    var: TypeVar,
    replacement: SignatureParamType,
) -> TypeVar | SignatureParamType:
    """Recursively replace a TypeVar in a type annotation with a concrete or generic type.

    Args:
        value: The type annotation to process, which may contain the TypeVar.
        var: The TypeVar to replace.
        replacement: The concrete or generic type to substitute for the TypeVar.

    Returns:
        A new type annotation with the TypeVar replaced by the concrete type if found,
        otherwise returns the original type annotation.
    Example:
        >>> replace_typevar(I, list[I], int)  # Returns list[int]
        >>> replace_typevar(I, list[O], int)  # Returns list[I]"""
    if value == var:
        return replacement

    origin = get_origin(value)
    if origin is None:
        return value

    args = get_args(value)
    if not args:
        return value

    replaced_args = tuple(replace_typevar(arg, var, replacement) for arg in args)
    return origin[replaced_args]  # type: ignore


# ──────── Flow Factory Signature Inference ────────


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

    return IcoSignature(
        wrap_type_if_any(i),
        None,
        wrap_type_if_any(o),
    )


# ──────── Callable Signature Inference ────────


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
        fn = cast(Callable[[Any], Any], obj._fn)  # type: ignore
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
    origins_args = get_args(i)
    origins_args += get_args(c)
    origins_args += get_args(o)
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

    if (
        is_supported_type(i)
        and is_supported_type(o)
        and (c is None or is_supported_type(c))
    ):
        return IcoSignature(
            i=wrap_type_if_any(i),
            c=wrap_type_if_any(c) if c is not None else None,
            o=wrap_type_if_any(o),
        )

    return None


# ─── Type formatting ───


def format_ico_type(tp: Any) -> str:
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

    if isinstance(origin, type) and len(args) == 1 and args[0] is Any:
        return "Any"

    if args:
        return f"{name}[{', '.join(format_ico_type(a) for a in args)}]"
    return name


# ──────── Misc Utilities ────────


def wrap_type_if_any(t: Any) -> Any:
    """Utility to wrap a type in Any if it's not supported for ICO signatures."""
    return type[Any] if t is Any else t
