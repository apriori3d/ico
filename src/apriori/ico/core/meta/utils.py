from __future__ import annotations

from collections.abc import Iterator
from typing import (
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
)

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
