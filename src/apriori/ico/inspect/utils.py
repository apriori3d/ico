from __future__ import annotations

from types import FunctionType

# ──────────── Helpers  ────────────


def extract_fn_display_name(fn: object) -> str | None:
    cls = getattr(fn, "__class__", None)

    if cls is FunctionType:
        return getattr(fn, "__name__", None)

    return None


def extract_class_display_name(obj: object) -> str | None:
    cls = getattr(obj, "__class__", None)

    if cls is None:
        return None

    return getattr(cls, "__name__", None)
