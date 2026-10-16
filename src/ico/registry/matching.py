"""Type matching for registry search.

Unlike :func:`ico.core.signature_check.is_assignable`, registry search is
deliberately strict: plain classes match exactly or by subclass, generics
match when their origins do and their arguments are equal (no variance).
"""

from __future__ import annotations

from typing import Any, Literal, get_args, get_origin

from ico.core.signature_check import is_any_type

MatchRank = Literal[0, 1]
"""0 for an exact match, 1 for a subclass match."""

EXACT: MatchRank = 0
SUBCLASS: MatchRank = 1


def match_type(sub: Any, sup: Any) -> MatchRank | None:
    """Check whether a value of type ``sub`` can be used where ``sup`` is expected.

    Rules:
        * equal types match exactly;
        * plain classes match when ``sub`` is a subclass of ``sup``;
        * generics match when the origin of ``sub`` is a subclass of the
          origin of ``sup`` and their arguments are equal, e.g.
          ``list[int]`` matches ``Sequence[int]`` but not ``list[object]``;
        * ``Any``, unions and other special forms only match exactly.

    Returns:
        :data:`EXACT`, :data:`SUBCLASS` or ``None`` when the types do not match.

    Example:
        >>> match_type(int, int)  # EXACT
        >>> match_type(bool, int)  # SUBCLASS
        >>> match_type(list[int], Sequence[int])  # SUBCLASS
        >>> match_type(list[bool], list[int])  # None (no variance)
    """
    if is_any_type(sub) or is_any_type(sup):
        return None
    if sub == sup:
        return EXACT

    sub_origin = get_origin(sub) or sub
    sup_origin = get_origin(sup) or sup
    if not isinstance(sub_origin, type) or not isinstance(sup_origin, type):
        return None
    if get_args(sub) != get_args(sup):
        return None

    try:
        return SUBCLASS if issubclass(sub_origin, sup_origin) else None
    except TypeError:  # non-runtime protocols and other exotic classes
        return None
