"""Operator registry: named, versioned ICO operators with metadata and search.

Example:
    >>> from ico.registry import register, default_registry
    >>> @register("tokenize", 1, tags={"text"})
    ... def tokenize(text: str) -> list[str]:
    ...     '''Split text into words.'''
    ...     return text.split()
    >>> default_registry.search(input=str, tags={"text"})
"""

from ico.registry.matching import EXACT, SUBCLASS, MatchRank, match_type
from ico.registry.registry import (
    STATUSES,
    IcoRegistry,
    OperatorMeta,
    RegistryEntry,
    Status,
    default_registry,
    format_ref,
    get_meta,
    parse_ref,
    register,
)

__all__ = [
    "EXACT",
    "STATUSES",
    "SUBCLASS",
    "IcoRegistry",
    "MatchRank",
    "OperatorMeta",
    "RegistryEntry",
    "Status",
    "default_registry",
    "format_ref",
    "get_meta",
    "match_type",
    "parse_ref",
    "register",
]
