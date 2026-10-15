"""Operator registry: named, versioned operators with metadata and search."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypeVar, cast, overload

from ico.core.node import IcoNodeProtocol
from ico.core.operator import I, IcoOperatorProtocol, O, wrap_operator
from ico.core.signature import IcoSignature
from ico.registry.matching import EXACT, match_type
from ico.spec.model import JsonValue

N = TypeVar("N", bound=IcoNodeProtocol)

Status: TypeAlias = Literal["draft", "verified", "stable"]
STATUSES: tuple[Status, ...] = ("draft", "verified", "stable")

# Attribute under which a registered node carries its metadata.
META_ATTR = "ico_meta"


# ────────────────────────────────────────────────
# Metadata and entries
# ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OperatorMeta:
    """Registry metadata attached to a registered node as ``node.ico_meta``.

    Attributes:
        name: Operator name, unique together with ``version``.
        version: Positive integer version.
        tags: Free-form labels used by search (e.g. ``"text"``).
        caps: Capabilities the operator requires or provides
            (e.g. ``"network"``, ``"gpu"``).
        status: Review status: ``"draft"``, ``"verified"`` or ``"stable"``.
    """

    name: str
    version: int
    tags: frozenset[str] = frozenset()
    caps: frozenset[str] = frozenset()
    status: Status = "draft"

    @property
    def ref(self) -> str:
        """Registry reference ``"name@version"``."""
        return format_ref(self.name, self.version)


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """A registered operator.

    Attributes:
        meta: Registry metadata.
        node: The registered node (operator, source, sink or a whole flow).
        signature: Inferred ICO signature of the node.
        description: First paragraph of the docstring.
    """

    meta: OperatorMeta
    node: IcoNodeProtocol
    signature: IcoSignature
    description: str

    @property
    def ref(self) -> str:
        return self.meta.ref

    def card(self) -> dict[str, JsonValue]:
        """Compact JSON-compatible description of the operator."""
        return {
            "name": self.meta.name,
            "version": self.meta.version,
            "sig": self.signature.format(),
            "description": self.description,
            "tags": _json_strings(self.meta.tags),
            "caps": _json_strings(self.meta.caps),
            "status": self.meta.status,
        }


def _json_strings(items: frozenset[str]) -> list[JsonValue]:
    out: list[JsonValue] = []
    out.extend(sorted(items))
    return out


# ────────────────────────────────────────────────
# References
# ────────────────────────────────────────────────


def format_ref(name: str, version: int) -> str:
    """Format a registry reference ``"name@version"``."""
    return f"{name}@{version}"


def parse_ref(ref: str) -> tuple[str, int]:
    """Parse ``"name@version"`` into ``(name, version)``.

    Raises:
        ValueError: If the reference is malformed.
    """
    name, sep, version = ref.rpartition("@")
    if not sep or not version.isdigit():
        raise ValueError(
            f"Invalid registry reference {ref!r}, expected 'name@version'."
        )
    _validate_name(name)
    _validate_version(int(version))
    return name, int(version)


def _validate_name(name: str) -> None:
    if not name or "@" in name or any(c.isspace() for c in name):
        raise ValueError(
            f"Invalid operator name {name!r}: must be non-empty, "
            "without '@' or whitespace."
        )


def _validate_version(version: int) -> None:
    # Exact type check: rejects bool and non-int values coming from untyped callers.
    if type(version) is not int or version < 1:
        raise ValueError(
            f"Operator version must be a positive integer, got {version!r}."
        )


# ────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────


class IcoRegistry:
    """Store of registered operators keyed by ``"name@version"``.

    Example:
        >>> registry = IcoRegistry()
        >>> @register("tokenize", 1, tags={"text"}, registry=registry)
        ... def tokenize(text: str) -> list[str]:
        ...     '''Split text into words.'''
        ...     return text.split()
        >>> registry.get("tokenize@1").card()["sig"]  # 'str → list[str]'
        >>> [e.ref for e in registry.search(input=str)]  # ['tokenize@1']
    """

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    # ─── Registration ───

    def add(
        self,
        node: IcoNodeProtocol,
        meta: OperatorMeta,
        *,
        description: str | None = None,
        replace: bool = False,
    ) -> RegistryEntry:
        """Register a node under ``meta.ref`` and attach ``meta`` to it.

        Raises:
            ValueError: On invalid metadata, an uninferred signature, a
                duplicate reference (unless ``replace``), or a node already
                registered under another reference.
        """
        _validate_name(meta.name)
        _validate_version(meta.version)
        if meta.status not in STATUSES:
            raise ValueError(
                f"Operator status must be one of {STATUSES}, got {meta.status!r}."
            )

        existing_meta = get_meta(node)
        if existing_meta is not None and existing_meta.ref != meta.ref:
            raise ValueError(
                f"Node is already registered as '{existing_meta.ref}', "
                f"cannot register it as '{meta.ref}'."
            )
        if meta.ref in self._entries and not replace:
            raise ValueError(
                f"Operator '{meta.ref}' is already registered; "
                "bump the version or pass replace=True."
            )

        signature = node_signature(node)
        if signature is None:
            raise ValueError(
                f"Cannot register '{meta.ref}': its signature cannot be inferred. "
                "Add type annotations to the operator."
            )

        entry = RegistryEntry(
            meta=meta,
            node=node,
            signature=signature,
            description=description
            if description is not None
            else node_description(node),
        )
        setattr(node, META_ATTR, meta)
        self._entries[meta.ref] = entry
        return entry

    def unregister(self, ref: str) -> RegistryEntry:
        """Remove an entry and detach its metadata from the node.

        Raises:
            KeyError: If the reference is not registered.
        """
        entry = self._entries.pop(ref)
        if get_meta(entry.node) == entry.meta:
            delattr(entry.node, META_ATTR)
        return entry

    # ─── Lookup ───

    def get(self, ref: str) -> RegistryEntry:
        """Return the entry for ``"name@version"``.

        Raises:
            KeyError: If the reference is not registered.
        """
        try:
            return self._entries[ref]
        except KeyError:
            raise KeyError(f"Operator '{ref}' is not registered.") from None

    def versions(self, name: str) -> list[int]:
        """Registered versions of an operator, ascending."""
        return sorted(
            e.meta.version for e in self._entries.values() if e.meta.name == name
        )

    def cards(self) -> list[dict[str, JsonValue]]:
        """Compact cards of all entries, ordered by reference."""
        return [entry.card() for entry in self]

    def __contains__(self, ref: object) -> bool:
        return ref in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[RegistryEntry]:
        return iter(sorted(self._entries.values(), key=_entry_order))

    # ─── Search ───

    def search(
        self,
        *,
        input: Any = None,
        output: Any = None,
        tags: Collection[str] = (),
        text: str | None = None,
    ) -> list[RegistryEntry]:
        """Find operators by input type, output type, tags and text.

        Args:
            input: Type of the values to feed in; the operator input must
                accept it (exact match or a subclass).
            output: Type the result must be usable as (exact or subclass).
            tags: Every given tag must be present.
            text: Case-insensitive substring of the name or description.

        Returns:
            Matching entries: exact type matches first, then subclass
            matches, each ordered by reference.

        Note:
            Generic arguments must be equal (no variance): ``list[bool]``
            does not match ``list[int]``. See :func:`match_type`.
        """
        required_tags = frozenset(tags)
        needle = text.lower() if text else None

        ranked: list[tuple[int, RegistryEntry]] = []
        for entry in self._entries.values():
            if not required_tags <= entry.meta.tags:
                continue
            if needle is not None and not (
                needle in entry.meta.name.lower() or needle in entry.description.lower()
            ):
                continue

            rank = EXACT
            if input is not None:
                input_rank = match_type(input, entry.signature.i)
                if input_rank is None:
                    continue
                rank = max(rank, input_rank)
            if output is not None:
                output_rank = match_type(entry.signature.o, output)
                if output_rank is None:
                    continue
                rank = max(rank, output_rank)
            ranked.append((rank, entry))

        ranked.sort(key=lambda r: (r[0], *_entry_order(r[1])))
        return [entry for _, entry in ranked]


def _entry_order(entry: RegistryEntry) -> tuple[str, int]:
    return entry.meta.name, entry.meta.version


default_registry = IcoRegistry()
"""Process-wide registry used by :func:`register` unless another is given."""


# ────────────────────────────────────────────────
# Decorator
# ────────────────────────────────────────────────


class _Registrar:
    """Callable returned by :func:`register`; keeps precise static types."""

    def __init__(
        self,
        meta: OperatorMeta,
        registry: IcoRegistry,
        description: str | None,
        replace: bool,
    ) -> None:
        self._meta = meta
        self._registry = registry
        self._description = description
        self._replace = replace

    @overload
    def __call__(self, target: N) -> N: ...

    @overload
    def __call__(self, target: Callable[[I], O]) -> IcoOperatorProtocol[I, O]: ...

    def __call__(self, target: object) -> object:
        if isinstance(target, IcoNodeProtocol):
            node: IcoNodeProtocol = target
        elif callable(target):
            node = cast(
                IcoNodeProtocol, wrap_operator(cast(Callable[[Any], Any], target))
            )
        else:
            raise TypeError(
                f"Cannot register {target!r}: expected an ICO node or a callable."
            )

        self._registry.add(
            node, self._meta, description=self._description, replace=self._replace
        )
        return node


def register(
    name: str,
    version: int,
    *,
    tags: Collection[str] = (),
    caps: Collection[str] = (),
    status: Status = "draft",
    description: str | None = None,
    registry: IcoRegistry | None = None,
    replace: bool = False,
) -> _Registrar:
    """Register an operator (or any ICO node) under ``"name@version"``.

    Attaches :class:`OperatorMeta` to the node as ``node.ico_meta`` and adds
    it to ``registry`` (the :data:`default_registry` by default). The node is
    returned unchanged; a plain function is first wrapped as an IcoOperator.

    Args:
        name: Operator name (no ``@`` or whitespace).
        version: Positive integer version.
        tags: Labels for search.
        caps: Capabilities, e.g. ``"network"``.
        status: ``"draft"`` (default), ``"verified"`` or ``"stable"``.
        description: Overrides the docstring-based description, e.g. for
            composite flows that have no docstring.
        registry: Target registry; defaults to :data:`default_registry`.
        replace: Allow replacing an existing ``name@version`` entry.

    Example:
        >>> @register("tokenize", 1, tags={"text"})
        ... def tokenize(text: str) -> list[str]:
        ...     '''Split text into words.'''
        ...     return text.split()
    """
    meta = OperatorMeta(
        name=name,
        version=version,
        tags=frozenset(tags),
        caps=frozenset(caps),
        status=status,
    )
    target = registry if registry is not None else default_registry
    return _Registrar(meta, target, description, replace)


# ────────────────────────────────────────────────
# Node helpers
# ────────────────────────────────────────────────


def get_meta(node: object) -> OperatorMeta | None:
    """Registry metadata of a node, or ``None`` if it is not registered."""
    meta = cast(object, getattr(node, META_ATTR, None))
    return meta if isinstance(meta, OperatorMeta) else None


def node_signature(node: IcoNodeProtocol) -> IcoSignature | None:
    """Inferred signature of a node, or ``None``."""
    try:
        signature = cast(object, getattr(node, "signature", None))
    except Exception:  # signature inference must never break registration checks
        return None
    if isinstance(signature, IcoSignature) and signature.infered:
        return signature
    return None


# Attributes holding the user callable of leaf nodes, in lookup order.
_CALLABLE_ATTRS = ("_fn", "fn", "provider", "consumer")


def node_description(node: object) -> str:
    """First docstring paragraph of the user callable wrapped by a node."""
    for attr in _CALLABLE_ATTRS:
        fn = cast(object, getattr(node, attr, None))
        # Skip structural nodes' own methods (e.g. IcoSource._fn).
        if cast(object, getattr(fn, "__self__", None)) is node:
            continue
        if fn is not None and callable(fn):
            doc = inspect.getdoc(fn)
            if doc:
                return doc.strip().split("\n\n", 1)[0].replace("\n", " ")
    return ""
