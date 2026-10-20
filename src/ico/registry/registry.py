"""Operator registry: named, versioned operators with metadata and search."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypeVar, cast, get_type_hints, overload

from ico.core.node import IcoNodeProtocol
from ico.core.operator import I, IcoOperator, IcoOperatorProtocol, O, wrap_operator
from ico.core.signature import IcoSignature
from ico.registry.matching import EXACT, match_type
from ico.resources.base import Resource, resource_meta, spec_of
from ico.spec.model import JsonValue, is_json_value

N = TypeVar("N", bound=IcoNodeProtocol)
C = TypeVar("C", bound=type[IcoOperator[Any, Any]])

Status: TypeAlias = Literal["draft", "verified", "stable"]
ResourceClass: TypeAlias = type[Resource[Any]]
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
class ParamInfo:
    """A constructor parameter of a registered operator class.

    Attributes:
        name: Parameter name (also the attribute holding its value).
        type: Formatted annotation, or ``None`` when not annotated.
        required: Whether the parameter has no default.
        default: Default value (meaningless when ``required``).
        resource: The resource class of a parameter annotated with one
            (see :mod:`ico.resources`); specs give it as
            ``{"ref": "name@version", "params": {...}}``.
    """

    name: str
    type: str | None
    required: bool
    default: Any = None
    resource: ResourceClass | None = None

    def card(self) -> dict[str, JsonValue]:
        out: dict[str, JsonValue] = {"name": self.name, "required": self.required}
        if self.type is not None:
            out["type"] = self.type
        if self.resource is not None:
            meta = resource_meta(self.resource)
            if meta is not None:
                out["resource"] = meta.ref
        spec = spec_of(cast(object, self.default))
        if not self.required and spec is not None:
            out["default"] = spec
        elif not self.required and is_json_value(self.default):
            out["default"] = cast(JsonValue, self.default)
        return out


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """A registered operator: a node instance or a parametric operator class.

    Attributes:
        meta: Registry metadata.
        node: The registered node (operator, source, sink or a whole flow),
            or ``None`` for a registered class.
        cls: The registered operator class, or ``None`` for a node. Instances
            are created with ``cls(**params)``.
        signature: Inferred ICO signature.
        description: First paragraph of the docstring.
        params: Constructor parameters of a registered class.
        module: Module that defines the registered function or class, used
            to find the entries of a module when it is reloaded; ``None``
            when unknown (e.g. a composite flow).
    """

    meta: OperatorMeta
    node: IcoNodeProtocol | None
    cls: type[IcoOperator[Any, Any]] | None
    signature: IcoSignature
    description: str
    params: tuple[ParamInfo, ...] = ()
    module: str | None = None

    @property
    def ref(self) -> str:
        return self.meta.ref

    def card(self) -> dict[str, JsonValue]:
        """Compact JSON-compatible description of the operator."""
        card: dict[str, JsonValue] = {
            "name": self.meta.name,
            "version": self.meta.version,
            "sig": self.signature.format(),
            "description": self.description,
            "tags": _json_strings(self.meta.tags),
            "caps": _json_strings(self.meta.caps),
            "status": self.meta.status,
        }
        if self.cls is not None:
            params: list[JsonValue] = []
            params.extend(p.card() for p in self.params)
            card["params"] = params
        return card


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

    def _check_new(self, target: object, meta: OperatorMeta, replace: bool) -> None:
        _validate_name(meta.name)
        _validate_version(meta.version)
        if meta.status not in STATUSES:
            raise ValueError(
                f"Operator status must be one of {STATUSES}, got {meta.status!r}."
            )

        existing_meta = get_meta(target)
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
        self._check_new(node, meta, replace)

        signature = node_signature(node)
        if signature is None:
            raise ValueError(
                f"Cannot register '{meta.ref}': its signature cannot be inferred. "
                "Add type annotations to the operator."
            )

        entry = RegistryEntry(
            meta=meta,
            node=node,
            cls=None,
            signature=signature,
            description=description
            if description is not None
            else node_description(node),
            module=node_module(node),
        )
        setattr(node, META_ATTR, meta)
        self._entries[meta.ref] = entry
        return entry

    def add_class(
        self,
        cls: type[IcoOperator[Any, Any]],
        meta: OperatorMeta,
        *,
        description: str | None = None,
        replace: bool = False,
    ) -> RegistryEntry:
        """Register a parametric operator class under ``meta.ref``.

        Instances are created with ``cls(**params)``. Following the sklearn
        convention, every ``__init__`` parameter must be stored in an
        attribute with the same name, so that a spec can capture it.

        Raises:
            ValueError: On invalid metadata, ``*args``/``**kwargs`` in
                ``__init__``, a signature that cannot be inferred from the
                generic base (e.g. ``IcoOperator[float, bool]``), or a
                duplicate reference (unless ``replace``).
        """
        self._check_new(cls, meta, replace)

        params = init_params(cls)
        signature = class_signature(cls)
        if signature is None:
            raise ValueError(
                f"Cannot register '{meta.ref}': its signature cannot be inferred. "
                "Subclass a parametrized base such as IcoOperator[float, bool]."
            )

        own_doc = cast(object, vars(cls).get("__doc__"))
        entry = RegistryEntry(
            meta=meta,
            node=None,
            cls=cls,
            signature=signature,
            description=description
            if description is not None
            else _first_paragraph(own_doc if isinstance(own_doc, str) else None),
            params=params,
            module=cls.__module__,
        )
        setattr(cls, META_ATTR, meta)
        self._entries[meta.ref] = entry
        return entry

    def unregister(self, ref: str) -> RegistryEntry:
        """Remove an entry and detach its metadata from the node or class.

        Raises:
            KeyError: If the reference is not registered.
        """
        entry = self._entries.pop(ref)
        target = entry.cls if entry.cls is not None else entry.node
        if get_meta(target) == entry.meta:
            delattr(target, META_ATTR)
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
    def __call__(self, target: C) -> C: ...

    @overload
    def __call__(self, target: Callable[[I], O]) -> IcoOperatorProtocol[I, O]: ...

    def __call__(self, target: object) -> object:
        if isinstance(target, type):
            if not issubclass(target, IcoOperator):
                raise TypeError(
                    f"Cannot register class {target.__name__}: only IcoOperator "
                    "subclasses can be registered as parametric operators."
                )
            operator_cls = cast(type[IcoOperator[Any, Any]], target)
            self._registry.add_class(
                operator_cls,
                self._meta,
                description=self._description,
                replace=self._replace,
            )
            return operator_cls
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

    An ``IcoOperator`` subclass can be registered as a parametric operator:
    specs then store its constructor kwargs and :func:`ico.spec.build` creates
    instances with ``cls(**params)``. Every ``__init__`` parameter must be
    stored in an attribute of the same name (sklearn convention), and heavy
    initialization (loading models, opening connections) should be lazy or
    happen in ``on_activate``, not in ``__init__``: instances are created
    whenever a flow is built, e.g. once per worker.

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


def get_meta(target: object) -> OperatorMeta | None:
    """Registry metadata of a node or class, or ``None`` if not registered.

    A node is registered when it was registered itself or when its exact
    class is a registered operator class (subclasses are not).
    """
    meta = _own_meta(target)
    if meta is None and not isinstance(target, type):
        meta = _own_meta(type(target))
    return meta


def _own_meta(target: object) -> OperatorMeta | None:
    namespace = cast(object, getattr(target, "__dict__", None))
    if namespace is None:
        return None
    meta = cast(object, cast(Any, namespace).get(META_ATTR))
    return meta if isinstance(meta, OperatorMeta) else None


def registered_class(node: object) -> type[IcoOperator[Any, Any]] | None:
    """The registered operator class of an instance, or ``None``."""
    if _own_meta(node) is not None or _own_meta(type(node)) is None:
        return None
    return cast(type[IcoOperator[Any, Any]], type(node))


def init_params(cls: type) -> tuple[ParamInfo, ...]:
    """Constructor parameters of an operator class (without ``self``).

    Raises:
        ValueError: If ``__init__`` takes ``*args`` or ``**kwargs``: such
            parameters cannot be captured by name.
    """
    from ico.core.signature_utils import format_ico_type

    signature = inspect.signature(cls.__init__)
    try:
        hints = get_type_hints(cls.__init__)
    except Exception:  # unresolvable annotations: keep them unformatted
        hints = {}

    params: list[ParamInfo] = []
    for param in list(signature.parameters.values())[1:]:
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ValueError(
                f"Cannot register {cls.__name__}: __init__ must not take "
                f"*args or **kwargs (found '{param.name}')."
            )
        hint = cast(object, hints.get(param.name))
        required = param.default is param.empty
        resource = (
            cast(type[Resource[Any]], hint)
            if isinstance(hint, type) and issubclass(hint, Resource)
            else None
        )
        params.append(
            ParamInfo(
                name=param.name,
                type=format_ico_type(hint) if hint is not None else None,
                required=required,
                default=None if required else cast(object, param.default),
                resource=resource,
            )
        )
    return tuple(params)


def class_signature(cls: type[IcoOperator[Any, Any]]) -> IcoSignature | None:
    """Signature of an operator class from its generic base, without __init__."""
    from ico.core.signature_utils import resolve_types_from_generic, wrap_type_if_any

    probe = cls.__new__(cls)  # not initialized: only the class hierarchy is used
    i_type, o_type = resolve_types_from_generic(probe, IcoOperator, I, O)
    if i_type is None or o_type is None:
        return None
    return IcoSignature(i=wrap_type_if_any(i_type), c=None, o=wrap_type_if_any(o_type))


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


def node_module(node: object) -> str | None:
    """Module of the user function wrapped by a plain operator, or ``None``."""
    if type(node) is not IcoOperator:
        return None
    fn = cast(object, cast(IcoOperator[Any, Any], node).fn)
    if cast(object, getattr(fn, "__self__", None)) is node:
        return None
    module = cast(object, getattr(fn, "__module__", None))
    return module if isinstance(module, str) else None


def node_description(node: object) -> str:
    """First docstring paragraph of the user callable wrapped by a node."""
    for attr in _CALLABLE_ATTRS:
        fn = cast(object, getattr(node, attr, None))
        # Skip structural nodes' own methods (e.g. IcoSource._fn).
        if cast(object, getattr(fn, "__self__", None)) is node:
            continue
        if fn is not None and callable(fn):
            return _first_paragraph(inspect.getdoc(fn))
    return ""


def _first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    return inspect.cleandoc(doc).split("\n\n", 1)[0].replace("\n", " ")
