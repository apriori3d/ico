"""Resource classes: loaded objects (models) declared as frozen dataclasses.

A resource is declared once, in a domain module, and used by reference::

    @resource("wan21.umt5", 1)
    @dataclass(frozen=True)
    class Umt5(Resource[UMT5EncoderModel]):
        '''UMT5-XXL text encoder (bf16, ~11 GB).'''
        weights: str = "wan21-umt5-xxl"      # an asset id

        def load(self) -> UMT5EncoderModel:
            return UMT5EncoderModel.from_pretrained(asset(self.weights) / "text_encoder")

    UMT5 = Umt5()

Fields are JSON params; the key of a resource is its registered reference
plus its fields, so two equal instances are one object in the pool.
Operators take resources as constructor params typed by their class and
call ``with self.encoder.use("cuda") as te: ...`` (see
:mod:`ico.resources.pool`).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from typing import Any, Generic, TypeVar, cast, get_type_hints

from ico.spec.model import JsonValue, is_json_value

T = TypeVar("T")
R = TypeVar("R", bound=type["Resource[Any]"])

META_ATTR = "ico_resource"


@dataclasses.dataclass(frozen=True, slots=True)
class ResourceMeta:
    name: str
    version: int

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"


class Resource(Generic[T]):
    """Base of resource classes (frozen dataclasses registered with
    :func:`resource`).

    Override :meth:`load` (build the object in RAM); override :meth:`to`
    when moving to a device is more than ``obj.to(device)`` (e.g. group
    offloading), :meth:`size` when the default (the bytes of the object's
    ``parameters()`` and ``buffers()``) does not apply, and :meth:`close`
    to release what garbage collection does not.
    """

    def load(self) -> T:
        raise NotImplementedError(f"{type(self).__name__}.load")

    def to(self, obj: T, device: str) -> T:
        """Move ``obj`` to ``device`` (``"cpu"`` is RAM); returns the moved
        object."""
        move = cast(Callable[[str], object] | None, getattr(obj, "to", None))
        if move is None:
            return obj
        moved = move(device)
        return cast(T, moved) if moved is not None else obj

    def size(self, obj: T) -> int | None:
        """Bytes the object takes on its device, or ``None`` if unknown."""
        total = 0
        found = False
        for attr in ("parameters", "buffers"):
            tensors = cast(
                Callable[[], Iterator[object]] | None, getattr(obj, attr, None)
            )
            if tensors is None:
                continue
            for tensor in cast(Iterator[Any], tensors()):
                total += int(tensor.numel()) * int(tensor.element_size())
                found = True
        return total if found else None

    def close(self, obj: T) -> None:
        """Release the object when the pool drops it."""

    # ─── Use ───

    def use(self, device: str = "cpu") -> AbstractContextManager[T]:
        """The loaded object on ``device`` for the ``with`` block; leaving
        it releases the resource to the pool (see
        :meth:`ResourcePool.use`)."""
        from ico.resources.pool import pool

        return pool().use(self, device)

    def preload(self) -> None:
        """Load into RAM now (e.g. while the GPU runs another step)."""
        from ico.resources.pool import pool

        pool().preload(self)

    # ─── Identity ───

    @property
    def ref(self) -> str:
        meta = resource_meta(type(self))
        if meta is None:
            raise TypeError(f"{type(self).__name__} is not registered with @resource")
        return meta.ref

    def params(self) -> dict[str, JsonValue]:
        """The fields, as JSON."""
        out: dict[str, JsonValue] = {}
        for f in dataclasses.fields(cast(Any, self)):
            value = cast(object, getattr(self, f.name))
            if not is_json_value(value):
                raise TypeError(
                    f"resource {self.ref}: field {f.name!r} = {value!r} is not JSON"
                )
            out[f.name] = cast(JsonValue, value)
        return out

    def assets(self) -> list[str]:
        """The values of its :data:`~ico.assets.AssetId` fields."""
        from ico.assets.ids import is_asset_hint

        hints = get_type_hints(type(self), include_extras=True)
        out: list[str] = []
        for f in dataclasses.fields(cast(Any, self)):
            value = cast(object, getattr(self, f.name))
            if is_asset_hint(hints.get(f.name)) and isinstance(value, str) and value:
                out.append(value)
        return out

    @property
    def key(self) -> str:
        """The identity in the pool: reference plus fields."""
        return f"{self.ref} {json.dumps(self.params(), sort_keys=True)}"

    @property
    def label(self) -> str:
        """The reference and the fields that tell this instance apart: those
        without a default and those set to another value, e.g.
        ``wan22.expert@1 (weights=wan22-i2v-high, storage=bf16)``."""
        shown: list[str] = []
        for f in dataclasses.fields(cast(Any, self)):
            value = cast(object, getattr(self, f.name))
            no_default = (
                f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING
            )
            if no_default or value != f.default:
                shown.append(f"{f.name}={value}")
        return f"{self.ref} ({', '.join(shown)})" if shown else self.ref

    def spec(self) -> dict[str, JsonValue]:
        """``{"ref": "name@version", "params": {...}}`` for specs and cards."""
        return {"ref": self.ref, "params": self.params()}


_registry: dict[str, type[Resource[Any]]] = {}


def resource(name: str, version: int) -> Callable[[R], R]:
    """Register a resource class under ``"name@version"``.

    The class must subclass :class:`Resource` and be a frozen dataclass
    (apply ``@dataclass(frozen=True)`` below this decorator).

    Raises:
        TypeError: Not a ``Resource`` subclass, or not a frozen dataclass.
        ValueError: A bad name or version, or the reference is taken by
            another class (re-registering a class of the same module and
            name replaces it, as when a module is reloaded).
    """
    if not name or "@" in name or any(c.isspace() for c in name):
        raise ValueError(f"invalid resource name {name!r}")
    if type(version) is not int or version < 1:
        raise ValueError(
            f"resource version must be a positive integer, got {version!r}"
        )
    meta = ResourceMeta(name, version)

    def decorator(cls: R) -> R:
        if not issubclass(cast(type, cls), Resource):  # untyped callers
            raise TypeError(f"@resource: {cls!r} must subclass Resource")
        params = (
            cast(Any, cls).__dataclass_params__
            if dataclasses.is_dataclass(cls)
            else None
        )
        if params is None or not params.frozen:
            raise TypeError(
                f"@resource: {cls.__name__} must be a frozen dataclass "
                "(@dataclass(frozen=True) below @resource)"
            )
        reserved = sorted(
            f.name
            for f in dataclasses.fields(cast(Any, cls))
            if hasattr(Resource, f.name)
        )
        if reserved:
            raise TypeError(
                f"@resource: {cls.__name__} has field(s) {', '.join(reserved)} that hide "
                "methods of Resource; rename them"
            )
        existing = _registry.get(meta.ref)
        if existing is not None and (existing.__module__, existing.__qualname__) != (
            cls.__module__,
            cls.__qualname__,
        ):
            raise ValueError(
                f"resource {meta.ref} is already {existing.__module__}.{existing.__qualname__}"
            )
        setattr(cls, META_ATTR, meta)
        _registry[meta.ref] = cls
        return cls

    return decorator


def resource_meta(cls: type) -> ResourceMeta | None:
    meta = cast(object, vars(cls).get(META_ATTR))
    return meta if isinstance(meta, ResourceMeta) else None


def resource_class(ref: str) -> type[Resource[Any]]:
    """The registered class of ``"name@version"``.

    Raises:
        KeyError: Not registered.
    """
    try:
        return _registry[ref]
    except KeyError:
        known = ", ".join(sorted(_registry)) or "none"
        raise KeyError(f"resource {ref!r} is not registered; known: {known}") from None


def unregister_resource(ref: str) -> None:
    cls = _registry.pop(ref)
    if META_ATTR in vars(cls):
        delattr(cls, META_ATTR)


def from_spec(value: JsonValue, expected: type[Resource[Any]]) -> Resource[Any]:
    """A resource from ``{"ref": ..., "params": {...}}``; the ref must name
    ``expected`` or a subclass of it.

    Raises:
        ValueError: A malformed value, an unknown reference, another class,
            or params the class does not take.
    """
    if (
        not isinstance(value, dict)
        or set(value) - {"ref", "params"}
        or "ref" not in value
    ):
        raise ValueError('a resource is {"ref": "name@version", "params": {...}}')
    ref, params = value["ref"], value.get("params", {})
    if not isinstance(ref, str) or not isinstance(params, dict):
        raise ValueError('a resource is {"ref": "name@version", "params": {...}}')
    try:
        cls = resource_class(ref)
    except KeyError as e:
        raise ValueError(str(e.args[0])) from None
    if not issubclass(cls, expected):
        wanted = resource_meta(expected)
        raise ValueError(
            f"{ref} is a {cls.__name__}, expected a {expected.__name__}"
            + (f" ({wanted.ref})" if wanted else "")
        )
    try:
        return cast(Callable[..., Resource[Any]], cls)(**params)
    except TypeError as e:
        raise ValueError(f"cannot create {ref} from params: {e}") from None


def spec_of(value: object) -> dict[str, JsonValue] | None:
    """The spec of ``value`` if it is a resource, else ``None``."""
    if isinstance(value, Resource):
        return cast(Resource[Any], value).spec()
    return None
