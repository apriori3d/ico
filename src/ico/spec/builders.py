"""Spec → flow reconstruction through an operator registry.

Leaf operators are resolved by their ``op="name@version"`` reference; the
structural nodes around them are rebuilt from ``kind``, ``params`` and
``children``. Importable ``ref="module:qualname"`` references are never
imported: a spec may come from a worker, so only registered operators can
be instantiated.

Round trip ``to_spec(build(spec)) == spec`` holds for every buildable kind.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from ico.core.batcher import IcoBatcher
from ico.core.chain import chain
from ico.core.filter import IcoFilter
from ico.core.fork import IcoFork
from ico.core.identity import IcoIdentity
from ico.core.node import IcoNodeProtocol
from ico.core.operator import IcoOperatorProtocol
from ico.core.partition import IcoPartition
from ico.core.pipeline import IcoPipeline
from ico.core.process import IcoProcess
from ico.core.route import IcoRoute
from ico.core.signature import IcoSignature
from ico.core.stream import IcoStream
from ico.core.tap import IcoTap
from ico.core.tree_utils import TreePathIndex
from ico.spec.model import JsonValue, NodeSpec, PipelineSpec, SpecError

AnyOp: TypeAlias = IcoOperatorProtocol[Any, Any]

Builder: TypeAlias = Callable[[NodeSpec, "BuildContext"], IcoNodeProtocol]

_BUILDERS: dict[str, Builder] = {}


def spec_builder(kind: str) -> Callable[[Builder], Builder]:
    """Register a builder that turns a spec node of ``kind`` into a flow node.

    Example:
        >>> @spec_builder("my_node")
        ... def _my_node(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
        ...     return MyNode(ctx.op(spec.children[0]), name=spec.name)
    """

    def decorator(fn: Builder) -> Builder:
        _BUILDERS[kind] = fn
        return fn

    return decorator


if TYPE_CHECKING:
    from ico.registry.registry import IcoRegistry, OperatorMeta, RegistryEntry


# ────────────────────────────────────────────────
# Operator resolution
# ────────────────────────────────────────────────


@dataclass(frozen=True)
class _FunctionRef:
    """Picklable reference to a registered function.

    Functions are not pickled directly: ``@register`` and ``@operator()``
    replace the module attribute with an IcoOperator, so pickling the
    function by reference would fail. The module comes from a registry
    entry, never from a spec.
    """

    module: str
    qualname: str

    def resolve(self) -> Callable[[Any], Any]:
        import importlib

        from ico.core.operator import IcoOperator

        obj = cast(object, importlib.import_module(self.module))
        for part in self.qualname.split("."):
            obj = cast(object, getattr(obj, part))
        if isinstance(obj, IcoOperator):
            obj = cast(object, cast(IcoOperator[Any, Any], obj).fn)
        return cast(Callable[[Any], Any], obj)


@dataclass(frozen=True)
class _Target:
    """What an ``op`` reference resolves to.

    ``cls`` and ``fn`` can create fresh instances; ``node`` is the registered
    instance itself and is never pickled (see :class:`SpecFactory`).
    """

    meta: OperatorMeta
    signature: str
    cls: type | None = None
    fn: Callable[[Any], Any] | _FunctionRef | None = None
    node: IcoNodeProtocol | None = None

    def function(self) -> Callable[[Any], Any] | None:
        return self.fn.resolve() if isinstance(self.fn, _FunctionRef) else self.fn

    @classmethod
    def from_entry(cls, entry: RegistryEntry) -> _Target:
        return cls(
            meta=entry.meta,
            signature=entry.signature.format(),
            cls=entry.cls,
            fn=_plain_function(entry.node),
            node=entry.node,
        )

    def snapshot(self) -> _Target:
        """A copy that recreates the operator without the registered instance."""
        if self.cls is None and self.fn is None:
            raise SpecError(
                f"Operator '{self.meta.ref}' is a registered instance and cannot "
                "be recreated for every worker; register its parts or an "
                "operator class instead."
            )
        return _Target(
            self.meta, self.signature, cls=self.cls, fn=_function_ref(self.fn)
        )


def _plain_function(node: IcoNodeProtocol | None) -> Callable[[Any], Any] | None:
    """The user function of a plain IcoOperator, i.e. one rebuildable by wrapping."""
    from ico.core.operator import IcoOperator

    if node is None:
        return None
    node_type: type = type(node)
    if node_type is not IcoOperator:  # subclasses carry their own state
        return None
    fn = cast(IcoOperator[Any, Any], node).fn
    if cast(object, getattr(fn, "__self__", None)) is node:
        return None
    return fn


def _function_ref(
    fn: Callable[[Any], Any] | _FunctionRef | None,
) -> Callable[[Any], Any] | _FunctionRef | None:
    """Module-level functions travel by reference; others (local functions,
    lambdas) stay as objects and work in-process only."""
    if fn is None or isinstance(fn, _FunctionRef):
        return fn
    module = cast(object, getattr(fn, "__module__", None))
    qualname = cast(object, getattr(fn, "__qualname__", None))
    if (
        isinstance(module, str)
        and isinstance(qualname, str)
        and "<" not in qualname  # <locals>, <lambda>
    ):
        return _FunctionRef(module, qualname)
    return fn


Lookup: TypeAlias = Callable[[str], _Target]


class BuildContext:
    """Traversal state passed to builders.

    Attributes:
        lookup: Resolves ``"name@version"`` to what it refers to.
        fresh: Create new instances for every operator (used by
            :class:`SpecFactory` workers) instead of returning registered
            instances.

    Every :class:`SpecError` raised while building a node is located at that
    node's path in the spec tree (see :meth:`NodeSpec.at`).
    """

    __slots__ = ("lookup", "fresh", "_stack", "paths")

    def __init__(self, lookup: Lookup, *, fresh: bool = False) -> None:
        self.lookup = lookup
        self.fresh = fresh
        self._stack: list[tuple[NodeSpec, TreePathIndex]] = []
        # id(built node) -> its spec path, to locate errors raised at run time.
        self.paths: dict[int, TreePathIndex] = {}

    @property
    def path(self) -> TreePathIndex:
        """Spec-tree path of the node being built."""
        return self._stack[-1][1] if self._stack else TreePathIndex()

    def build(self, spec: NodeSpec) -> IcoNodeProtocol:
        """Build a node: registered operators by reference, others by kind."""
        path = self._child_path(spec)
        self._stack.append((spec, path))
        try:
            node = self._build(spec)
            self.paths.setdefault(id(node), path)
            return node
        except SpecError as e:
            if e.path is not None:
                raise  # already located at a descendant
            raise e.at(path) from None
        finally:
            self._stack.pop()

    def _child_path(self, spec: NodeSpec) -> TreePathIndex:
        if not self._stack:
            return TreePathIndex()
        parent, parent_path = self._stack[-1]
        # Identity, not equality: equal sibling specs are common in a seq.
        for index, child in enumerate(parent.children):
            if child is spec:
                return parent_path.add_child(index)
        return parent_path

    def _build(self, spec: NodeSpec) -> IcoNodeProtocol:
        if spec.op is not None:
            return self._resolve(spec)

        builder = _BUILDERS.get(spec.kind)
        if builder is None:
            where = f" '{spec.name}'" if spec.name else ""
            if spec.kind == "op" or spec.ref is not None:
                raise SpecError(
                    f"Cannot build leaf{where} ({spec.ref or 'anonymous'}): only "
                    "registered operators (op='name@version') can be built."
                )
            raise SpecError(f"Cannot build node{where} of kind '{spec.kind}' yet.")
        return builder(spec, self)

    def op(self, spec: NodeSpec) -> AnyOp:
        """Build a child that is used as an operator."""
        return cast(AnyOp, self.build(spec))

    def factory(self, worker: NodeSpec) -> SpecFactory:
        """A picklable factory creating fresh copies of a worker subtree."""
        refs = sorted(_op_refs(worker))
        try:
            targets = tuple(self._target(ref).snapshot() for ref in refs)
        except SpecError as e:
            raise e.at(self._child_path(worker)) from None
        return SpecFactory(worker, targets)

    def _target(self, ref: str) -> _Target:
        from ico.registry.registry import parse_ref

        try:
            parse_ref(ref)
            return self.lookup(ref)
        except (ValueError, KeyError) as e:
            raise SpecError(f"Cannot resolve operator '{ref}': {e.args[0]}") from None

    def _resolve(self, spec: NodeSpec) -> IcoNodeProtocol:
        from ico.core.operator import IcoOperator
        from ico.registry.registry import META_ATTR

        ref = cast(str, spec.op)
        target = self._target(ref)

        if target.cls is not None:
            # Constructor kwargs are validated by the class itself.
            create = cast(Callable[..., IcoNodeProtocol], target.cls)
            try:
                node: IcoNodeProtocol = create(**spec.params)
            except TypeError as e:
                raise SpecError(f"Cannot create '{ref}' from params: {e}") from None
        else:
            if spec.params:
                raise SpecError(f"Operator '{ref}' takes no params.")
            if self.fresh:
                fn = target.snapshot().function()
                fresh_op: IcoOperator[Any, Any] = IcoOperator(
                    cast(Callable[[Any], Any], fn), name=spec.name
                )
                # Keep the reference so that the rebuilt flow exports the same spec.
                setattr(fresh_op, META_ATTR, target.meta)
                node = fresh_op
            else:
                node = cast(IcoNodeProtocol, target.node)

        actual = _signature_text(node, target.signature)
        if spec.sig is not None and spec.sig != actual:
            raise SpecError(
                f"Signature of '{ref}' changed: spec has {spec.sig!r}, "
                f"registry has {actual!r}."
            )
        return node


def _op_refs(spec: NodeSpec) -> set[str]:
    refs = {spec.op} if spec.op is not None else set[str]()
    for child in spec.children:
        refs |= _op_refs(child)
    return refs


@dataclass(frozen=True)
class SpecFactory:
    """Picklable worker factory rebuilt from a spec subtree.

    Used by :func:`build` for factory-based ``IcoAsyncStream`` pools and for
    workers. Every call builds a fresh worker: functions are wrapped anew and
    operator classes are instantiated with ``cls(**params)``. The operators
    it needs travel with it (functions and classes pickle by reference), so
    it also works in a spawned worker process without the parent registry.

    Attributes:
        worker: Spec of the worker flow.
        targets: Operators referenced by the worker.
    """

    worker: NodeSpec
    targets: tuple[_Target, ...]

    # Annotated with the untyped node protocol on purpose: workers infer their
    # signature from the factory's return annotation, and an operator type
    # here would claim "Any → Any" as inferred. build() parametrizes the
    # worker from probe_signature() instead.
    def __call__(self) -> IcoNodeProtocol:
        by_ref = {t.meta.ref: t for t in self.targets}
        return BuildContext(by_ref.__getitem__, fresh=True).build(self.worker)

    def probe_signature(self) -> IcoSignature:
        """Signature of the worker flow, from one probe worker."""
        return cast(AnyOp, self()).signature


def build(
    spec: PipelineSpec | NodeSpec,
    registry: IcoRegistry | None = None,
    *,
    params: Mapping[str, JsonValue] | None = None,
) -> IcoNodeProtocol:
    """Rebuild a flow from a spec, resolving operators through a registry.

    Args:
        spec: A spec document or a single node spec.
        registry: Registry for ``op`` references; defaults to
            :data:`ico.registry.default_registry`.
        params: Values of the pipeline parameters (``{"$param": ...}``
            references, see :mod:`ico.spec.params`); defaults apply to the
            others.

    Returns:
        The rebuilt flow. Registered operator instances are returned as is
        (not copied); operator classes are instantiated per node. Workers of
        factory-based async streams and workers are created by a
        :class:`SpecFactory`, so each worker is a distinct instance.

    Raises:
        SpecError: On unknown operators, signature drift between the spec
            and the registry, unregistered leaves, invalid params, registered
            instances inside factory-made workers, or kinds that cannot be
            built yet (epoch, context operators, runtime wrappers, monitors).

    Example:
        >>> flow = build(from_json(text))
    """
    from ico.spec.params import bind

    spec = bind(spec, params)
    root = spec.root if isinstance(spec, PipelineSpec) else spec
    return registry_context(registry).build(root)


def registry_context(
    registry: IcoRegistry | None = None, *, fresh: bool = False
) -> BuildContext:
    """A build context resolving ``op`` references through ``registry``.

    Use it instead of :func:`build` to keep ``BuildContext.paths`` after a
    build, e.g. to locate errors raised while the flow runs. ``fresh``
    creates new instances of registered functions too, instead of using the
    registered nodes (e.g. for a flow that will be instrumented).
    """
    from ico.registry.registry import default_registry

    target = registry if registry is not None else default_registry

    def lookup(ref: str) -> _Target:
        return _Target.from_entry(target.get(ref))

    return BuildContext(lookup, fresh=fresh)


def _signature_text(node: IcoNodeProtocol, fallback: str) -> str:
    from ico.spec.serializers import node_signature

    return node_signature(node) or fallback


# ────────────────────────────────────────────────
# Param helpers
# ────────────────────────────────────────────────


def _param(spec: NodeSpec, key: str, kind: type, *, optional: bool = False) -> Any:
    if key not in spec.params:
        if optional:
            return None
        raise SpecError(f"'{spec.kind}' requires param '{key}'.")
    value = spec.params[key]
    # bool is an int subclass; require exact types for numeric params.
    if type(value) is not kind:
        raise SpecError(
            f"Param '{key}' of '{spec.kind}' must be {kind.__name__}, got {value!r}."
        )
    return value


def _children(spec: NodeSpec, count: int | None = None, *, at_least: int = 0) -> None:
    n = len(spec.children)
    if (count is not None and n != count) or n < at_least:
        expected = count if count is not None else f"at least {at_least}"
        raise SpecError(f"'{spec.kind}' expects {expected} children, got {n}.")


# ────────────────────────────────────────────────
# Builders
# ────────────────────────────────────────────────


@spec_builder("seq")
def build_seq(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, at_least=2)
    from ico.core.signature_check import chain_links, check_link
    from ico.core.signature_utils import format_ico_type

    steps = [ctx.op(child) for child in spec.children]
    # Check every link before composing, so that a spec written by a worker
    # fails with the location, references and types of the bad step.
    for left, position in chain_links(steps):
        issue = check_link(steps[left], steps[position])
        if issue is not None and issue.is_error:
            expected, got = format_ico_type(issue.input), format_ico_type(issue.output)
            raise SpecError(
                f"expected {expected}, got {got}",
                path=ctx.path.add_child(position),
                expected=expected,
                got=got,
                refs=(issue.left, issue.right),
            )
    node = cast(IcoNodeProtocol, reduce(chain, steps))
    if spec.name is not None:
        node.name = spec.name
    return node


@spec_builder("stream")
def build_stream(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, 1)
    return IcoStream(ctx.op(spec.children[0]), name=spec.name)


@spec_builder("pipeline")
def build_pipeline(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, at_least=1)
    return IcoPipeline(*(ctx.op(c) for c in spec.children), name=spec.name)


@spec_builder("process")
def build_process(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    if "num_iterations" in spec.params:
        _children(spec, 1)
        return IcoProcess(
            ctx.op(spec.children[0]),
            num_iterations=_param(spec, "num_iterations", int),
            name=spec.name,
        )

    _children(spec, 2)
    on_limit = _param(spec, "on_limit", str)
    if on_limit not in ("return", "raise"):
        raise SpecError(f"Invalid on_limit {on_limit!r} in 'process'.")
    return IcoProcess(
        ctx.op(spec.children[0]),
        until=ctx.op(spec.children[1]),
        max_iterations=_param(spec, "max_iterations", int),
        on_limit=on_limit,
        name=spec.name,
    )


@spec_builder("fork")
def build_fork(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, at_least=2)
    node: IcoFork[Any, Any] = IcoFork(
        *(ctx.op(c) for c in spec.children), name=spec.name
    )
    return node


@spec_builder("route")
def build_route(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    keys = cast(list[JsonValue], _param(spec, "keys", list))
    has_default = _param(spec, "default", bool)
    _children(spec, 1 + len(keys) + int(has_default))

    key_op, *branches = (ctx.op(c) for c in spec.children)
    default = branches.pop() if has_default else None
    for key in keys:
        if isinstance(key, list | dict):
            raise SpecError(f"Route key {key!r} is not a JSON scalar.")
    return IcoRoute(
        key_op,
        dict(zip(keys, branches, strict=True)),
        default=default,
        name=spec.name,
    )


@spec_builder("filter")
def build_filter(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, 1)
    return IcoFilter(ctx.op(spec.children[0]), name=spec.name)


@spec_builder("tap")
def build_tap(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, 1)
    return IcoTap(ctx.op(spec.children[0]), name=spec.name)


@spec_builder("partition")
def build_partition(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, 1)
    return IcoPartition(ctx.op(spec.children[0]), name=spec.name)


@spec_builder("batcher")
def build_batcher(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, 0)
    # Not parametrized (IcoBatcher[T]): a spec carries no Python types, and an
    # explicit parameter would change the inferred signature.
    node: IcoBatcher[Any] = IcoBatcher(
        batch_size=_param(spec, "batch_size", int),
        drop_last=_param(spec, "drop_last", bool),
        name=spec.name,
    )
    return node


@spec_builder("progress")
def build_progress(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    from ico.core.runtime.progress import IcoProgress

    _children(spec, 0)
    # Not parametrized, like batcher and identity: specs carry no Python types.
    # A null total (e.g. a parameter's default) is an unknown count.
    has_total = spec.params.get("total") is not None
    total = _number(spec, "total") if has_total else None
    try:
        node: IcoProgress[Any] = IcoProgress(total=total, name=spec.name)
    except ValueError as e:
        raise SpecError(str(e)) from None
    return node


@spec_builder("identity")
def build_identity(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, 0)
    node: IcoIdentity[Any] = IcoIdentity(name=spec.name)
    return node


@spec_builder("async_stream")
def build_async_stream(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    from ico.core.async_stream import IcoAsyncStream

    pool_size = _param(spec, "pool_size", int)
    ordered = _param(spec, "ordered", bool)
    on_error = _param(spec, "on_error", str, optional=True) or "raise"
    if on_error not in ("raise", "skip", "collect"):
        raise SpecError(f"Invalid on_error {on_error!r} in 'async_stream'.")
    options: dict[str, Any] = {
        "ordered": ordered,
        "on_error": on_error,
        "retry": _retry_policy(spec),
        "name": spec.name,
    }

    if _param(spec, "factory", bool):
        # One worker spec; every pool slot gets a fresh worker from it.
        _children(spec, 1)
        factory = ctx.factory(spec.children[0])
        # Not parametrized: the stream infers its signature from its workers.
        # SpecFactory returns an operator protocol; the pool accepts any worker.
        stream: IcoAsyncStream[Any, Any] = IcoAsyncStream(
            cast(Callable[[], Any], factory), pool_size=pool_size, **options
        )
        return stream

    _children(spec, pool_size)
    workers = [cast(Any, ctx.build(child)) for child in spec.children]
    stream = IcoAsyncStream(workers, **options)
    return stream


def _retry_policy(spec: NodeSpec) -> Any:
    import builtins

    from ico.core.async_stream import RetryPolicy

    raw = _param(spec, "retry", dict, optional=True)
    if raw is None:
        return None
    retry = NodeSpec(kind="async_stream.retry", params=cast(dict[str, JsonValue], raw))

    exceptions: list[type[Exception]] = []
    for ref in cast(list[JsonValue], _param(retry, "retry_on", list)):
        # Only builtin exception types: never import a module named by a spec.
        module, _, name = str(ref).partition(":")
        exc = (
            cast(object, getattr(builtins, name, None))
            if module == "builtins"
            else None
        )
        if not (isinstance(exc, type) and issubclass(exc, Exception)):
            raise SpecError(
                f"Unsupported retry_on {ref!r}: only builtin exception types "
                "(e.g. 'builtins:TimeoutError') can be restored from a spec."
            )
        exceptions.append(exc)

    try:
        return RetryPolicy(
            max_retries=_param(retry, "max_retries", int),
            delay=_number(retry, "delay"),
            backoff=_number(retry, "backoff"),
            retry_on=tuple(exceptions),
        )
    except ValueError as e:
        raise SpecError(f"Invalid retry policy: {e}") from None


def _number(spec: NodeSpec, key: str) -> float:
    value = spec.params.get(key)
    if type(value) not in (int, float):
        raise SpecError(
            f"Param '{key}' of '{spec.kind}' must be a number, got {value!r}."
        )
    return cast(float, value)


@spec_builder("worker")
def build_worker(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    from ico.runtime.worker.mp.mp_worker import MPWorker

    # Only MPWorker can be rebuilt for now (YAGNI: no worker allowlist yet).
    expected = f"{MPWorker.__module__}:{MPWorker.__qualname__}"
    worker_type = _param(spec, "type", str)
    if worker_type != expected:
        raise SpecError(
            f"Cannot build worker of type {worker_type!r}: only {expected!r} is supported."
        )

    _children(spec, 1)
    start_timeout = _start_timeout(spec)
    factory = ctx.factory(spec.children[0])
    # Parametrize with the worker types so the worker signature is inferred,
    # as it was from the original factory's return annotation.
    signature = factory.probe_signature()
    if signature.infered:
        return MPWorker[signature.i, signature.o](  # type: ignore[name-defined]
            cast(Callable[[], AnyOp], factory),
            name=spec.name,
            start_timeout=start_timeout,
        )
    # Unparametrized, like the original worker: MPWorker[Any, Any] would report
    # an inferred "Any → Any" signature and break the round trip.
    worker: MPWorker[Any, Any] = MPWorker(
        cast(Callable[[], AnyOp], factory),
        name=spec.name,
        start_timeout=start_timeout,
    )
    return worker


def _start_timeout(spec: NodeSpec) -> float | None:
    """``start_timeout`` of a worker: a positive number, or ``null`` for no
    limit; ``MPWorker``'s default when absent."""
    from ico.runtime.worker.mp.mp_worker import DEFAULT_START_TIMEOUT

    if "start_timeout" not in spec.params:
        return DEFAULT_START_TIMEOUT
    value = spec.params["start_timeout"]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise SpecError(
            f"Param 'start_timeout' of 'worker' must be a positive number or "
            f"null, got {value!r}."
        )
    return float(value)
