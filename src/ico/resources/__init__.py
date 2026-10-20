"""Resources: loaded objects (models) shared through a pool per process.

See :mod:`ico.resources.base` (declaring) and :mod:`ico.resources.pool`
(using), and ``docs/proposals/2026-10-20-assets-and-resources.md``.
"""

from ico.resources.base import (
    Resource,
    ResourceMeta,
    from_spec,
    resource,
    resource_class,
    resource_meta,
    unregister_resource,
)
from ico.resources.pool import Held, PoolEvent, ResourcePool, parse_size, pool, set_pool

__all__ = [
    "Held",
    "PoolEvent",
    "Resource",
    "ResourceMeta",
    "ResourcePool",
    "from_spec",
    "parse_size",
    "pool",
    "resource",
    "resource_class",
    "resource_meta",
    "set_pool",
    "unregister_resource",
]
