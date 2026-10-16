"""Machine-readable pipeline specs (JSON).

Example:
    >>> from ico.spec import to_spec, to_json
    >>> print(to_json(to_spec(flow)))
"""

from ico.spec.builders import BuildContext, SpecFactory, build, spec_builder
from ico.spec.model import SPEC_VERSION, JsonValue, NodeSpec, PipelineSpec, SpecError
from ico.spec.registry import (
    SpecContext,
    find_serializer,
    from_json,
    spec_serializer,
    to_json,
    to_spec,
)
from ico.spec.run import RunError, RunResult, run

__all__ = [
    "BuildContext",
    "build",
    "run",
    "spec_builder",
    "SPEC_VERSION",
    "JsonValue",
    "NodeSpec",
    "PipelineSpec",
    "RunError",
    "RunResult",
    "SpecContext",
    "SpecError",
    "SpecFactory",
    "find_serializer",
    "from_json",
    "spec_serializer",
    "to_json",
    "to_spec",
]
