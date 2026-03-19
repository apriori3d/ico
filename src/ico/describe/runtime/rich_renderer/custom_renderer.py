from abc import ABC, abstractmethod

from ico.core.runtime.node import IcoRuntimeNode
from ico.describe.runtime.options import RuntimeRendererOptions
from ico.describe.runtime.rich_renderer.render_target import (
    RuntimeRenderTarget,
)


class RuntimeCustomRenderer(ABC):
    """
    Base class for specialized runtime node rendering.

    Use for runtime nodes requiring custom display logic
    beyond standard row rendering.
    """

    options: RuntimeRendererOptions

    def __init__(self, options: RuntimeRendererOptions) -> None:
        """Initialize runtime custom renderer with options."""
        self.options = options

    @abstractmethod
    def render(self, runtime_tree: RuntimeRenderTarget, node: IcoRuntimeNode) -> None:
        """Implement custom rendering logic for specialized runtime nodes."""
        ...
