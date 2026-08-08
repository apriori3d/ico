from abc import ABC, abstractmethod

from ico.core.node import IcoNodeProtocol
from ico.describe.plan.options import PlanRendererOptions
from ico.describe.plan.rich_renderer.render_target import PlanRenderTarget


class CustomRenderer(ABC):
    """
    Base class for specialized node rendering with full control.

    Use for complex nodes requiring custom layout beyond standard
    row/group rendering (e.g., IcoEpoch with iteration details).
    """

    options: PlanRendererOptions

    def __init__(self, options: PlanRendererOptions) -> None:
        """Initialize custom renderer with options."""
        self.options = options

    @abstractmethod
    def render(self, plan: PlanRenderTarget, node: IcoNodeProtocol) -> None:
        """Implement custom rendering logic for specialized nodes."""
        ...
