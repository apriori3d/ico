from abc import ABC, abstractmethod

from apriori.ico.core.node import IcoNode
from apriori.ico.describe.plan.options import PlanRendererOptions
from apriori.ico.describe.plan.rich_renderer.render_target import PlanRenderTarget


class CustomRenderer(ABC):
    options: PlanRendererOptions

    def __init__(self, options: PlanRendererOptions) -> None:
        self.options = options

    @abstractmethod
    def render(self, plan: PlanRenderTarget, node: IcoNode) -> None: ...
