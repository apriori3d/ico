from abc import ABC, abstractmethod

from apriori.ico.core.node import IcoNode
from apriori.ico.describe.plan.options import RenderOptions
from apriori.ico.describe.plan.rich_render.render_target import RenderTarget


class CustomRenderer(ABC):
    options: RenderOptions

    def __init__(self, options: RenderOptions) -> None:
        self.options = options

    @abstractmethod
    def render(self, plan: RenderTarget, node: IcoNode) -> None: ...
