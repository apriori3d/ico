from abc import ABC, abstractmethod

from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.describe.runtime.options import RuntimeRendererOptions
from apriori.ico.describe.runtime.rich_renderer.render_target import (
    RuntimeTreeRenderTarget,
)


class RuntimeCustomRenderer(ABC):
    options: RuntimeRendererOptions

    def __init__(self, options: RuntimeRendererOptions) -> None:
        self.options = options

    @abstractmethod
    def render(
        self, runtime_tree: RuntimeTreeRenderTarget, node: IcoRuntimeNode
    ) -> None: ...
