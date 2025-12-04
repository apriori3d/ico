from collections.abc import Callable

from apriori.ico.core.node import IcoNode
from apriori.ico.tools.progress.types import ProgressProtocol


class IcoProgressTool:
    progress: ProgressProtocol

    def __init__(self, progress: ProgressProtocol) -> None:
        self.progress = progress


def assign_progress(
    progress: ProgressProtocol,
    node: IcoNode,
    progress_scheme: Callable[[IcoNode], None],
) -> None:
    pass
