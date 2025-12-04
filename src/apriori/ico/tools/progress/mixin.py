from apriori.ico.tools.progress.noop import NoOpProgress
from apriori.ico.tools.progress.types import ProgressProtocol


class ProgressMixin:
    progress: ProgressProtocol
    task: int | None

    def __init__(self) -> None:
        self.progress = NoOpProgress()
        self.task = None
