from apriori.ico.core.runtime.progress.noop import NoOpProgress
from apriori.ico.core.runtime.progress.types import ProgressProtocol


class ProgressMixin:
    progress: ProgressProtocol

    def __init__(self) -> None:
        super().__init__()
        self.progress = NoOpProgress()
