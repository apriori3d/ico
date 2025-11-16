from typing import Any

from apriori.ico.core.runtime.progress.noop import NoOpProgress
from apriori.ico.core.runtime.progress.types import ProgressProtocol


class ProgressMixin:
    progress: ProgressProtocol

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.progress = NoOpProgress()
