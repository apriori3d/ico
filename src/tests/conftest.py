import multiprocessing
from typing import Any


def pytest_sessionstart(session: Any) -> None:
    multiprocessing.set_start_method("spawn", force=True)
