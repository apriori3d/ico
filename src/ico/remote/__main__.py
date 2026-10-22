"""``python -m ico.remote --workdir DIR``: run the job (JSON) on stdin and
write its events as JSON lines to stdout (see :mod:`ico.remote.runner`).

The entry point lives here, not in the runner, which the package imports:
``python -m ico.remote.runner`` would run that module a second time.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import cast

from ico.remote.job import JsonObject, load_job
from ico.remote.runner import serve


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m ico.remote")
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args(argv)
    # The events own the original stdout; operators' output goes to stderr.
    with os.fdopen(os.dup(1), "w", encoding="utf-8") as events:
        os.dup2(2, 1)
        job = load_job(cast(JsonObject, json.loads(sys.stdin.read())))
        serve(job, cast(Path, args.workdir), events)


if __name__ == "__main__":
    main()
