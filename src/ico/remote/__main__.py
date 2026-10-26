"""``python -m ico.remote --workdir DIR``: run the job (JSON) on stdin and
write its events as JSON lines to stdout (see :mod:`ico.remote.runner`).

``python -m ico.remote --serve``: a job server, which runs the jobs of
the request lines on stdin one after another in this process (see
:mod:`ico.remote.server`).

The entry point lives here, not in the runner, which the package imports:
``python -m ico.remote.runner`` would run that module a second time.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import cast

from ico.remote.job import JsonObject, load_job
from ico.remote.runner import serve
from ico.remote.server import serve_jobs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m ico.remote")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--workdir", type=Path)
    mode.add_argument("--serve", action="store_true")
    args = parser.parse_args(argv)
    # The events own the original stdout; operators' output goes to stderr.
    with os.fdopen(os.dup(1), "w", encoding="utf-8") as events:
        os.dup2(2, 1)
        if args.serve:
            code = Path(tempfile.mkdtemp(prefix="ico-code-"))
            try:
                serve_jobs(sys.stdin, events, code)
            finally:
                shutil.rmtree(code, ignore_errors=True)
            return
        job = load_job(cast(JsonObject, json.loads(sys.stdin.read())))
        serve(job, cast(Path, args.workdir), events)


if __name__ == "__main__":
    main()
