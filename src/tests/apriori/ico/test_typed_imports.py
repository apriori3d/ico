"""Deep imports of ico are typed for consumers checked by pyright strict.

Subpackages without ``__init__.py`` are namespace packages, and pyright does
not apply the ``py.typed`` of ``ico`` to them: ``from ico.core.operator
import IcoOperator`` failed with ``reportMissingTypeStubs``.
"""

import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "ico"

CONSUMER = """\
from ico.core.operator import IcoOperator
from ico.core.runtime.progress import IcoProgress
from ico.describe.plan.json_renderer import plan_view
from ico.runtime.worker.mp.mp_worker import MPWorker
from ico.tools.board import LiveBoardTool
from ico.tools.progress.json_progress_tool import JsonProgressTool

__all__ = [
    "IcoOperator",
    "IcoProgress",
    "plan_view",
    "MPWorker",
    "LiveBoardTool",
    "JsonProgressTool",
]
"""


def test_every_subpackage_is_a_regular_package() -> None:
    missing = [
        str(d.relative_to(SRC))
        for d in SRC.rglob("*")
        if d.is_dir()
        and d.name != "__pycache__"
        and any(d.glob("*.py"))
        and not (d / "__init__.py").is_file()
    ]
    assert missing == []


def test_deep_imports_pass_pyright_strict(tmp_path: Path) -> None:
    (tmp_path / "consumer.py").write_text(CONSUMER)
    (tmp_path / "pyrightconfig.json").write_text(
        json.dumps({"typeCheckingMode": "strict"})
    )
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyright",
            "--outputjson",
            "--pythonpath",
            sys.executable,
            "-p",
            str(tmp_path / "pyrightconfig.json"),
            str(tmp_path / "consumer.py"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads(done.stdout)
    errors = [
        f"{d.get('rule')}: {d['message']}"
        for d in report["generalDiagnostics"]
        if d["severity"] == "error"
    ]
    assert errors == []
