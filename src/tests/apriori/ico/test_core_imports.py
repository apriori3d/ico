"""Core (`ico`) must import only the standard library and `rich` at runtime.

Optional dependencies (pydot, torch, ...) may be imported lazily inside
functions, and typing-only helpers belong under `if TYPE_CHECKING:`. A
module-level third-party import in core would break `pip install ico`
without extras, even when the test environment happens to have it.
"""

import subprocess
import sys
import textwrap

GUARDED_IMPORT = textwrap.dedent(
    """
    import importlib.abc
    import sys

    ALLOWED = set(sys.stdlib_module_names) | {"rich", "ico"}


    class CoreImportGuard(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.partition(".")[0] in ALLOWED:
                return None
            # Find the module whose code runs the import statement.
            frame = sys._getframe(1)
            while frame is not None and frame.f_globals.get(
                "__name__", ""
            ).startswith(("importlib", "_frozen_importlib")):
                frame = frame.f_back
            importer = frame.f_globals.get("__name__", "") if frame else ""
            if importer == "ico" or importer.startswith("ico."):
                raise ImportError(
                    f"{importer} imports third-party module {name!r}; "
                    "core depends only on rich"
                )
            return None


    sys.meta_path.insert(0, CoreImportGuard())

    import ico
    import ico.registry
    import ico.spec

    print("ok")
    """
)


def test_core_imports_only_stdlib_and_rich() -> None:
    result = subprocess.run(
        [sys.executable, "-c", GUARDED_IMPORT],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
