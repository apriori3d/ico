import json
import sys
import textwrap
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from ico.core.operator import IcoOperator
from ico.registry import default_registry, load_operators
from ico.runtime.worker.mp.mp_worker import MPWorker

DOUBLE = '''
from ico.registry import register


@register("{pkg}.double", 1)
def double(x: int) -> int:
    """Double a number."""
    return x * 2


def make_flow():
    return double
'''

SCALE = """
from ico import IcoOperator
from ico.registry import register


@register("{pkg}.scale", 1, status="{status}")
class Scale(IcoOperator[int, int]):
    def __init__(self, factor: int = {factor}) -> None:
        self.factor = factor
        super().__init__(fn=self._apply)

    def _apply(self, x: int) -> int:
        return x * self.factor
"""


def scale_of(ref: str) -> IcoOperator[int, int]:
    """A default instance of the registered Scale class."""
    cls = default_registry.get(ref).cls
    assert cls is not None
    return cast(Callable[[], IcoOperator[int, int]], cls)()


class Package:
    def __init__(self, root: Path) -> None:
        self.name = f"ops_{uuid.uuid4().hex[:8]}"
        self.dir = root / self.name
        self.dir.mkdir()
        (self.dir / "__init__.py").write_text("")

    def write(self, module: str, source: str, **fields: Any) -> Path:
        path = self.dir / f"{module}.py"
        path.write_text(textwrap.dedent(source).format(pkg=self.name, **fields))
        return path


@pytest.fixture
def package(tmp_path: Path) -> Iterator[Package]:
    pkg = Package(tmp_path)
    yield pkg
    for ref in [e.ref for e in default_registry if e.ref.startswith(pkg.name)]:
        default_registry.unregister(ref)
    for name in [
        m for m in sys.modules if m == pkg.name or m.startswith(pkg.name + ".")
    ]:
        del sys.modules[name]
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))


def test_loads_all_modules(package: Package) -> None:
    package.write("double", DOUBLE)
    package.write("scale", SCALE, status="draft", factor=3)

    report = load_operators(package.dir)

    assert report.loaded == (f"{package.name}.double@1", f"{package.name}.scale@1")
    assert report.errors == ()
    entry = default_registry.get(f"{package.name}.double@1")
    assert entry.module == f"{package.name}.double"
    assert cast(IcoOperator[int, int], entry.node)(4) == 8
    json.dumps(report.to_dict())


def test_broken_file_does_not_stop_others(package: Package) -> None:
    package.write("a_broken", "def oops(:\n")
    package.write("double", DOUBLE)

    report = load_operators(package.dir)

    assert report.loaded == (f"{package.name}.double@1",)
    ((file, error),) = report.errors
    assert file.endswith("a_broken.py")
    assert error.startswith("SyntaxError:")
    assert f"{package.name}.a_broken" not in sys.modules


def test_strict_raises(package: Package) -> None:
    package.write("broken", "raise RuntimeError('boom')\n")
    with pytest.raises(RuntimeError, match="boom"):
        load_operators(package.dir, strict=True)


def test_failed_module_leaves_no_entries(package: Package) -> None:
    package.write(
        "half",
        DOUBLE + "\nraise RuntimeError('after register')\n",
    )
    report = load_operators(package.dir)

    assert report.loaded == ()
    assert f"{package.name}.double@1" not in default_registry


def test_duplicate_ref_is_an_error(package: Package) -> None:
    package.write("a", DOUBLE)
    package.write("b", DOUBLE)

    report = load_operators(package.dir)

    assert report.loaded == (f"{package.name}.double@1",)
    ((file, error),) = report.errors
    assert file.endswith("b.py")
    assert "already registered" in error


def test_second_call_skips_imported_modules(package: Package) -> None:
    package.write("double", DOUBLE)
    load_operators(package.dir)
    package.write("scale", SCALE, status="draft", factor=2)

    report = load_operators(package.dir)

    assert report.loaded == (f"{package.name}.scale@1",)
    assert report.skipped == (f"{package.name}.double",)


def test_replace_reloads_draft_module(package: Package) -> None:
    package.write("scale", SCALE, status="draft", factor=2)
    load_operators(package.dir)
    package.write("scale", SCALE, status="draft", factor=5)

    report = load_operators(package.dir, replace=True)

    assert report.loaded == (f"{package.name}.scale@1",)
    assert scale_of(f"{package.name}.scale@1")(1) == 5


def test_replace_refuses_stable_operators(package: Package) -> None:
    package.write("scale", SCALE, status="stable", factor=2)
    load_operators(package.dir)
    package.write("scale", SCALE, status="stable", factor=5)

    report = load_operators(package.dir, replace=True)

    ((_, error),) = report.errors
    assert "cannot replace non-draft operators" in error
    assert scale_of(f"{package.name}.scale@1")(1) == 2  # unchanged


def test_failed_reload_restores_previous_entries(package: Package) -> None:
    package.write("scale", SCALE, status="draft", factor=2)
    load_operators(package.dir)
    package.write("scale", "def oops(:\n")

    report = load_operators(package.dir, replace=True)

    assert report.errors and report.errors[0][1].startswith("SyntaxError:")
    assert scale_of(f"{package.name}.scale@1")(1) == 2


def test_rejects_non_package(tmp_path: Path) -> None:
    (tmp_path / "plain").mkdir()
    with pytest.raises(ValueError, match="no __init__.py"):
        load_operators(tmp_path / "plain")


def test_rejects_package_name_conflict(package: Package, tmp_path: Path) -> None:
    load_operators(package.dir)
    other = tmp_path / "elsewhere" / package.name
    other.mkdir(parents=True)
    (other / "__init__.py").write_text("")

    with pytest.raises(ValueError, match="already refers to"):
        load_operators(other)


def test_loaded_operator_runs_in_spawned_worker(package: Package) -> None:
    package.write("double", DOUBLE)
    load_operators(package.dir)
    module = sys.modules[f"{package.name}.double"]

    # The spawned process imports the module by name to unpickle the factory.
    worker = MPWorker(module.make_flow)
    worker.activate()
    try:
        assert worker(21) == 42
    finally:
        worker.deactivate()
