import sys
import textwrap
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

import ico.registry.fingerprint as fingerprint_module
from ico.core.operator import IcoOperator
from ico.registry import (
    IcoRegistry,
    default_registry,
    load_operators,
    module_fingerprint,
    module_sources,
    register,
)


class Packages:
    """Throwaway packages in a directory on ``sys.path``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.names: list[str] = []
        sys.path.insert(0, str(root))

    def new(self) -> str:
        name = f"fp_{uuid.uuid4().hex[:8]}"
        (self.root / name).mkdir()
        (self.root / name / "__init__.py").write_text("")
        self.names.append(name)
        return name

    def write(self, module: str, source: str) -> None:
        path = self.root / Path(*module.split(".")).with_suffix(".py")
        path.write_text(textwrap.dedent(source))
        sys.modules.pop(module, None)

    def cleanup(self) -> None:
        for name in [m for m in sys.modules if m.split(".")[0] in self.names]:
            del sys.modules[name]
        sys.path.remove(str(self.root))


@pytest.fixture
def packages(tmp_path: Path) -> Iterator[Packages]:
    pkgs = Packages(tmp_path)
    yield pkgs
    pkgs.cleanup()


def test_an_imported_helper_changes_the_fingerprint(packages: Packages) -> None:
    pkg = packages.new()
    packages.write(f"{pkg}.helper", "SCALE = 2\n")
    packages.write(f"{pkg}.ops", "from .helper import SCALE\n")
    before = module_fingerprint(f"{pkg}.ops")

    packages.write(f"{pkg}.helper", "SCALE = 3\n")

    assert before is not None and before.startswith("sha256:")
    assert module_fingerprint(f"{pkg}.ops") != before


def test_follows_absolute_lazy_and_transitive_imports(packages: Packages) -> None:
    pkg = packages.new()
    packages.write(f"{pkg}.deep", "")
    packages.write(f"{pkg}.helper", f"import {pkg}.deep\n")
    packages.write(
        f"{pkg}.ops",
        f"""
        def apply() -> None:
            from {pkg} import helper
        """,
    )

    sources = module_sources(f"{pkg}.ops")

    assert sources is not None
    assert set(sources) == {f"{pkg}.ops", pkg, f"{pkg}.helper", f"{pkg}.deep"}


def test_a_module_it_does_not_import_does_not_count(packages: Packages) -> None:
    pkg = packages.new()
    packages.write(f"{pkg}.ops", "X = 1\n")
    packages.write(f"{pkg}.other", "Y = 1\n")
    before = module_fingerprint(f"{pkg}.ops")

    packages.write(f"{pkg}.other", "Y = 2\n")

    assert module_fingerprint(f"{pkg}.ops") == before


def test_skips_stdlib_installed_and_ico(
    packages: Packages, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = packages.new()
    installed = packages.new()
    packages.write(f"{installed}.lib", "")
    packages.write(
        f"{pkg}.ops",
        f"""
        import json
        from ico.core.operator import IcoOperator
        import {installed}.lib
        """,
    )
    roots = fingerprint_module._installed_roots()  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(
        fingerprint_module,
        "_installed_roots",
        lambda: (*roots, (packages.root / installed).resolve()),
    )

    sources = module_sources(f"{pkg}.ops")

    assert sources is not None
    assert set(sources) == {f"{pkg}.ops"}


def test_import_cycles_end(packages: Packages) -> None:
    pkg = packages.new()
    packages.write(f"{pkg}.a", "from . import b\n")
    packages.write(f"{pkg}.b", "from . import a\n")

    sources = module_sources(f"{pkg}.a")

    assert sources is not None
    assert set(sources) == {f"{pkg}.a", f"{pkg}.b", pkg}


def test_unknown_module() -> None:
    assert module_fingerprint("no_such_module_fp") is None


def test_entry_fingerprint(packages: Packages) -> None:
    pkg = packages.new()
    packages.write(
        f"{pkg}.ops",
        f"""
        from ico.registry import register


        @register("{pkg}.double", 1)
        def double(x: int) -> int:
            '''Double a number.'''
            return x * 2
        """,
    )
    load_operators(packages.root / pkg, strict=True)
    try:
        entry = default_registry.get(f"{pkg}.double@1")
        assert entry.fingerprint() == module_fingerprint(f"{pkg}.ops")
        assert entry.fingerprint() is not None
    finally:
        default_registry.unregister(f"{pkg}.double@1")


def test_entry_without_module_has_no_fingerprint() -> None:
    registry = IcoRegistry()
    one = IcoOperator[int, int](fn=lambda x: x + 1)
    two = IcoOperator[int, int](fn=lambda x: x * 2)
    register("flow", 1, registry=registry, description="A flow.")(one | two)

    assert registry.get("flow@1").fingerprint() is None


def test_deterministic_flag() -> None:
    registry = IcoRegistry()

    def model(prompt: str) -> str:
        """Ask an image model."""
        return prompt

    def pure(text: str) -> str:
        """Pass the text on."""
        return text

    register("model", 1, deterministic=False, registry=registry)(model)
    register("pure", 1, registry=registry)(pure)

    assert not registry.get("model@1").meta.deterministic
    assert registry.get("model@1").card()["deterministic"] is False
    assert registry.get("pure@1").meta.deterministic
    assert "deterministic" not in registry.get("pure@1").card()


def test_deterministic_flag_of_a_class() -> None:
    registry = IcoRegistry()

    class Sampler(IcoOperator[int, int]):
        """Sample a number."""

        def __init__(self, seed: int = 0) -> None:
            self.seed = seed
            super().__init__(fn=self._apply)

        def _apply(self, x: int) -> int:
            return x + self.seed

    register("sampler", 1, deterministic=False, registry=registry)(Sampler)
    assert registry.get("sampler@1").card()["deterministic"] is False
