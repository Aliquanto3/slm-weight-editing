"""uv.lock : torch CPU par défaut (CI), roue CUDA seulement avec `--extra cuda` (poste GPU)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

LOCK = tomllib.loads((Path(__file__).resolve().parents[1] / "uv.lock").read_text("utf-8"))
CUDA_EXTRA = "extra-18-slm-weight-editing-cuda"
PROJECT = next(p for p in LOCK["package"] if p["name"] == "slm-weight-editing")


def torch_deps(entries: list[dict]) -> dict[str, str]:
    return {e["version"]: e.get("marker", "") for e in entries if e["name"] == "torch"}


def test_default_resolution_never_selects_cuda_wheel():
    """Sans extra : PyPI (Windows, macOS) et index CPU (Linux) ; la roue cu132 exige l'extra."""
    deps = torch_deps(PROJECT["dependencies"])
    assert set(deps) == {"2.15.0", "2.15.0+cpu", "2.15.0+cu132"}
    assert f"extra == '{CUDA_EXTRA}'" in deps["2.15.0+cu132"]
    assert f"extra != '{CUDA_EXTRA}'" in deps["2.15.0"]
    assert deps["2.15.0+cpu"].startswith("sys_platform == 'linux'")
    sources = {
        e["version"]: e["source"]["registry"]
        for e in PROJECT["dependencies"]
        if e["name"] == "torch"
    }
    assert sources["2.15.0"] == "https://pypi.org/simple"
    assert sources["2.15.0+cpu"] == "https://download.pytorch.org/whl/cpu"


def test_no_nvidia_package_in_lock():
    assert [p["name"] for p in LOCK["package"] if p["name"].startswith("nvidia")] == []


def test_cuda_extra_resolves_cu132_for_windows():
    cuda = [e for e in PROJECT["optional-dependencies"]["cuda"] if e["name"] == "torch"]
    cu132 = [e for e in cuda if e["version"] == "2.15.0+cu132"]
    assert len(cu132) == 1
    assert cu132[0]["source"]["registry"] == "https://download.pytorch.org/whl/cu132"
    wheels = next(
        p["wheels"]
        for p in LOCK["package"]
        if p["name"] == "torch" and p["version"] == "2.15.0+cu132"
    )
    assert any("cp314-cp314-win_amd64" in w["url"] for w in wheels)


def test_cuda_extra_is_not_a_default_dependency_of_ci():
    assert PROJECT["metadata"]["provides-extras"] == ["cpu", "cuda"]
    assert [
        wf
        for wf in Path(__file__).resolve().parents[1].glob(".github/workflows/*.yml")
        if "--extra" in wf.read_text("utf-8") or "--all-extras" in wf.read_text("utf-8")
    ] == []


@pytest.mark.parametrize("name", ["cpu", "cuda"])
def test_extras_declared(name):
    assert name in PROJECT["metadata"]["provides-extras"]
