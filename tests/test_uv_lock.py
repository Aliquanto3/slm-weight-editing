"""uv.lock : torch CPU par défaut (CI), roue CUDA seulement avec `--extra cuda` (poste GPU).

Asserts indépendants des versions : ils tiennent après un `uv lock --upgrade`.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCK = tomllib.loads((ROOT / "uv.lock").read_text("utf-8"))
CUDA_EXTRA = "extra-18-slm-weight-editing-cuda"
PROJECT = next(p for p in LOCK["package"] if p["name"] == "slm-weight-editing")
CUDA_INDEX = "https://download.pytorch.org/whl/cu"


def torch_entries(entries: list[dict]) -> list[dict]:
    return [e for e in entries if e["name"] == "torch"]


def is_cuda(entry: dict) -> bool:
    return "+cu" in entry.get("version", "") or entry["source"]["registry"].startswith(CUDA_INDEX)


def test_default_resolution_never_selects_cuda_wheel():
    """Sans extra : toute roue CUDA du projet est conditionnée à l'extra `cuda`."""
    deps = torch_entries(PROJECT["dependencies"])
    assert deps
    for e in deps:
        marker = e.get("marker", "")
        if is_cuda(e):
            assert f"extra == '{CUDA_EXTRA}'" in marker, e
            assert f"extra != '{CUDA_EXTRA}'" not in marker, e
        else:
            assert not e["source"]["registry"].startswith(CUDA_INDEX), e
    linux_cpu = [e for e in deps if e.get("marker", "").startswith("sys_platform == 'linux'")]
    assert linux_cpu and all(
        e["source"]["registry"] == "https://download.pytorch.org/whl/cpu" for e in linux_cpu
    )


def test_no_nvidia_package_in_lock():
    assert [p["name"] for p in LOCK["package"] if p["name"].startswith("nvidia")] == []


def test_cuda_extra_resolves_a_cuda_wheel_for_windows():
    cuda = [e for e in torch_entries(PROJECT["optional-dependencies"]["cuda"]) if is_cuda(e)]
    assert len(cuda) == 1
    wheels = next(
        p["wheels"]
        for p in LOCK["package"]
        if p["name"] == "torch" and p["version"] == cuda[0]["version"]
    )
    assert any("win_amd64" in w["url"] for w in wheels)


def test_cuda_extra_is_not_installed_by_ci():
    for wf in (ROOT / ".github" / "workflows").glob("*.yml"):
        text = wf.read_text("utf-8")
        assert "--extra cuda" not in text and "--all-extras" not in text, wf.name


@pytest.mark.parametrize("name", ["cpu", "cuda"])
def test_extras_declared(name):
    assert name in PROJECT["metadata"]["provides-extras"]
