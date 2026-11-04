"""Fabriques synthétiques : checkpoints safetensors bf16, GGUF minimal et magasin Ollama factice.

Le GGUF « officiel » est écrit par gguf-py (GGUFWriter et gguf.quants), indépendant du code
testé : le contrôle de fidélité compare donc deux implémentations de Q8_0.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import gguf
import numpy as np
import pytest

ROWS, COLS = 8, 64  # forme HF (sortie, entrée) ; ggml : (64, 8)
TARGET = "model.language_model.layers.1.mlp.down_proj.weight"
LICENSE = "Licence factice\nligne 2"
PARAMS = {"presence_penalty": 1.5, "temperature": 1, "top_k": 20, "top_p": 0.95}
CONFIG = {"model_format": "gguf", "file_type": "Q8_0", "renderer": "qwen3.5", "parser": "qwen3.5"}


def to_bf16(values: np.ndarray) -> np.ndarray:
    """float32 -> motifs bf16 (troncature), en uint16."""
    return (np.asarray(values, np.float32).view(np.uint32) >> 16).astype("<u2")


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def write_safetensors(path: Path, tensors: dict[str, np.ndarray], dtype: str = "BF16") -> Path:
    header, chunks, offset = {}, [], 0
    for name, arr in tensors.items():
        data = np.ascontiguousarray(arr).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(chunks))
    return path


def q8_0_reference(u16: np.ndarray) -> np.ndarray:
    return gguf.quants.quantize(bf16_to_f32(u16), gguf.GGMLQuantizationType.Q8_0)


def write_gguf(path: Path, q8_tensors: dict[str, np.ndarray]) -> Path:
    writer = gguf.GGUFWriter(str(path), arch="qwen35")
    writer.add_block_count(len(q8_tensors))
    writer.add_string("general.name", "synthétique")
    writer.add_array("tokenizer.ggml.tokens", ["a", "bé", "c"])
    writer.add_array("tokenizer.ggml.scores", [0.0, 1.0, 2.0])
    writer.add_tensor("token_norm.weight", np.arange(16, dtype=np.float32))
    for name, data in q8_tensors.items():
        writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def write_store(root: Path, gguf_path: Path, model: str = "qwen3.5:0.8b", **extra) -> Path:
    blobs = root / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)

    def blob(data: bytes) -> str:
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        (blobs / digest.replace(":", "-")).write_bytes(data)
        return digest

    params = extra.get("params", PARAMS)
    layers = [
        ("model", gguf_path.read_bytes()),
        ("license", LICENSE.encode()),
        ("params", json.dumps(params).encode()),
    ]
    for kind in ("template", "system"):
        if kind in extra:
            layers.append((kind, extra[kind].encode()))
    manifest = {
        "schemaVersion": 2,
        "config": {"digest": blob(json.dumps(extra.get("config", CONFIG)).encode())},
        "layers": [
            {"mediaType": f"application/vnd.ollama.image.{kind}", "digest": blob(data)}
            for kind, data in layers
        ],
    }
    name, tag = model.split(":")
    path = root / "manifests" / "registry.ollama.ai" / "library" / name / tag
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), "utf-8")
    return root


@dataclass
class Synth:
    store: Path
    official_gguf: Path
    checkpoint: Path
    base_checkpoint: Path
    base: dict[str, np.ndarray]
    edited: dict[str, np.ndarray]

    @property
    def expected_sha(self) -> dict[str, str]:
        return {TARGET: hashlib.sha256(self.edited[TARGET].tobytes()).hexdigest()}


def hf_name(layer: int) -> str:
    return f"model.language_model.layers.{layer}.mlp.down_proj.weight"


def build(tmp_path: Path, tamper=None, **store_extra) -> Synth:
    rng = np.random.default_rng(0)
    base = {}
    for layer in range(3):
        values = rng.normal(0, 0.05, (ROWS, COLS)).astype(np.float32)
        values[0, :32] = 0.0  # un bloc nul par tenseur
        base[hf_name(layer)] = to_bf16(values)
    edited = dict(base)
    target = bf16_to_f32(base[TARGET]).copy()
    target[:, 5] = 0.0  # « colonne retirée », comme l'édition du MIT
    target[3, 40] = 0.2
    edited[TARGET] = to_bf16(target)
    official = {f"blk.{i}.ffn_down.weight": q8_0_reference(base[hf_name(i)]) for i in range(3)}
    if tamper:
        tamper(official)
    gguf_path = write_gguf(tmp_path / "officiel.gguf", official)
    store = write_store(tmp_path / "store", gguf_path, **store_extra)
    ckpt = tmp_path / "edite"
    ckpt.mkdir()
    write_safetensors(ckpt / "model-00001.safetensors", {k: edited[k] for k in sorted(edited)[:2]})
    write_safetensors(ckpt / "model-00002.safetensors", {k: edited[k] for k in sorted(edited)[2:]})
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    write_safetensors(base_dir / "model.safetensors", base)
    return Synth(store, gguf_path, ckpt, base_dir, base, edited)


@pytest.fixture(autouse=True)
def no_real_cuda(monkeypatch):
    """Aucun test n'interroge le pilote CUDA : le poste peut avoir un benchmark en cours."""
    from slm_weight_editing import harness

    monkeypatch.setattr(harness, "_cuda_available", lambda: False)


@pytest.fixture
def synth(tmp_path: Path) -> Synth:
    return build(tmp_path)
