import json

import numpy as np
import pytest
from conftest import CONFIG, LICENSE, PARAMS, write_gguf, write_safetensors, write_store

from slm_weight_editing.gguf_file import GGUFError, read_gguf
from slm_weight_editing.ollama_store import (
    OllamaStoreError,
    default_store,
    load_model,
    manifest_path,
)
from slm_weight_editing.safetensors_file import SafetensorsError, open_checkpoint


def test_gguf_header_offsets_and_bytes(tmp_path):
    data = np.arange(2 * 68, dtype=np.uint8).reshape(2, 68)  # 2 lignes de 2 blocs Q8_0
    path = write_gguf(tmp_path / "m.gguf", {"blk.0.ffn_down.weight": data})
    g = read_gguf(path)
    assert g.version == 3
    assert g.alignment == 32
    assert g.metadata["general.architecture"] == "qwen35"
    assert g.metadata["general.name"] == "synthétique"
    assert g.metadata["tokenizer.ggml.tokens"] == "<tableau de 3>"
    t = g.tensor("blk.0.ffn_down.weight")
    assert (t.dims, t.type_name, t.n_bytes) == ((64, 2), "Q8_0", 136)
    assert t.offset % 32 == 0
    assert g.read_tensor_bytes(t.name) == data.tobytes()
    norm = g.tensor("token_norm.weight")
    assert (norm.dims, norm.type_name) == ((16,), "F32")
    assert g.read_tensor_bytes(norm.name) == np.arange(16, dtype=np.float32).tobytes()


def test_gguf_rejects_other_files_and_missing_tensors(tmp_path):
    bad = tmp_path / "x.gguf"
    bad.write_bytes(b"NOPE" + bytes(32))
    with pytest.raises(GGUFError, match="pas un fichier GGUF"):
        read_gguf(bad)
    g = read_gguf(write_gguf(tmp_path / "m.gguf", {}))
    with pytest.raises(GGUFError, match="absent"):
        g.tensor("blk.9.ffn_down.weight")


def test_safetensors_directory_index(tmp_path):
    a = np.arange(6, dtype="<u2").reshape(2, 3)
    write_safetensors(tmp_path / "a.safetensors", {"x": a})
    write_safetensors(tmp_path / "b.safetensors", {"y": a[:1]})
    index = open_checkpoint(tmp_path)
    assert sorted(index) == ["x", "y"]
    assert index["x"].shape == (2, 3) and index["x"].dtype == "BF16"
    assert index["x"].read_bytes() == a.tobytes()
    assert open_checkpoint(tmp_path / "b.safetensors")["y"].read_bytes() == a[:1].tobytes()


def test_safetensors_duplicate_and_missing(tmp_path):
    a = np.zeros(2, dtype="<u2")
    write_safetensors(tmp_path / "a.safetensors", {"x": a})
    write_safetensors(tmp_path / "b.safetensors", {"x": a})
    with pytest.raises(SafetensorsError, match="deux fichiers"):
        open_checkpoint(tmp_path)
    with pytest.raises(SafetensorsError, match="introuvable"):
        open_checkpoint(tmp_path / "absent")
    (tmp_path / "vide").mkdir()
    with pytest.raises(SafetensorsError, match="aucun fichier"):
        open_checkpoint(tmp_path / "vide")


def test_manifest_path_layout(tmp_path):
    lib = tmp_path / "manifests" / "registry.ollama.ai" / "library"
    assert manifest_path(tmp_path, "qwen3.5:0.8b") == lib / "qwen3.5" / "0.8b"
    assert manifest_path(tmp_path, "qwen3.5") == lib / "qwen3.5" / "latest"
    assert manifest_path(tmp_path, "user/m:1") == lib.parent / "user" / "m" / "1"


def test_store_resolution_order(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "env"))
    assert default_store(tmp_path / "opt") == tmp_path / "opt"
    assert default_store() == tmp_path / "env"
    monkeypatch.delenv("OLLAMA_MODELS")
    assert default_store().parts[-2:] == (".ollama", "models")


def test_load_model_reads_config_layers_and_blob(tmp_path, monkeypatch):
    gguf_path = write_gguf(tmp_path / "m.gguf", {})
    store = write_store(tmp_path / "store", gguf_path, template="{{ .Prompt }}")
    monkeypatch.setenv("OLLAMA_MODELS", str(store))
    m = load_model("qwen3.5:0.8b")
    assert (m.renderer, m.parser) == (CONFIG["renderer"], CONFIG["parser"])
    assert m.params == PARAMS
    assert m.license == LICENSE
    assert m.template == "{{ .Prompt }}"
    assert m.system is None
    assert m.model_blob.read_bytes() == gguf_path.read_bytes()
    assert m.model_digest == "sha256:" + m.model_blob.name.removeprefix("sha256-")


def test_load_model_missing_manifest_or_blob(tmp_path):
    with pytest.raises(OllamaStoreError, match="manifeste"):
        load_model("qwen3.5:0.8b", tmp_path)
    store = write_store(tmp_path / "store", write_gguf(tmp_path / "m.gguf", {}))
    manifest = json.loads(manifest_path(store, "qwen3.5:0.8b").read_text("utf-8"))
    (store / "blobs" / manifest["layers"][0]["digest"].replace(":", "-")).unlink()
    with pytest.raises(OllamaStoreError, match="blob GGUF"):
        load_model("qwen3.5:0.8b", store)
