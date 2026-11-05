from pathlib import Path

import gguf
import numpy as np
import pytest
from conftest import PARAMS, to_bf16, write_store

from slm_weight_editing import main
from slm_weight_editing.gguf_add_output import AddOutputError, add_output_weight
from slm_weight_editing.gguf_file import read_gguf
from slm_weight_editing.q8_0 import dequantize_q8_0


def make(
    path: Path, embd_type: str, extra: dict[str, np.ndarray] | None = None, alignment: int = 0
) -> np.ndarray:
    """GGUF synthétique minuscule ; renvoie les valeurs float32 de token_embd."""
    rng = np.random.default_rng(3)
    values = rng.normal(0, 0.5, (5, 64)).astype(np.float32)
    w = gguf.GGUFWriter(str(path), arch="qwen35")
    if alignment:
        w.add_custom_alignment(alignment)
    w.add_string("general.name", "synthétique")
    w.add_array("tokenizer.ggml.tokens", ["a", "bé", "c"])
    w.add_tensor("blk.0.norm.weight", np.arange(16, dtype=np.float32))
    if embd_type == "Q8_0":
        q = gguf.quants.quantize(values, gguf.GGMLQuantizationType.Q8_0)
        w.add_tensor("token_embd.weight", q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    elif embd_type == "BF16":
        bf = to_bf16(values)
        w.add_tensor("token_embd.weight", bf, raw_dtype=gguf.GGMLQuantizationType.BF16)
    else:
        dtype = {"F16": np.float16, "F32": np.float32}[embd_type]
        w.add_tensor("token_embd.weight", values.astype(dtype))
    for name, data in (extra or {}).items():
        w.add_tensor(name, data)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return values


def kv_bytes(path: Path) -> bytes:
    """Octets des métadonnées (tableaux compris), après le nombre de tenseurs."""
    from slm_weight_editing.gguf_add_output import _header_ends

    return path.read_bytes()[16 : _header_ends(path)[0]]


@pytest.mark.parametrize(
    ("embd_type", "alignment"), [("Q8_0", 0), ("F16", 0), ("F32", 0), ("BF16", 0), ("Q8_0", 64)]
)
def test_adds_output_weight_and_keeps_everything_else(tmp_path, embd_type, alignment):
    src, dst = tmp_path / "in.gguf", tmp_path / "out" / "out.gguf"
    values = make(src, embd_type, alignment=alignment)
    res = add_output_weight(src, dst, store=tmp_path / "store")
    a, b = read_gguf(src), read_gguf(dst)
    assert list(b.tensors) == [*a.tensors, "output.weight"]
    assert b.metadata == a.metadata and b.alignment == a.alignment == (alignment or 32)
    assert kv_bytes(dst) == kv_bytes(src)
    for name in a.tensors:
        assert b.read_tensor_bytes(name) == a.read_tensor_bytes(name)
        assert b.tensors[name].dims == a.tensors[name].dims
        assert b.tensors[name].ggml_type == a.tensors[name].ggml_type
    out, embd = b.tensor("output.weight"), b.tensor("token_embd.weight")
    assert (out.dims, out.type_name) == (embd.dims, "Q8_0")
    assert out.offset % b.alignment == 0 and b.data_offset % b.alignment == 0
    got = b.read_tensor_bytes("output.weight")
    if embd_type == "Q8_0":
        assert got == a.read_tensor_bytes("token_embd.weight")
        assert res["mode"] == "copie d'octets"
    else:
        assert np.abs(dequantize_q8_0(got).reshape(values.shape) - values).max() < 0.03
        assert res["mode"] == "requantifié"
    assert res["added_bytes"] == dst.stat().st_size - src.stat().st_size > 0


def test_gguf_py_reads_the_output(tmp_path):
    src, dst = tmp_path / "in.gguf", tmp_path / "out.gguf"
    make(src, "Q8_0")
    add_output_weight(src, dst, store=tmp_path / "store")
    reader = gguf.GGUFReader(str(dst))
    assert [t.name for t in reader.tensors][-1] == "output.weight"
    assert reader.tensors[-1].tensor_type == gguf.GGMLQuantizationType.Q8_0


def test_refuses_existing_output_weight(tmp_path):
    src = tmp_path / "in.gguf"
    make(src, "Q8_0", {"output.weight": np.zeros((5, 64), np.float32)})
    with pytest.raises(AddOutputError, match="déjà un output.weight"):
        add_output_weight(src, tmp_path / "out.gguf", store=tmp_path / "store")
    assert not (tmp_path / "out.gguf").exists()


def test_never_overwrites_input_existing_file_or_store(tmp_path):
    src = tmp_path / "in.gguf"
    make(src, "Q8_0")
    before = src.read_bytes()
    store = tmp_path / "store"
    (store / "blobs").mkdir(parents=True)
    other = tmp_path / "other.gguf"
    other.write_bytes(b"x")
    for dst, msg in [
        (src, "fichier d'entrée"),
        (other, "existe déjà"),
        (store / "blobs" / "sha256-1", "magasin d'Ollama"),
    ]:
        with pytest.raises(AddOutputError, match=msg):
            add_output_weight(src, dst, store=store)
    assert src.read_bytes() == before and other.read_bytes() == b"x"


def test_missing_token_embd_is_an_error(tmp_path):
    from slm_weight_editing.gguf_file import GGUFError

    src = tmp_path / "in.gguf"
    w = gguf.GGUFWriter(str(src), arch="qwen35")
    w.add_tensor("blk.0.norm.weight", np.zeros(32, np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with pytest.raises(GGUFError, match="token_embd.weight absent"):
        add_output_weight(src, tmp_path / "out.gguf", store=tmp_path / "store")
    assert not (tmp_path / "out.gguf").exists()


def test_cli(tmp_path, capsys):
    src, dst = tmp_path / "in.gguf", tmp_path / "o.gguf"
    make(src, "F16")
    argv = ["gguf-add-output", "--gguf", str(src), "--out", str(dst)]
    assert main([*argv, "--ollama-models", str(tmp_path / "s")]) == 0
    assert "Q8_0 (requantifié)" in capsys.readouterr().out
    assert main([*argv, "--ollama-models", str(tmp_path / "s")]) == 2
    assert "existe déjà" in capsys.readouterr().err


def test_cli_model_writes_faithful_modelfiles(tmp_path, capsys, monkeypatch):
    src = tmp_path / "in.gguf"
    make(src, "Q8_0")
    store = write_store(tmp_path / "store", src, template="{{ .Prompt }}")
    monkeypatch.chdir(tmp_path)
    argv = ["gguf-add-output", "--model", "qwen3.5:0.8b", "--ollama-models", str(store)]
    assert main(argv) == 0
    dst = tmp_path / "models" / "qwen3.5-0.8b-output.gguf"
    assert read_gguf(dst).tensors.keys() >= {"output.weight"}
    test_mf = (tmp_path / "models" / "qwen3.5-0.8b-output.Modelfile").read_text("utf-8")
    control_mf = (tmp_path / "models" / "qwen3.5-0.8b-output-temoin.Modelfile").read_text("utf-8")
    assert test_mf.splitlines()[0] == f"FROM {dst.resolve().as_posix()}"
    first = control_mf.splitlines()[0]
    assert first.startswith("FROM ") and "/blobs/sha256-" in first
    assert test_mf.splitlines()[1:] == control_mf.splitlines()[1:]
    for line in ("RENDERER qwen3.5", "PARSER qwen3.5", "PARAMETER presence_penalty 1.5"):
        assert line in test_mf
    assert len([x for x in test_mf.splitlines() if x.startswith("PARAMETER")]) == len(PARAMS)
    assert 'TEMPLATE """{{ .Prompt }}"""' in test_mf and "LICENSE" in test_mf
    capsys.readouterr()
    dst.unlink()
    assert main(argv) == 2  # Modelfiles existants : rien n'est écrasé
    assert "Modelfile existe déjà" in capsys.readouterr().err
    assert not dst.exists()


def test_cli_model_cleans_up_when_modelfile_write_fails(tmp_path, capsys, monkeypatch):
    src = tmp_path / "in.gguf"
    make(src, "Q8_0")
    store = write_store(tmp_path / "store", src)
    monkeypatch.chdir(tmp_path)
    real_write = Path.write_text

    def failing_write(self, *args, **kwargs):
        if self.name.endswith("-temoin.Modelfile"):
            raise OSError("disque plein")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write)
    argv = ["gguf-add-output", "--model", "qwen3.5:0.8b", "--ollama-models", str(store)]
    assert main(argv) == 2
    assert "disque plein" in capsys.readouterr().err
    assert list((tmp_path / "models").iterdir()) == []
