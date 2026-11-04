import hashlib
import json

import numpy as np
import pytest
from conftest import LICENSE, TARGET, build, hf_name, write_safetensors

import slm_weight_editing
from slm_weight_editing.gguf_file import read_gguf
from slm_weight_editing.q8_0 import bf16_bytes_to_f32, quantize_q8_0
from slm_weight_editing.report_gguf import (
    WSP_BASE_REVISION,
    WSP_EDITED_SHA256,
    FidelityError,
    ReportError,
    ShaMismatchError,
    gguf_name,
    report_gguf,
)

TARGET_GGUF = "blk.1.ffn_down.weight"


def run(s, out, **kw):
    kw.setdefault("expected_sha256", s.expected_sha)
    return report_gguf(s.checkpoint, out, store=s.store, **kw)


def test_hf_to_gguf_names():
    assert gguf_name("model.language_model.layers.15.mlp.down_proj.weight") == (
        "blk.15.ffn_down.weight"
    )
    assert gguf_name("model.layers.19.mlp.down_proj.weight") == "blk.19.ffn_down.weight"
    assert gguf_name("model.language_model.layers.15.mlp.up_proj.weight") is None
    assert gguf_name("mtp.layers.0.mlp.down_proj.weight") is None


def test_published_constants_match_reference():
    assert WSP_BASE_REVISION == "2fc06364715b967f1860aea9cf38778875588b17"
    assert WSP_EDITED_SHA256 == {
        "model.language_model.layers.15.mlp.down_proj.weight": (
            "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53"
        ),
        "model.language_model.layers.19.mlp.down_proj.weight": (
            "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2"
        ),
    }


def test_report_changes_only_the_target_bytes(synth, tmp_path):
    out = tmp_path / "out"
    result = run(synth, out, command="uv run slm-weight-editing report-gguf ...")
    official = synth.official_gguf.read_bytes()
    edited = result.gguf.read_bytes()
    info = read_gguf(synth.official_gguf).tensor(TARGET_GGUF)
    start, end = info.offset, info.offset + info.n_bytes
    assert len(edited) == len(official)
    assert edited[:start] == official[:start]
    assert edited[end:] == official[end:]
    new = quantize_q8_0(bf16_bytes_to_f32(synth.edited[TARGET].tobytes()))
    assert edited[start:end] == new != official[start:end]
    reread = read_gguf(result.gguf)
    assert reread.tensors == read_gguf(synth.official_gguf).tensors
    assert reread.metadata == read_gguf(synth.official_gguf).metadata
    assert not list(out.glob("*.partial"))
    assert result.gguf.name == "qwen3.5-wsp-0.8b.gguf"


def test_diff_is_verifiable(synth, tmp_path):
    result = run(synth, tmp_path / "out", command="cmd")
    diff = json.loads(result.diff.read_text("utf-8"))
    official = synth.official_gguf.read_bytes()
    info = read_gguf(synth.official_gguf).tensor(TARGET_GGUF)
    (entry,) = diff["tensors"]
    assert entry["hf_name"] == TARGET and entry["gguf_name"] == TARGET_GGUF
    assert entry["ggml_type"] == "Q8_0"
    assert entry["hf_shape"] == [8, 64] and entry["gguf_dims"] == [64, 8]
    assert entry["hf_bf16_sha256_before"] is None  # pas de checkpoint de base
    assert entry["hf_bf16_sha256_after"] == synth.expected_sha[TARGET]
    old = official[info.offset : info.offset + info.n_bytes]
    assert entry["gguf_sha256_before"] == hashlib.sha256(old).hexdigest()
    new = result.gguf.read_bytes()[info.offset : info.offset + info.n_bytes]
    assert entry["gguf_sha256_after"] == hashlib.sha256(new).hexdigest()
    # colonne 5 nulle : bloc 0 des lignes 1 à 7 (la ligne 0 y est déjà nulle), plus [3, 40]
    assert entry["gguf_blocks_changed"] == 8
    assert diff["gguf"]["source_sha256"] == hashlib.sha256(official).hexdigest()
    assert diff["gguf"]["output_sha256"] == hashlib.sha256(result.gguf.read_bytes()).hexdigest()
    assert diff["fidelity"] == {"intermediate": "f32", "controls": 2, "passed": 2, "ok": True}
    assert diff["base"]["revision"] == WSP_BASE_REVISION
    assert diff["apply"] == "cmd"
    assert diff["ollama_create"].startswith("ollama create qwen3.5-wsp:0.8b -f ")


def test_base_checkpoint_adds_controls_and_before_sha(synth, tmp_path):
    result = run(synth, tmp_path / "out", base_checkpoint=synth.base_checkpoint)
    fidelity = json.loads(result.fidelity.read_text("utf-8"))
    origins = [t["origin"] for t in fidelity["tensors"]]
    assert origins.count("checkpoint de base, original du tenseur visé") == 1
    assert fidelity["controls"] == 3 and fidelity["ok"]
    (entry,) = json.loads(result.diff.read_text("utf-8"))["tensors"]
    assert (
        entry["hf_bf16_sha256_before"] == hashlib.sha256(synth.base[TARGET].tobytes()).hexdigest()
    )


def test_modelfile_reuses_official_settings(synth, tmp_path):
    result = run(synth, tmp_path / "out")
    text = result.modelfile.read_text("utf-8")
    assert text == (
        f"FROM {result.gguf.resolve().as_posix()}\n\n"
        "RENDERER qwen3.5\nPARSER qwen3.5\n"
        "PARAMETER presence_penalty 1.5\nPARAMETER temperature 1\n"
        "PARAMETER top_k 20\nPARAMETER top_p 0.95\n"
        f'LICENSE """{LICENSE}"""\n'
    )


def test_modelfile_lists_template_system_and_stop(tmp_path):
    s = build(
        tmp_path,
        template="{{ .Prompt }}",
        system="Sois bref.",
        params={"stop": ["<|im_end|>", "<x>"], "num_ctx": 4096},
    )
    text = run(s, tmp_path / "out").modelfile.read_text("utf-8")
    assert 'TEMPLATE """{{ .Prompt }}"""\nSYSTEM """Sois bref."""\n' in text
    assert 'PARAMETER stop "<|im_end|>"\nPARAMETER stop "<x>"\nPARAMETER num_ctx 4096\n' in text


def test_wrong_sha_refuses_and_writes_nothing(synth, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ShaMismatchError, match="report refusé"):
        run(synth, out, expected_sha256={TARGET: "0" * 64})
    assert not out.exists()


def flip_quantum(official):
    data = official["blk.2.ffn_down.weight"].copy()
    data[3, 34 + 2 + 7] ^= 1  # ligne 3, bloc 1 (colonnes 32-63), quantum 7
    official["blk.2.ffn_down.weight"] = data


def test_fidelity_failure_writes_no_gguf_and_reports_the_gap(tmp_path):
    s = build(tmp_path, tamper=flip_quantum)
    out = tmp_path / "out"
    with pytest.raises(FidelityError, match="blk.2.ffn_down.weight") as exc:
        run(s, out)
    assert list(out.iterdir()) == [out / "fidelite.json"]
    report = json.loads((out / "fidelite.json").read_text("utf-8"))
    assert report == exc.value.report
    assert (report["controls"], report["passed"], report["ok"]) == (2, 1, False)
    (bad,) = [t for t in report["tensors"] if not t["ok"]]
    gap = bad["gap"]
    assert (gap["blocks_total"], gap["blocks_differing"], gap["scales_differing"]) == (16, 1, 0)
    assert (gap["quanta_differing"], gap["bytes_differing"]) == (1, 1)
    (first,) = gap["first_blocks"]
    assert (first["block"], first["row"], first["col"], first["quanta_differing"]) == (7, 3, 32, 1)
    assert first["scale_official"] == first["scale_recomputed"]


def test_fidelity_reports_scale_gaps(tmp_path):
    def bump_scale(official):
        data = official["blk.0.ffn_down.weight"].copy()
        data[1, 0] ^= 1  # bit de poids faible de l'échelle du bloc 2
        official["blk.0.ffn_down.weight"] = data

    s = build(tmp_path, tamper=bump_scale)
    with pytest.raises(FidelityError) as exc:
        run(s, tmp_path / "out")
    (bad,) = [t for t in exc.value.report["tensors"] if not t["ok"]]
    first = bad["gap"]["first_blocks"][0]
    assert bad["gap"]["scales_differing"] == 1 and first["block"] == 2
    assert first["scale_official"] != first["scale_recomputed"]


def test_no_control_tensor_is_a_fidelity_failure(synth, tmp_path):
    lonely = tmp_path / "seul.safetensors"
    write_safetensors(lonely, {TARGET: synth.edited[TARGET]})
    with pytest.raises(FidelityError, match="aucun tenseur témoin"):
        report_gguf(lonely, tmp_path / "out", store=synth.store, expected_sha256=synth.expected_sha)
    assert not list((tmp_path / "out").glob("*.gguf"))


def test_shape_mismatch_is_refused(synth, tmp_path):
    ckpt = tmp_path / "transpose.safetensors"
    tensors = {k: v.T.copy() if k == TARGET else v for k, v in synth.edited.items()}
    write_safetensors(ckpt, tensors)
    sha = {TARGET: hashlib.sha256(tensors[TARGET].tobytes()).hexdigest()}
    with pytest.raises(ReportError, match="sans transposition"):
        report_gguf(ckpt, tmp_path / "out", store=synth.store, expected_sha256=sha)


def test_non_bf16_is_refused(synth, tmp_path):
    ckpt = tmp_path / "f16.safetensors"
    write_safetensors(ckpt, synth.edited, dtype="F16")
    with pytest.raises(ReportError, match="BF16 attendu"):
        report_gguf(ckpt, tmp_path / "out", store=synth.store, expected_sha256=synth.expected_sha)


def test_cli_refuses_a_checkpoint_without_the_published_sha(tmp_path, capsys):
    s = build(tmp_path)
    for layer in (15, 19):  # noms publiés, contenu synthétique : sha256 faux
        write_safetensors(
            s.checkpoint / f"l{layer}.safetensors", {hf_name(layer): np.zeros((8, 64), "<u2")}
        )
    out = tmp_path / "out"
    code = slm_weight_editing.main(
        [
            "report-gguf",
            "--checkpoint",
            str(s.checkpoint),
            "--ollama-models",
            str(s.store),
            "--out",
            str(out),
        ]
    )
    assert code == 3
    assert "sha256" in capsys.readouterr().err
    assert not out.exists()


def test_cli_reports_a_missing_manifest(tmp_path, capsys):
    code = slm_weight_editing.main(
        ["report-gguf", "--checkpoint", str(tmp_path), "--ollama-models", str(tmp_path)]
    )
    assert code == 1
    assert "manifeste de qwen3.5:0.8b introuvable" in capsys.readouterr().err


def test_failed_rerun_removes_the_previous_outputs(tmp_path):
    out = tmp_path / "out"
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    good = run(build(tmp_path / "a"), out)
    assert good.gguf.exists() and good.modelfile.exists() and good.diff.exists()
    with pytest.raises(FidelityError, match="aucun GGUF écrit"):
        run(build(tmp_path / "b", tamper=flip_quantum), out)
    assert list(out.iterdir()) == [out / "fidelite.json"]
    assert not json.loads((out / "fidelite.json").read_text("utf-8"))["ok"]


def test_sha_failure_after_a_success_leaves_no_output(synth, tmp_path):
    out = tmp_path / "out"
    run(synth, out)
    with pytest.raises(ShaMismatchError):
        run(synth, out, expected_sha256={TARGET: "0" * 64})
    assert list(out.iterdir()) == []


def test_unrenderable_modelfile_writes_no_gguf(tmp_path):
    s = build(tmp_path, system='dit """ trois guillemets')
    out = tmp_path / "out"
    with pytest.raises(ReportError, match="SYSTEM"):
        run(s, out)
    assert list(out.iterdir()) == [out / "fidelite.json"]


def test_corrupted_blob_is_refused_before_the_fidelity_check(synth, tmp_path):
    (blob,) = [p for p in (synth.store / "blobs").iterdir() if p.stat().st_size > 1000]
    data = bytearray(blob.read_bytes())
    data[-1] ^= 1
    blob.write_bytes(bytes(data))
    with pytest.raises(ReportError, match="corrompu") as exc:
        run(synth, tmp_path / "out")
    assert not isinstance(exc.value, FidelityError)


def test_replaying_the_diff_gives_the_same_output_sha(synth, tmp_path):
    first = json.loads(run(synth, tmp_path / "un").diff.read_text("utf-8"))
    second = json.loads(run(synth, tmp_path / "deux").diff.read_text("utf-8"))
    assert first["gguf"]["output_sha256"] == second["gguf"]["output_sha256"]
    assert first["tensors"] == second["tensors"]


def test_diff_masks_the_home_directory(synth, tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    out = tmp_path / "out"
    result = run(synth, out, command=f"uv run slm-weight-editing --out {out.as_posix()}")
    text = result.diff.read_text("utf-8")
    assert str(tmp_path) not in text and tmp_path.as_posix() not in text
    diff = json.loads(text)
    assert diff["apply"] == "uv run slm-weight-editing --out ~/out"
    assert diff["ollama_create"] == "ollama create qwen3.5-wsp:0.8b -f ~/out/Modelfile"


def test_cli_reports_an_unreadable_manifest(synth, tmp_path, capsys):
    manifest = synth.store / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5" / "0.8b"
    manifest.write_text("{pas du json", "utf-8")
    code = slm_weight_editing.main(
        ["report-gguf", "--checkpoint", str(synth.checkpoint), "--ollama-models", str(synth.store)]
    )
    assert code == 1
    assert "illisible" in capsys.readouterr().err
