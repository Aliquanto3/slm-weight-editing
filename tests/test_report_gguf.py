import hashlib
import json

import numpy as np
import pytest
from conftest import (
    COLS,
    LICENSE,
    ROWS,
    TARGET,
    bf16_to_f32,
    build,
    hf_name,
    q8_0_reference,
    to_bf16,
    write_gguf,
    write_safetensors,
    write_store,
)

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
    quantize_tensor,
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
    assert diff["fidelity"] == {
        "intermediate": "f32",
        "controls": 2,
        "passed": 2,
        "ok": True,
        "target_originals_checked": False,
    }
    # sha256 fournis sans provenance : le diff ne se prétend pas être l'édition du MIT
    assert diff["base"] == {"repo": None, "revision": None} and diff["edit"] == {"repo": None}
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


def test_modelfile_bytes_have_no_added_cr_but_keep_template_cr(tmp_path):
    # Octets exacts (LF) sous tout OS ; le CR du gabarit d'origine est conservé.
    s = build(tmp_path, template="a\r\nb\n{{ .Prompt }}", system="Sois bref.")
    result = run(s, tmp_path / "out")
    raw = result.modelfile.read_bytes()
    assert b"\r" not in result.diff.read_bytes() and b"\r" not in result.fidelity.read_bytes()
    assert raw.count(b"\r") == 1
    assert b'TEMPLATE """a\r\nb\n{{ .Prompt }}"""\nSYSTEM """Sois bref."""\n' in raw


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
            "--base-checkpoint",
            str(s.base_checkpoint),
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
        [
            "report-gguf",
            "--checkpoint",
            str(tmp_path),
            "--base-checkpoint",
            str(tmp_path),
            "--ollama-models",
            str(tmp_path),
        ]
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
        [
            "report-gguf",
            "--checkpoint",
            str(synth.checkpoint),
            "--base-checkpoint",
            str(synth.base_checkpoint),
            "--ollama-models",
            str(synth.store),
        ]
    )
    assert code == 1
    assert "illisible" in capsys.readouterr().err


def test_cli_requires_the_base_checkpoint(synth, capsys):
    with pytest.raises(SystemExit) as exc:
        slm_weight_editing.main(["report-gguf", "--checkpoint", str(synth.checkpoint)])
    assert exc.value.code == 2
    assert "--base-checkpoint" in capsys.readouterr().err


def test_cli_reports_an_undecodable_license(tmp_path, capsys):
    s = build(tmp_path)
    manifest = s.store / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5" / "0.8b"
    data = json.loads(manifest.read_text("utf-8"))
    (layer,) = [x for x in data["layers"] if x["mediaType"].endswith(".license")]
    (s.store / "blobs" / layer["digest"].replace(":", "-")).write_bytes(bytes([0xFF, 0xFE, 0xFA]))
    code = slm_weight_editing.main(
        [
            "report-gguf",
            "--checkpoint",
            str(s.checkpoint),
            "--base-checkpoint",
            str(s.base_checkpoint),
            "--ollama-models",
            str(s.store),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
    assert "couche license" in capsys.readouterr().err


EMBED = "model.language_model.embed_tokens.weight"


def build_embed(tmp_path):
    """GGUF officiel avec token_embd.weight (sans output.weight) ; la ligne 3 est éditée."""
    rng = np.random.default_rng(7)
    base = to_bf16(rng.normal(0, 0.05, (ROWS, COLS)).astype(np.float32))
    edited = base.copy()
    row = bf16_to_f32(base[3]).copy()
    row[40] += 0.3
    edited[3] = to_bf16(row)
    gguf_path = write_gguf(tmp_path / "officiel.gguf", {"token_embd.weight": q8_0_reference(base)})
    store = write_store(tmp_path / "store", gguf_path)
    ckpt = tmp_path / "edite"
    ckpt.mkdir()
    write_safetensors(ckpt / "model.safetensors", {EMBED: edited})
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    write_safetensors(base_dir / "model.safetensors", {EMBED: base})
    return store, gguf_path, ckpt, base_dir, edited


def test_embed_tokens_maps_to_token_embd():
    assert gguf_name(EMBED) == "token_embd.weight"
    assert gguf_name("model.embed_tokens.weight") == "token_embd.weight"
    assert gguf_name("model.language_model.layers.3.self_attn.q_proj.weight") is None
    assert gguf_name("lm_head.weight") == "output.weight"


def test_report_embed_row_changes_only_that_rows_blocks(tmp_path):
    store, official_path, ckpt, base_dir, edited = build_embed(tmp_path)
    out = tmp_path / "out"
    sha = hashlib.sha256(edited.tobytes()).hexdigest()
    result = report_gguf(
        ckpt, out, store=store, base_checkpoint=base_dir, expected_sha256={EMBED: sha}
    )
    official = official_path.read_bytes()
    new = result.gguf.read_bytes()
    info = read_gguf(official_path).tensor("token_embd.weight")
    assert info.dims == (COLS, ROWS)
    block = 34  # Q8_0 : échelle f16 + 32 quanta
    row_bytes = COLS // 32 * block
    changed = [i for i in range(len(official)) if official[i] != new[i]]
    start = info.offset + 3 * row_bytes
    # Le bloc 1 de la ligne 3 (colonnes 32 à 63) porte seul la modification : tout octet hors de
    # ce bloc est identique à l'officiel, et le bloc a bien changé.
    lo, hi = start + block, start + 2 * block
    assert changed and lo <= min(changed) and max(changed) < hi
    assert new[:lo] == official[:lo] and new[hi:] == official[hi:]
    assert len(new) == len(official)
    diff = json.loads(result.diff.read_text("utf-8"))
    assert [t["gguf_name"] for t in diff["tensors"]] == ["token_embd.weight"]
    assert diff["tensors"][0]["gguf_blocks_changed"] == 1
    fidelity = json.loads(result.fidelity.read_text("utf-8"))
    assert fidelity["ok"] and fidelity["controls"] == 1
    assert fidelity["tensors"][0]["gguf_name"] == "token_embd.weight"
    assert fidelity["tensors"][0]["origin"] == "checkpoint de base, original du tenseur visé"


def test_report_embed_fidelity_failure_on_tampered_official(tmp_path):
    store, official_path, ckpt, base_dir, edited = build_embed(tmp_path)
    bad = tmp_path / "bad"
    bad.mkdir()
    rng = np.random.default_rng(9)
    other = to_bf16(rng.normal(0, 0.05, (ROWS, COLS)).astype(np.float32))
    store2 = write_store(bad / "store", write_gguf(bad / "o.gguf", {"token_embd.weight": other}))
    sha = hashlib.sha256(edited.tobytes()).hexdigest()
    with pytest.raises(FidelityError, match="token_embd.weight"):
        report_gguf(
            ckpt,
            tmp_path / "o",
            store=store2,
            base_checkpoint=base_dir,
            expected_sha256={EMBED: sha},
        )


def test_unsupported_tensor_is_still_refused(tmp_path):
    store, _, ckpt, base_dir, edited = build_embed(tmp_path)
    q = "model.language_model.layers.0.self_attn.q_proj.weight"
    write_safetensors(ckpt / "model.safetensors", {EMBED: edited, q: edited})
    with pytest.raises(ReportError, match="pas de correspondance GGUF vérifiée"):
        report_gguf(ckpt, tmp_path / "o", store=store, expected_sha256={q: sha256_of(edited)})


def sha256_of(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()


def embed_sha(tmp_path):
    store, official_path, ckpt, base_dir, edited = build_embed(tmp_path)
    return store, ckpt, base_dir, {EMBED: sha256_of(edited)}


def test_report_embed_without_base_checkpoint_is_refused(tmp_path):
    store, ckpt, _, expected = embed_sha(tmp_path)
    with pytest.raises(ReportError, match="--base-checkpoint requis"):
        report_gguf(ckpt, tmp_path / "o", store=store, expected_sha256=expected)
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize(
    "config",
    [{"tie_word_embeddings": False}, {"text_config": {"tie_word_embeddings": False}}],
)
def test_report_embed_refuses_untied_config(tmp_path, config):
    store, ckpt, base_dir, expected = embed_sha(tmp_path)
    (ckpt / "config.json").write_text(json.dumps(config), "utf-8")
    with pytest.raises(ReportError, match="tie_word_embeddings: false"):
        report_gguf(
            ckpt, tmp_path / "o", store=store, base_checkpoint=base_dir, expected_sha256=expected
        )


def test_report_embed_accepts_tied_config(tmp_path):
    store, ckpt, base_dir, expected = embed_sha(tmp_path)
    config = {"tie_word_embeddings": True, "text_config": {"tie_word_embeddings": True}}
    (ckpt / "config.json").write_text(json.dumps(config), "utf-8")
    report_gguf(
        ckpt, tmp_path / "o", store=store, base_checkpoint=base_dir, expected_sha256=expected
    )


def test_report_embed_refuses_a_distinct_lm_head(tmp_path):
    store, official_path, ckpt, base_dir, edited = build_embed(tmp_path)
    write_safetensors(ckpt / "model.safetensors", {EMBED: edited, "lm_head.weight": edited})
    with pytest.raises(ReportError, match="lm_head.weight distinct"):
        report_gguf(
            ckpt,
            tmp_path / "o",
            store=store,
            base_checkpoint=base_dir,
            expected_sha256={EMBED: sha256_of(edited)},
        )


def test_report_embed_refuses_a_gguf_with_output_weight(tmp_path):
    store, official_path, ckpt, base_dir, edited = build_embed(tmp_path)
    q8 = q8_0_reference(edited)
    gguf_path = write_gguf(tmp_path / "o2.gguf", {"token_embd.weight": q8, "output.weight": q8})
    store2 = write_store(tmp_path / "store2", gguf_path)
    with pytest.raises(ReportError, match="output.weight distinct"):
        report_gguf(
            ckpt,
            tmp_path / "o",
            store=store2,
            base_checkpoint=base_dir,
            expected_sha256={EMBED: sha256_of(edited)},
        )


def test_mit_path_does_not_use_embed_tokens_as_a_witness(tmp_path):
    """Chemin du lot 0 : un token_embd officiel non reproductible ne bloque pas le report."""
    s = build(tmp_path)
    rng = np.random.default_rng(3)
    embed = to_bf16(rng.normal(0, 0.05, (ROWS, COLS)).astype(np.float32))
    official = {f"blk.{i}.ffn_down.weight": q8_0_reference(s.base[hf_name(i)]) for i in range(3)}
    other = to_bf16(rng.normal(0, 0.05, (ROWS, COLS)).astype(np.float32))
    official["token_embd.weight"] = q8_0_reference(other)  # ne correspond pas à `embed`
    gguf_path = write_gguf(tmp_path / "o3.gguf", official)
    store = write_store(tmp_path / "store3", gguf_path)
    write_safetensors(s.checkpoint / "model-00003.safetensors", {EMBED: embed})
    result = report_gguf(s.checkpoint, tmp_path / "o", store=store, expected_sha256=s.expected_sha)
    fidelity = json.loads(result.fidelity.read_text("utf-8"))
    assert fidelity["ok"]
    assert "token_embd.weight" not in [t["gguf_name"] for t in fidelity["tensors"]]


def test_quantize_tensor_by_chunks_matches_a_single_pass(tmp_path, monkeypatch):
    from slm_weight_editing import report_gguf as mod
    from slm_weight_editing.safetensors_file import open_checkpoint

    rng = np.random.default_rng(5)
    values = to_bf16(rng.normal(0, 0.05, (ROWS, COLS)).astype(np.float32))
    st = open_checkpoint(write_safetensors(tmp_path / "m.safetensors", {EMBED: values}))[EMBED]
    whole = quantize_q8_0(bf16_bytes_to_f32(values.tobytes()), "f32")
    monkeypatch.setattr(mod, "_QUANT_CHUNK_BLOCKS", 3)  # tranches qui coupent les lignes
    assert quantize_tensor(st, "f32") == whole
    assert quantize_tensor(st, "f16") == quantize_q8_0(bf16_bytes_to_f32(values.tobytes()), "f16")
