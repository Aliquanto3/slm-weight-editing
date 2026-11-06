"""CAP-10 : `lm_head` détaché des embeddings liés (story 2.8), sur des modèles minuscules.

Deux fabriques : un checkpoint safetensors synthétique (embed 8 x 64, config liée) pour le gel,
le rejeu, le report GGUF et la comparaison ; un petit Llama aux embeddings liés (transformers,
construit en mémoire) pour la preuve que les lots rejoués donnent les mêmes sorties.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import (
    COLS,
    ROWS,
    q8_0_reference,
    to_bf16,
    write_gguf,
    write_safetensors,
    write_store,
)

import slm_weight_editing
from slm_weight_editing import freeze, harness, lm_head
from slm_weight_editing.compare_edits import compare
from slm_weight_editing.freeze import DiffError, FreezeError
from slm_weight_editing.gguf_file import read_gguf
from slm_weight_editing.harness import HarnessError
from slm_weight_editing.report_gguf import (
    PROTOCOL_GAP,
    ReportError,
    gguf_name,
    report_gguf,
)
from slm_weight_editing.safetensors_file import open_checkpoint

EMBED = "model.embed_tokens.weight"
NORM = "model.norm.weight"
BLOCK = 34  # Q8_0 : échelle f16 + 32 quanta


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_bytes(directory: Path, name: str) -> bytes:
    return open_checkpoint(directory)[name].read_bytes()


def make_base(path: Path, config: dict | None = None) -> Path:
    rng = np.random.default_rng(7)
    path.mkdir(parents=True)
    write_safetensors(
        path / "model.safetensors",
        {
            EMBED: to_bf16(rng.normal(0, 0.05, (ROWS, COLS)).astype(np.float32)),
            NORM: to_bf16(rng.normal(1, 0.05, (COLS,)).astype(np.float32)),
        },
    )
    cfg = config or {"tie_word_embeddings": True, "text_config": {"tie_word_embeddings": True}}
    (path / "config.json").write_text(json.dumps(cfg), "utf-8")
    return path


@pytest.fixture
def sess(tmp_path):
    base = make_base(tmp_path / "base")
    root = tmp_path / "sessions"
    harness.open_session("s1", 10, base, base_revision="rev", sessions_root=root)
    return {"base": base, "root": root, "dir": root / "s1", "work": root / "s1" / "work"}


def detach(s):
    return lm_head.detach_lm_head("s1", sessions_root=s["root"])


def set_values(work: Path, name: str, values: dict[int, float], col: int = 40) -> None:
    """Écrit `values[ligne]` (bf16) en colonne `col` de la ligne : édition creuse en place."""
    st = open_checkpoint(work)[name]
    wide = "<u2" if st.dtype == "BF16" else "<u4"
    arr = np.frombuffer(st.read_bytes(), wide).reshape(st.shape).copy()
    for row, value in values.items():
        bits = to_bf16(np.float32(value)) if st.dtype == "BF16" else np.float32(value).view("<u4")
        arr[row, col] = bits
    with st.path.open("r+b") as f:
        f.seek(st.start)
        f.write(arr.tobytes())


def freeze_it(s):
    return freeze.freeze_session("s1", s["base"], sessions_root=s["root"])


# --- Critère 1 et 2 : la commande, ses refus ---------------------------------------------------


def test_detach_writes_equal_lm_head_untied_config_and_journal(sess):
    before = harness.tensor_hashes(sess["work"])
    config_before = (sess["work"] / "config.json").read_bytes()
    entry = detach(sess)
    after = harness.tensor_hashes(sess["work"])
    assert sorted(after) == sorted([*before, "lm_head.weight"])
    assert all(after[n] == before[n] for n in before)
    assert after["lm_head.weight"] == before[EMBED]
    index = open_checkpoint(sess["work"])
    assert index["lm_head.weight"].shape == index[EMBED].shape == (ROWS, COLS)
    assert index["lm_head.weight"].dtype == index[EMBED].dtype == "BF16"
    config = json.loads((sess["work"] / "config.json").read_text("utf-8"))
    assert config == {"tie_word_embeddings": False, "text_config": {"tie_word_embeddings": False}}
    journal = [e for e in harness.read_journal(sess["dir"]) if e["event"] == "detach_lm_head"]
    assert journal == [entry]
    assert (entry["source"], entry["lm_head"]) == (EMBED, "lm_head.weight")
    assert entry["embed_sha256"] == entry["lm_head_sha256"] == before[EMBED]
    assert entry["config"]["sha256_before"] == sha(config_before)
    assert entry["config"]["sha256_after"] == sha((sess["work"] / "config.json").read_bytes())
    assert entry["config"]["keys"] == ["tie_word_embeddings", "text_config.tie_word_embeddings"]
    assert entry["file_sha256_before"] != entry["file_sha256_after"]
    # Ni lot ni jeton de probe consommés.
    assert harness.used_batches(sess["dir"]) == 0 and harness.used_probe_tokens(sess["dir"]) == 0


def test_detach_twice_is_refused(sess):
    detach(sess)
    with pytest.raises(HarnessError, match="déjà détaché"):
        detach(sess)
    events = [e["event"] for e in harness.read_journal(sess["dir"])]
    assert events.count("detach_lm_head") == 1


def test_detach_after_freeze_is_refused_and_changes_nothing(sess):
    freeze_it(sess)
    before = harness.tensor_hashes(sess["work"])
    with pytest.raises(HarnessError, match="séance gelée"):
        detach(sess)
    assert harness.tensor_hashes(sess["work"]) == before


def test_detach_refused_while_the_lock_is_held(sess):
    with harness._session_lock(sess["dir"]), pytest.raises(HarnessError, match="un lot tourne"):
        detach(sess)
    assert "lm_head.weight" not in open_checkpoint(sess["work"])


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"tie_word_embeddings": False}, "déjà délié"),
        (
            {"tie_word_embeddings": True, "text_config": {"tie_word_embeddings": False}},
            "déjà délié",
        ),
        ({"hidden_size": 8}, "embeddings non liés"),
    ],
)
def test_detach_refuses_an_untied_or_undeclared_checkpoint(tmp_path, config, message):
    base = make_base(tmp_path / "base", config)
    root = tmp_path / "sessions"
    harness.open_session("s1", 10, base, base_revision="rev", sessions_root=root)
    with pytest.raises(HarnessError, match=message):
        lm_head.detach_lm_head("s1", sessions_root=root)
    assert harness.tensor_hashes(root / "s1" / "work") == harness.tensor_hashes(base)


def test_detach_refuses_a_checkpoint_that_already_has_lm_head(tmp_path):
    base = tmp_path / "base"
    make_base(base)
    st = open_checkpoint(base)[EMBED]
    lm_head.append_tensor(
        base / "model.safetensors", "lm_head.weight", "BF16", [ROWS, COLS], b"\0" * st.end
    )
    root = tmp_path / "sessions"
    harness.open_session("s1", 10, base, base_revision="rev", sessions_root=root)
    with pytest.raises(HarnessError, match=r"déjà délié : lm_head.weight présent"):
        lm_head.detach_lm_head("s1", sessions_root=root)


def test_transform_config_is_deterministic_and_refuses_bad_json():
    text, keys = lm_head.transform_config('{"a": 1, "tie_word_embeddings": true}')
    assert keys == ["tie_word_embeddings"]
    assert text == '{\n  "a": 1,\n  "tie_word_embeddings": false\n}\n'
    assert lm_head.transform_config(text)[0] == text
    with pytest.raises(HarnessError, match="illisible"):
        lm_head.transform_config("{")
    with pytest.raises(HarnessError, match="objet JSON"):
        lm_head.transform_config("[]")


def test_cli_detach_and_refusal_exit_codes(sess, capsys):
    root = str(sess["root"])
    assert (
        slm_weight_editing.main(["detach-lm-head", "--session", "s1", "--sessions-root", root]) == 0
    )
    assert "lm_head détaché de model.embed_tokens.weight" in capsys.readouterr().out
    assert (
        slm_weight_editing.main(["detach-lm-head", "--session", "s1", "--sessions-root", root]) == 2
    )
    assert "déjà détaché" in capsys.readouterr().err


# --- Critère 3 : les lots rejoués donnent les mêmes sorties -------------------------------------

WORDS = ["<eos>", "<unk>", "<pad>", "combien", "font", "deux", "et", "trois", "?", "x"]


def build_tied_llama(path: Path) -> Path:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token=WORDS[0], unk_token="<unk>", pad_token="<pad>"
    )
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(WORDS),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=harness.MAX_NEW_TOKENS + 64,
        eos_token_id=0,
        pad_token_id=2,
        bos_token_id=None,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)
    with torch.no_grad():
        model.model.norm.weight.zero_()
    path.mkdir(parents=True)
    model.save_pretrained(path)
    fast.save_pretrained(path)
    return path


def test_replayed_batch_and_logits_are_identical_after_detach(tmp_path):
    import torch
    from transformers import LlamaForCausalLM

    base = build_tied_llama(tmp_path / "tied")
    assert "lm_head.weight" not in open_checkpoint(base)
    root = tmp_path / "sessions"
    harness.open_session("s1", 10, base, base_revision="rev", sessions_root=root)
    data = tmp_path / "data" / "gsm8k" / "train.jsonl"
    data.parent.mkdir(parents=True)
    rows = [
        {"id": f"g{i}", "question": "combien font deux et trois ?", "split": "train"}
        for i in range(3)
    ]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    work = root / "s1" / "work"
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    with torch.no_grad():
        logits_before = LlamaForCausalLM.from_pretrained(work)(ids).logits
    first = harness.run_batch("s1", "gsm8k", "train", data, sessions_root=root, thinking="off")
    lm_head.detach_lm_head("s1", sessions_root=root)
    second = harness.run_batch("s1", "gsm8k", "train", data, sessions_root=root, thinking="off")
    assert second["outputs"] == first["outputs"] and len(first["outputs"]) == 3
    assert "lm_head.weight" in second["tensors_sha256"]
    model = LlamaForCausalLM.from_pretrained(work)
    assert model.config.tie_word_embeddings is False
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    with torch.no_grad():
        assert torch.equal(model(ids).logits, logits_before)
    # Éditer la sortie d'un jeton ne bouge plus l'embedding d'entrée.
    set_values(work, "lm_head.weight", {3: 5.0}, col=2)
    edited = LlamaForCausalLM.from_pretrained(work)
    assert torch.equal(edited.model.embed_tokens.weight, model.model.embed_tokens.weight)
    assert not torch.equal(edited.lm_head.weight, model.lm_head.weight)


# --- Critères 4 et 5 : gel et rejeu -------------------------------------------------------------


def test_freeze_describes_lm_head_as_added_tensor_and_apply_diff_replays(sess, tmp_path):
    base_embed = tensor_bytes(sess["base"], EMBED)
    detach(sess)
    set_values(sess["work"], "lm_head.weight", {1: 0.3, 3: -0.3})
    diff = freeze_it(sess)
    assert diff["format"] == "slm-weight-editing/edit-diff-v2"
    by_name = {t["name"]: t for t in diff["tensors"]}
    assert sorted(by_name) == ["lm_head.weight"]
    head = by_name["lm_head.weight"]
    assert (head["added"], head["source"], head["changed"]) == (True, EMBED, 2)
    assert head["sha256_before"] == sha(base_embed)
    assert head["sha256_after"] == sha(tensor_bytes(sess["work"], "lm_head.weight"))
    assert head["shape"] == [ROWS, COLS] and head["dtype"] == "BF16"
    assert diff["detached_lm_head"]["source"] == EMBED
    assert diff["detached_lm_head"]["config"]["keys"] == [
        "tie_word_embeddings",
        "text_config.tie_word_embeddings",
    ]
    # La config déliée est une transformation connue, pas un fichier modifié.
    assert diff["files_changed"] == []
    out = tmp_path / "rejoue"
    freeze.apply_diff(sess["dir"] / "freeze" / "diff.json", sess["base"], out)
    for name in (EMBED, NORM, "lm_head.weight"):
        assert tensor_bytes(out, name) == tensor_bytes(sess["work"], name)
    assert (out / "config.json").read_bytes() == (sess["work"] / "config.json").read_bytes()
    assert (sess["base"] / "config.json").read_text("utf-8").count("true") == 2  # base intacte


def test_detach_without_edit_still_replays(sess, tmp_path):
    detach(sess)
    diff = freeze_it(sess)
    (head,) = diff["tensors"]
    assert (head["name"], head["changed"], head["sha256_before"] == head["sha256_after"]) == (
        "lm_head.weight",
        0,
        True,
    )
    out = tmp_path / "rejoue"
    freeze.apply_diff(sess["dir"] / "freeze" / "diff.json", sess["base"], out)
    assert tensor_bytes(out, "lm_head.weight") == tensor_bytes(sess["base"], EMBED)
    assert json.loads((out / "config.json").read_text("utf-8"))["tie_word_embeddings"] is False


def test_detach_after_embed_edit_copies_the_current_embed_and_replays(sess, tmp_path):
    set_values(sess["work"], EMBED, {2: 0.4})
    detach(sess)
    set_values(sess["work"], "lm_head.weight", {5: -0.4})
    diff = freeze_it(sess)
    by_name = {t["name"]: t for t in diff["tensors"]}
    assert sorted(by_name) == sorted([EMBED, "lm_head.weight"])
    # lm_head est compté contre l'embed de la base : la ligne 2 (déjà éditée) et la ligne 5.
    assert by_name["lm_head.weight"]["changed"] == 2 and by_name[EMBED]["changed"] == 1
    out = tmp_path / "rejoue"
    freeze.apply_diff(sess["dir"] / "freeze" / "diff.json", sess["base"], out)
    for name in (EMBED, "lm_head.weight"):
        assert tensor_bytes(out, name) == tensor_bytes(sess["work"], name)


def test_freeze_ceiling_applies_to_lm_head_against_the_base_embed(sess):
    detach(sess)
    set_values(sess["work"], "lm_head.weight", {r: 0.3 for r in range(ROWS)})  # 8 / 512 : sous 10 %
    st = open_checkpoint(sess["work"])["lm_head.weight"]
    arr = np.frombuffer(st.read_bytes(), "<u2").copy()
    arr[:60] ^= 1  # 60 éléments de plus : plus de 10 % de 512
    with st.path.open("r+b") as f:
        f.seek(st.start)
        f.write(arr.tobytes())
    with pytest.raises(FreezeError, match=r"lm_head\.weight : .*plafond 10%"):
        freeze_it(sess)
    assert harness.read_state(sess["dir"]).frozen is False


def test_lm_head_written_without_the_journal_line_is_refused_at_freeze(sess):
    # Écriture directe (script ou outil) : ni détachement ni diff.
    st = open_checkpoint(sess["work"])[EMBED]
    lm_head.append_tensor(
        sess["work"] / "model.safetensors", "lm_head.weight", "BF16", [ROWS, COLS], st.read_bytes()
    )
    (sess["work"] / "config.json").write_text('{"tie_word_embeddings": false}', "utf-8")
    with pytest.raises(FreezeError, match=r"en trop : \['lm_head.weight'\]"):
        freeze_it(sess)
    assert harness.read_state(sess["dir"]).frozen is False
    assert not (sess["dir"] / "freeze").exists()


def test_another_extra_tensor_is_still_refused_after_detach(sess):
    detach(sess)
    lm_head.append_tensor(sess["work"] / "model.safetensors", "autre.weight", "BF16", [1], b"\0\0")
    with pytest.raises(FreezeError, match=r"en trop : \['autre.weight'\]"):
        freeze_it(sess)


def test_config_edited_after_detach_is_still_reported(sess):
    detach(sess)
    config = sess["work"] / "config.json"
    data = json.loads(config.read_text("utf-8"))
    data["hidden_size"] = 12
    config.write_text(json.dumps(data), "utf-8")
    diff = freeze_it(sess)
    assert [f["path"] for f in diff["files_changed"]] == ["config.json"]


def test_config_rewritten_by_hand_instead_of_detach_is_reported(sess):
    (sess["work"] / "config.json").write_text('{"tie_word_embeddings": false}', "utf-8")
    diff = freeze_it(sess)
    assert [f["path"] for f in diff["tensors"]] == [] and diff["detached_lm_head"] is None
    assert [f["path"] for f in diff["files_changed"]] == ["config.json"]


def test_v1_diff_is_still_readable_and_replayable(sess, tmp_path):
    set_values(sess["work"], EMBED, {2: 0.4})
    freeze_it(sess)
    path = sess["dir"] / "freeze" / "diff.json"
    meta = json.loads(path.read_text("utf-8"))
    meta["format"] = "slm-weight-editing/edit-diff-v1"
    del meta["detached_lm_head"]
    path.write_text(json.dumps(meta), "utf-8")
    loaded, _ = freeze.load_diff(path)
    assert loaded["detached_lm_head"] is None
    out = tmp_path / "rejoue"
    freeze.apply_diff(path, sess["base"], out)
    assert tensor_bytes(out, EMBED) == tensor_bytes(sess["work"], EMBED)
    assert not (out / "lm_head.weight").exists()


def test_malformed_detached_fields_are_refused(sess, tmp_path):
    detach(sess)
    freeze_it(sess)
    path = sess["dir"] / "freeze" / "diff.json"
    good = json.loads(path.read_text("utf-8"))

    def write(mutator):
        meta = json.loads(json.dumps(good))
        mutator(meta)
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(meta), "utf-8")
        (tmp_path / "diff.bin").write_bytes((path.parent / "diff.bin").read_bytes())
        return bad

    with pytest.raises(DiffError, match="detached_lm_head mal formé"):
        freeze.load_diff(write(lambda m: m["detached_lm_head"].pop("config")))
    with pytest.raises(DiffError, match="sans tenseur lm_head.weight"):
        freeze.load_diff(write(lambda m: m["tensors"].clear()))
    with pytest.raises(DiffError, match="tenseur ajouté sans détachement"):
        freeze.load_diff(write(lambda m: m.update(detached_lm_head=None)))
    with pytest.raises(DiffError, match="sha256 avant du détachement"):

        def tamper(m):
            m["detached_lm_head"]["config"]["sha256_before"] = "0" * 64

        out = tmp_path / "o"
        freeze.apply_diff(write(tamper), sess["base"], out)
    assert not out.exists()


# --- Critère 8 : comparaison ---------------------------------------------------------------------


def test_compare_edits_reports_added_tensor_and_protocol_gap(sess):
    detach(sess)
    set_values(sess["work"], "lm_head.weight", {1: 0.3, 3: -0.3})
    freeze_it(sess)
    diff_path = sess["dir"] / "freeze" / "diff.json"
    result, markdown = compare(
        sess["base"], None, sess["base"], diff_a=diff_path, label_a="detache", label_b="base"
    )
    weights = result["weights"]
    assert weights["a"]["tensors"] == {"lm_head.weight": 2}
    assert weights["a"]["modified_weights"] == 2
    assert weights["a"]["added"] == ["lm_head.weight"] and weights["a"]["lm_head_detached"] is True
    assert weights["b"]["added"] == [] and weights["b"]["lm_head_detached"] is False
    assert "tenseur ajouté : `lm_head.weight`" in markdown
    assert "par rapport à l'`embed_tokens` de la base" in markdown
    assert "Écart de protocole : detache a détaché `lm_head`" in markdown
    assert "base a détaché" not in markdown


def test_compare_edits_without_detach_has_no_protocol_gap(sess):
    set_values(sess["work"], EMBED, {2: 0.4})
    freeze_it(sess)
    _, markdown = compare(
        sess["base"], None, sess["base"], diff_a=sess["dir"] / "freeze" / "diff.json"
    )
    assert "Écart de protocole" not in markdown and "tenseur ajouté" not in markdown


# --- Critères 6 et 7 : report GGUF ----------------------------------------------------------------


def gguf_store(tmp_path: Path, base: Path):
    embed = np.frombuffer(tensor_bytes(base, EMBED), "<u2").reshape(ROWS, COLS)
    path = write_gguf(tmp_path / "officiel.gguf", {"token_embd.weight": q8_0_reference(embed)})
    return write_store(tmp_path / "store", path), path


def report_args(tmp_path, store, ckpt, base, diff, out):
    return [
        "report-gguf",
        "--checkpoint",
        str(ckpt),
        "--base-checkpoint",
        str(base),
        "--tag",
        "t:1",
        "--ollama-models",
        str(store),
        "--edit-diff",
        str(diff),
        "--out",
        str(out),
    ]


def test_report_gguf_adds_output_weight_and_patches_only_edited_blocks(sess, tmp_path, capsys):
    store, official = gguf_store(tmp_path, sess["base"])
    detach(sess)
    set_values(sess["work"], "lm_head.weight", {3: 0.3})
    freeze_it(sess)
    diff_path = sess["dir"] / "freeze" / "diff.json"
    ckpt = tmp_path / "models" / "rejoue"
    freeze.apply_diff(diff_path, sess["base"], ckpt)
    out = tmp_path / "gguf"
    code = slm_weight_editing.main(report_args(tmp_path, store, ckpt, sess["base"], diff_path, out))
    assert code == 0, capsys.readouterr().err
    old, new = read_gguf(official), read_gguf(out / "t-1.gguf")
    assert list(new.tensors) == [*old.tensors, "output.weight"]
    # token_embd est inchangé (embed_tokens ne l'est pas), ainsi que tout autre tenseur.
    for name in old.tensors:
        assert new.read_tensor_bytes(name) == old.read_tensor_bytes(name)
    assert new.tensor("output.weight").type_name == "Q8_0"
    token, output = (
        new.read_tensor_bytes("token_embd.weight"),
        new.read_tensor_bytes("output.weight"),
    )
    row_bytes = COLS // 32 * BLOCK
    changed = [i for i in range(len(output)) if output[i] != token[i]]
    # La colonne 40 de la ligne 3 est dans le bloc 1 de la ligne : seul ce bloc diffère.
    lo, hi = 3 * row_bytes + BLOCK, 3 * row_bytes + 2 * BLOCK
    assert changed and lo <= min(changed) and max(changed) < hi
    # Le bloc édité est la requantification exacte de la ligne de lm_head.
    expected = gguf_reference_block(ckpt, row=3, block=1)
    assert output[lo:hi] == expected
    diff = json.loads((out / "diff.json").read_text("utf-8"))
    assert diff["lm_head_detached"] is True and diff["protocol_gap"] == PROTOCOL_GAP
    (entry,) = diff["tensors"]
    assert (entry["hf_name"], entry["gguf_name"], entry["gguf_blocks_changed"]) == (
        "lm_head.weight",
        "output.weight",
        1,
    )
    assert entry["rows_edited"] == [3]
    assert entry["gguf_sha256_after"] == sha(output)
    assert entry["gguf_sha256_before"] == sha(token)
    assert entry["gguf_offset"] == new.tensor("output.weight").offset
    assert diff["gguf"]["data_offset"] == new.data_offset
    assert diff["gguf"]["output_sha256"] == sha((out / "t-1.gguf").read_bytes())
    fidelity = json.loads((out / "fidelite.json").read_text("utf-8"))
    assert fidelity["ok"] is True and fidelity["controls"] == 1
    assert fidelity["tensors"][0]["gguf_name"] == "output.weight"
    assert "embed_tokens d'origine" in fidelity["tensors"][0]["origin"]


def gguf_reference_block(ckpt: Path, row: int, block: int) -> bytes:
    rows = np.frombuffer(tensor_bytes(ckpt, "lm_head.weight"), "<u2").reshape(ROWS, COLS)
    return bytes(q8_0_reference(rows[row : row + 1])[0, block * BLOCK : (block + 1) * BLOCK])


def test_report_gguf_with_embed_and_lm_head_both_edited(sess, tmp_path, capsys):
    store, official = gguf_store(tmp_path, sess["base"])
    set_values(sess["work"], EMBED, {1: 0.3})
    detach(sess)
    set_values(sess["work"], "lm_head.weight", {5: -0.3})
    freeze_it(sess)
    diff_path = sess["dir"] / "freeze" / "diff.json"
    ckpt = tmp_path / "models" / "rejoue"
    freeze.apply_diff(diff_path, sess["base"], ckpt)
    out = tmp_path / "gguf"
    assert (
        slm_weight_editing.main(report_args(tmp_path, store, ckpt, sess["base"], diff_path, out))
        == 0
    )
    new, old = read_gguf(out / "t-1.gguf"), read_gguf(official)
    token, output = (
        new.read_tensor_bytes("token_embd.weight"),
        new.read_tensor_bytes("output.weight"),
    )
    row_bytes = COLS // 32 * BLOCK
    assert token != old.read_tensor_bytes("token_embd.weight")
    # La ligne 1 de la sortie suit l'embed édité (copiée de lui) ; seule la ligne 5 diffère.
    differing = {i // row_bytes for i in range(len(output)) if output[i] != token[i]}
    assert differing == {5}
    diff = json.loads((out / "diff.json").read_text("utf-8"))
    assert [t["gguf_name"] for t in diff["tensors"]] == ["token_embd.weight", "output.weight"]
    assert [t["gguf_blocks_changed"] for t in diff["tensors"]] == [1, 1]


def test_report_gguf_refuses_lm_head_without_declared_detach(sess, tmp_path):
    store, _ = gguf_store(tmp_path, sess["base"])
    detach(sess)
    set_values(sess["work"], "lm_head.weight", {3: 0.3})
    ckpt = sess["work"]
    expected = {"lm_head.weight": sha(tensor_bytes(ckpt, "lm_head.weight"))}
    with pytest.raises(ReportError, match="sans détachement déclaré"):
        report_gguf(
            ckpt,
            tmp_path / "o",
            store=store,
            base_checkpoint=sess["base"],
            expected_sha256=expected,
        )
    expected = {EMBED: sha(tensor_bytes(ckpt, EMBED))}
    with pytest.raises(ReportError, match="ne porte pas lm_head.weight"):
        report_gguf(
            ckpt,
            tmp_path / "o2",
            store=store,
            base_checkpoint=sess["base"],
            expected_sha256=expected,
            lm_head_detached=True,
        )
    assert not (tmp_path / "o").exists() or not any((tmp_path / "o").glob("*.gguf"))


def test_report_gguf_refuses_a_gguf_that_already_has_output_weight(sess, tmp_path):
    embed = np.frombuffer(tensor_bytes(sess["base"], EMBED), "<u2").reshape(ROWS, COLS)
    q8 = q8_0_reference(embed)
    gguf_path = write_gguf(tmp_path / "o2.gguf", {"token_embd.weight": q8, "output.weight": q8})
    store = write_store(tmp_path / "store2", gguf_path)
    detach(sess)
    set_values(sess["work"], "lm_head.weight", {3: 0.3})
    expected = {"lm_head.weight": sha(tensor_bytes(sess["work"], "lm_head.weight"))}
    with pytest.raises(ReportError, match="a déjà un output.weight"):
        report_gguf(
            sess["work"],
            tmp_path / "o",
            store=store,
            base_checkpoint=sess["base"],
            expected_sha256=expected,
            lm_head_detached=True,
        )
    assert not list((tmp_path / "o").glob("*.gguf*"))


def test_lm_head_name_maps_to_output_weight_only_at_the_root():
    assert gguf_name("lm_head.weight") == "output.weight"
    assert gguf_name("model.lm_head.weight") is None


# --- Revue passe 1 : config reliée, config modifiée avant, écriture interrompue -----------------


def test_config_relinked_after_detach_is_refused_at_freeze(sess):
    # Les lots mesurés après le retour à tie true ignoraient lm_head : pas de diff délié.
    detach(sess)
    config = sess["work"] / "config.json"
    config.write_bytes((sess["base"] / "config.json").read_bytes())
    with pytest.raises(FreezeError, match="ne déclare plus tie_word_embeddings: false"):
        freeze_it(sess)
    assert harness.read_state(sess["dir"]).frozen is False
    assert not (sess["dir"] / "freeze").exists()


def test_detach_refuses_a_config_modified_since_open(sess):
    config = sess["work"] / "config.json"
    data = json.loads(config.read_text("utf-8"))
    data["hidden_size"] = 12
    config.write_text(json.dumps(data), "utf-8")
    before = (sess["work"] / "model.safetensors").read_bytes()
    with pytest.raises(HarnessError, match="modifié depuis l'ouverture"):
        detach(sess)
    assert (sess["work"] / "model.safetensors").read_bytes() == before
    assert lm_head.detach_event(harness.read_journal(sess["dir"])) is None


def test_detach_interrupted_before_the_journal_line_restores_work(sess, monkeypatch):
    work = sess["work"]
    model, config = (work / "model.safetensors").read_bytes(), (work / "config.json").read_bytes()

    def fail(directory, entry):
        raise OSError("disque plein")

    monkeypatch.setattr(harness, "_append_journal", fail)
    with pytest.raises(OSError, match="disque plein"):
        detach(sess)
    monkeypatch.undo()
    assert (work / "model.safetensors").read_bytes() == model
    assert (work / "config.json").read_bytes() == config
    assert sorted(p.name for p in work.iterdir()) == ["config.json", "model.safetensors"]
    detach(sess)  # la séance n'est pas bloquée : le détachement se refait
    assert lm_head.detach_event(harness.read_journal(sess["dir"])) is not None
