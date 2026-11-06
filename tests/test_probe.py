"""Commande probe (story 2.4, CAP-1) : modèle Llama minuscule sur CPU, tenseurs synthétiques.

Aucun téléchargement, aucun GPU. La référence est écrite dans le test (crochets indépendants du
code testé, et `output_hidden_states` en recoupement) sur les mêmes poids.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from slm_weight_editing import harness, main, probe, workspace  # noqa: E402
from slm_weight_editing.harness import HarnessError  # noqa: E402

WORDS = ["<eos>", "<unk>", "<pad>", "combien", "font", "deux", "et", "trois", "?", "x"]
TEXTS = ["combien font deux et trois ?", "deux et x x"]
LAYERS = 2


def build_checkpoint(path: Path, dtype: torch.dtype = torch.float32) -> Path:
    vocab = {w: i for i, w in enumerate(WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<eos>", unk_token="<unk>", pad_token="<pad>"
    )
    torch.manual_seed(1)
    config = LlamaConfig(
        vocab_size=len(WORDS),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        eos_token_id=0,
        pad_token_id=2,
        bos_token_id=None,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).to(dtype)
    with torch.no_grad():
        model.model.norm.weight.uniform_(0.5, 1.5)
    path.mkdir(parents=True)
    model.save_pretrained(path)
    fast.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory) -> Path:
    return build_checkpoint(tmp_path_factory.mktemp("probe-ckpt") / "f32")


@pytest.fixture(scope="module")
def ckpt_bf16(tmp_path_factory) -> Path:
    return build_checkpoint(tmp_path_factory.mktemp("probe-ckpt-bf16") / "bf16", torch.bfloat16)


def make_ws(tmp_path: Path, ckpt: Path, **kwargs) -> Path:
    data = tmp_path / "src-data" / "gsm8k-train.jsonl"
    data.parent.mkdir(exist_ok=True)
    data.write_text('{"split": "train", "question": "1+1"}\n', encoding="utf-8")
    return workspace.create_workspace(tmp_path / "ws", "s1", 10, ckpt, [data], **kwargs)


@pytest.fixture
def ws(tmp_path, ckpt) -> Path:
    return make_ws(tmp_path, ckpt)


def session(ws: Path) -> Path:
    return ws / "sessions" / "s1"


def write_texts(ws: Path, texts: list[str] = TEXTS, name: str = "textes.jsonl") -> Path:
    path = ws / "notes" / name
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
    return path


def call(ws: Path, texts: Path | None = None, **kwargs):
    kwargs.setdefault("capture", list(probe.CAPTURES))
    return probe.run_probe(
        "s1",
        texts or write_texts(ws),
        sessions_root=ws / "sessions",
        probes_root=ws / "probes",
        **kwargs,
    )


def journal(ws: Path) -> list[dict]:
    return harness.read_journal(session(ws))


def reference(work: Path, texts: list[str]):
    """Forward de référence : crochets propres au test et `output_hidden_states`."""
    tok = AutoTokenizer.from_pretrained(work)
    model = AutoModelForCausalLM.from_pretrained(work, dtype="auto").eval()
    rows = []
    seen = {}
    for text in texts:
        ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]])
        seen.clear()
        hooks = []
        for i, layer in enumerate(model.model.layers):
            hooks.append(
                layer.register_forward_hook(
                    lambda m, a, o, i=i: seen.__setitem__(
                        f"resid_l{i}", o[0] if isinstance(o, tuple) else o
                    )
                )
            )
            hooks.append(
                layer.self_attn.register_forward_hook(
                    lambda m, a, o, i=i: seen.__setitem__(f"attn_out_l{i}", o[0])
                )
            )
            hooks.append(
                layer.mlp.down_proj.register_forward_pre_hook(
                    lambda m, a, i=i: seen.__setitem__(f"mlp_act_l{i}", a[0])
                )
            )
        hooks.append(
            model.model.norm.register_forward_hook(
                lambda m, a, o: seen.__setitem__("final_norm", o)
            )
        )
        with torch.inference_mode():
            res = model(input_ids=ids, output_hidden_states=True)
        for h in hooks:
            h.remove()
        for i in range(LAYERS):
            seen[f"resid_pre_l{i}"] = res.hidden_states[i]
        seen["logits"] = res.logits
        rows.append({k: v[0].float().numpy() for k, v in seen.items()})
    return rows


@pytest.mark.parametrize("which", ["f32", "bf16"])
def test_captures_equal_reference_forward(tmp_path, ckpt, ckpt_bf16, which):
    ws = make_ws(tmp_path, ckpt if which == "f32" else ckpt_bf16)
    entry = call(ws, positions="all")
    ref = reference(session(ws) / "work", TEXTS)
    out = ws / "probes" / "1"
    lens = [len(r["final_norm"]) for r in ref]
    assert entry["tokens"] == sum(lens) == 10
    names = [f"resid_pre_l{i}" for i in range(LAYERS)] + [f"resid_l{i}" for i in range(LAYERS)]
    names += [f"mlp_act_l{i}" for i in range(LAYERS)] + [f"attn_out_l{i}" for i in range(LAYERS)]
    names.append("final_norm")
    for name in names:
        got = np.load(out / f"{name}.npy")
        assert got.dtype == np.float32
        want = np.concatenate([r[name] for r in ref])
        assert np.array_equal(got, want), name
    ids = np.load(out / "top_logit_ids.npy")
    vals = np.load(out / "top_logit_values.npy")
    want_logits = np.concatenate([r["logits"] for r in ref])
    assert ids.shape == (10, len(WORDS)) and vals.dtype == np.float32
    top = torch.topk(torch.from_numpy(want_logits), len(WORDS), dim=-1)
    assert np.array_equal(ids, top.indices.numpy())
    assert np.array_equal(vals, top.values.numpy())
    # résidu avant bloc 1 = sortie du bloc 0 (recoupement interne)
    assert np.array_equal(np.load(out / "resid_pre_l1.npy"), np.load(out / "resid_l0.npy"))


def test_top_logits_are_twenty_when_vocab_is_large(tmp_path):
    big = tmp_path / "big"
    # même modèle, vocabulaire de 40 jetons : 20 meilleures logits par position
    vocab = {w: i for i, w in enumerate(WORDS)} | {f"w{i}": 10 + i for i in range(30)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", unk_token="<unk>")
    torch.manual_seed(2)
    cfg = LlamaConfig(
        vocab_size=40, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
    )  # fmt: skip
    big.mkdir()
    LlamaForCausalLM(cfg).save_pretrained(big)
    fast.save_pretrained(big)
    ws = make_ws(tmp_path, big)
    call(ws, capture=["final_norm"], positions="last")
    assert np.load(ws / "probes" / "1" / "top_logit_ids.npy").shape == (2, 20)


def test_positions_layers_token_ids_and_full_vocab(ws):
    entry = call(
        ws,
        capture=["resid", "final_norm"],
        layers=[1],
        positions="1:3",
        token_ids=[7, 9],
        full_vocab_positions=[3],
    )
    ref = reference(session(ws) / "work", TEXTS)
    out = ws / "probes" / "1"
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert [t["positions"] for t in manifest["texts"]] == [[1, 2], [1, 2]]
    assert [t["offset"] for t in manifest["texts"]] == [0, 2]
    assert manifest["layers"] == [1] and manifest["capture"] == ["resid", "final_norm"]
    assert manifest["texts"][0]["token_ids"] == [3, 4, 5, 6, 7, 8]
    assert not (out / "resid_l0.npy").exists() and not (out / "mlp_act_l1.npy").exists()
    got = np.load(out / "resid_l1.npy")
    want = np.concatenate([r["resid_l1"][1:3] for r in ref])
    assert np.array_equal(got, want)
    tl = np.load(out / "token_logits.npy")
    assert np.array_equal(tl, np.concatenate([r["logits"][1:3][:, [7, 9]] for r in ref]))
    fv = np.load(out / "full_vocab_logits.npy")
    assert fv.shape == (2, len(WORDS))
    assert np.array_equal(fv, np.stack([r["logits"][3] for r in ref]))
    assert manifest["full_vocab_rows"] == [
        {"text": 0, "position": 3},
        {"text": 1, "position": 3},
    ]
    assert manifest["arrays"]["token_logits"] == {"shape": [4, 2], "dtype": "float32"}
    assert manifest["arrays"]["top_logit_ids"]["dtype"] == "int64"
    assert entry["arguments"] == {}


def test_last_position(ws):
    call(ws, capture=["final_norm"], positions="last")
    arr = np.load(ws / "probes" / "1" / "final_norm.npy")
    ref = reference(session(ws) / "work", TEXTS)
    assert np.array_equal(arr, np.stack([r["final_norm"][-1] for r in ref]))


def test_journal_lines_and_batch_budget_untouched(ws):
    before = harness.read_state(session(ws))
    call(ws, argv={"capture": ["resid"]})
    lines = journal(ws)
    reserve, entry = lines[-2], lines[-1]
    assert [e["event"] for e in lines] == ["open", "probe_reserve", "probe"]
    assert (reserve["probe"], reserve["tokens"], reserve["budget"]) == (1, 10, 2_000_000)
    assert reserve["estimated_bytes"] > 0
    assert entry["probe"] == 1 and entry["tokens"] == 10 and entry["remaining"] == 2_000_000 - 10
    assert entry["arguments"] == {"capture": ["resid"]}
    work = session(ws) / "work"
    assert entry["tensors_sha256"] == harness.tensor_hashes(work)
    assert entry["files_sha256"] == harness.work_file_hashes(work)
    manifest = ws / "probes" / "1" / "manifest.json"
    import hashlib

    assert entry["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert "final_norm.npy" in entry["output_files"]
    assert harness.read_state(session(ws)) == before
    assert harness.used_batches(session(ws)) == 0
    assert harness.used_probe_tokens(session(ws)) == 10


def test_second_call_uses_next_number_and_accumulates(ws):
    call(ws, capture=["final_norm"])
    entry = call(ws, capture=["final_norm"])
    assert entry["probe"] == 2 and entry["remaining"] == 2_000_000 - 20
    assert (ws / "probes" / "2" / "manifest.json").is_file()


def test_budget_exact_then_refused_before_anything(tmp_path, ckpt, monkeypatch):
    ws = make_ws(tmp_path, ckpt, probe_budget=20)
    call(ws, capture=["final_norm"])
    call(ws, capture=["final_norm"])  # 20/20 : atteint exactement
    assert harness.used_probe_tokens(session(ws)) == 20
    log_before = (session(ws) / "journal.jsonl").read_bytes()
    monkeypatch.setattr(
        AutoModelForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("modèle chargé")
    )
    with pytest.raises(HarnessError, match=r"budget de probe dépassé : 20 jetons utilisés \+ 10 "):
        call(ws, capture=["final_norm"])
    assert (session(ws) / "journal.jsonl").read_bytes() == log_before
    assert sorted(p.name for p in (ws / "probes").iterdir()) == ["1", "2"]


def test_refused_budget_creates_no_probe_dir(tmp_path, ckpt):
    ws = make_ws(tmp_path, ckpt, probe_budget=9)
    with pytest.raises(HarnessError, match="budget de probe dépassé"):
        call(ws, capture=["final_norm"])
    assert not (ws / "probes").exists()
    assert [e["event"] for e in journal(ws)] == ["open"]


def test_hand_edited_state_cannot_reset_probe_count(ws):
    call(ws, capture=["final_norm"])
    state = session(ws) / "state.json"
    raw = json.loads(state.read_text("utf-8"))
    raw["used"] = 0
    state.write_text(json.dumps(raw), "utf-8")
    assert harness.used_probe_tokens(session(ws)) == 10


def test_too_many_texts_refused_with_ceiling_and_value(ws):
    path = write_texts(ws, ["deux"] * 65)
    with pytest.raises(HarnessError, match="plafond de 64 textes par appel : 65 demandés"):
        call(ws, path)
    assert [e["event"] for e in journal(ws)] == ["open"]


def test_exactly_64_texts_accepted(ws):
    entry = call(ws, write_texts(ws, ["deux"] * 64), capture=["final_norm"], positions="last")
    assert entry["tokens"] == 64


def test_text_over_8192_tokens_refused(ws):
    path = write_texts(ws, [" ".join(["x"] * 8193)])
    with pytest.raises(
        HarnessError, match=r"plafond de 8192 jetons par texte : le texte 0 en a 8193"
    ):
        call(ws, path, capture=["final_norm"])
    assert [e["event"] for e in journal(ws)] == ["open"]


def test_text_of_8192_tokens_passes_the_cap(tmp_path, ckpt):
    ws = make_ws(tmp_path, ckpt, probe_budget=10_000)
    path = write_texts(ws, [" ".join(["x"] * 8192)])
    entry = call(ws, path, capture=["final_norm"], positions="last")
    assert entry["tokens"] == 8192 and entry["remaining"] == 10_000 - 8192


def test_estimated_output_over_2gb_refused_before_model(tmp_path, ckpt, monkeypatch):
    ws = make_ws(tmp_path, ckpt, probe_budget=10_000_000)
    path = write_texts(ws, [" ".join(["x"] * 8000)] * 64)
    cfg = session(ws) / "work" / "config.json"
    raw = json.loads(cfg.read_text("utf-8"))
    raw.update(hidden_size=4096, intermediate_size=11008, num_hidden_layers=32)
    cfg.write_text(json.dumps(raw), "utf-8")
    monkeypatch.setattr(
        AutoModelForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("modèle chargé")
    )
    with pytest.raises(HarnessError, match="plafond de sortie de 2000000000 octets"):
        call(ws, path, capture=["resid", "mlp_act", "attn_out"])
    assert [e["event"] for e in journal(ws)] == ["open"]


def test_estimate_output_bytes_formula():
    dims = probe.Dims(hidden=8, intermediate=16, layers=2, vocab=10)
    one = probe.estimate_output_bytes(dims, 1, ["final_norm"], 2, 0, 0)
    assert one == int((20 * 12 + 8 * 4) * 1.01) + 1
    full = probe.estimate_output_bytes(dims, 3, list(probe.CAPTURES), 2, 2, 1)
    per_row = 20 * 12 + 2 * 4 + 2 * 2 * 8 * 4 + 2 * 16 * 4 + 2 * 8 * 4 + 8 * 4
    assert full == int((3 * per_row + 10 * 4) * 1.01) + 1


def test_disk_cap_cumulative_refused_before_calculation(tmp_path, ckpt):
    ws = make_ws(tmp_path, ckpt, probe_disk_max=5000)
    first = call(ws, capture=["final_norm"], positions="last")
    reserved = journal(ws)[1]["estimated_bytes"]
    assert first["probe"] == 1 and reserved < 5000
    with pytest.raises(HarnessError, match=r"plafond disque de probe de 5000 octets"):
        call(ws, capture=["resid", "mlp_act", "attn_out"], positions="all")
    assert harness.used_probe_tokens(session(ws)) == 10


def test_session_without_probe_budget_refuses_with_clear_message(tmp_path, ckpt):
    root = tmp_path / "plain"
    harness.open_session("s1", 10, ckpt, sessions_root=root)
    texts = tmp_path / "t.jsonl"
    texts.write_text('{"text": "deux"}\n', "utf-8")
    log = (root / "s1" / "journal.jsonl").read_bytes()
    with pytest.raises(HarnessError, match="ouverte sans budget de probe"):
        probe.run_probe("s1", texts, capture=["final_norm"], sessions_root=root)
    assert (root / "s1" / "journal.jsonl").read_bytes() == log
    assert not (root / "s1" / "probes").exists()


def test_open_records_probe_fields_only_when_given(tmp_path, ckpt):
    ws = make_ws(tmp_path, ckpt, probe_budget=123, probe_disk_max=456)
    opened = journal(ws)[0]
    assert (opened["probe_budget"], opened["probe_disk_max"]) == (123, 456)
    info = json.loads((ws / "workspace.json").read_text("utf-8"))
    assert (info["probe_budget"], info["probe_disk_max"]) == (123, 456)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(HarnessError, match="probe_budget"):
            harness.open_session("z", 10, ckpt, sessions_root=tmp_path / "r", probe_budget=bad)


def test_workspace_defaults_are_two_million_and_twenty_gb(ws):
    opened = journal(ws)[0]
    assert (opened["probe_budget"], opened["probe_disk_max"]) == (2_000_000, 20 * 1000**3)
    assert "`2 000 000` jetons" in (ws / "CONSIGNE.md").read_text("utf-8")
    assert "<BUDGET_PROBE>" not in (ws / "CLAUDE.md").read_text("utf-8")


def test_frozen_session_refuses_probe(ws):
    harness.update_state(session(ws), frozen=True)
    with pytest.raises(HarnessError, match="séance gelée : plus aucune probe"):
        call(ws, capture=["final_norm"])
    assert [e["event"] for e in journal(ws)] == ["open"]


def test_lock_present_refuses_probe_and_is_released_after(ws):
    harness.read_state(session(ws))
    lock = session(ws) / "session.lock"
    lock.write_text(json.dumps({"pid": os.getpid(), "acquired_at": "x"}), "utf-8")
    with pytest.raises(HarnessError, match="un lot tourne déjà"):
        call(ws, capture=["final_norm"])
    lock.unlink()
    call(ws, capture=["final_norm"])
    assert not lock.exists()


def test_no_generate_no_grad_inference_mode_and_reloads_work(ws, monkeypatch):
    from transformers import GenerationMixin

    def boom(*a, **k):
        raise AssertionError("generate appelé")

    monkeypatch.setattr(GenerationMixin, "generate", boom)
    seen = {}
    real = AutoModelForCausalLM.from_pretrained

    def spy(path, *a, **k):
        seen["path"] = Path(path)
        model = real(path, *a, **k)
        seen["model"] = model
        orig = model.forward

        def forward(*fa, **fk):
            seen["inference"] = torch.is_inference_mode_enabled()
            return orig(*fa, **fk)

        model.forward = forward
        return model

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", spy)
    call(ws, capture=["final_norm"])
    assert seen["path"] == session(ws) / "work"
    assert seen["inference"] is True
    assert all(not p.requires_grad for p in seen["model"].parameters())
    assert not seen["model"].training


def test_crash_after_reservation_consumes_tokens(ws, monkeypatch):
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("plantage")),
    )
    with pytest.raises(RuntimeError, match="plantage"):
        call(ws, capture=["final_norm"])
    assert [e["event"] for e in journal(ws)] == ["open", "probe_reserve"]
    assert harness.used_probe_tokens(session(ws)) == 10
    assert not (session(ws) / "session.lock").exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"layers": [2]}, r"--layers refusé : le modèle a 2 couches \(0 à 1\)"),
        ({"token_ids": [10]}, r"--token-ids refusé : le vocabulaire a 10 jetons"),
        ({"full_vocab_positions": [6]}, r"--full-vocab-positions 6 refusé : un texte n'a que 4"),
        ({"positions": "9:12"}, r"--positions 9:12 refusé : le texte n'a que 6 jetons"),
        ({"positions": "3:1"}, r"--positions '3:1' refusé"),
        ({"positions": "first"}, r"--positions 'first' refusé"),
    ],
)
def test_bad_arguments_refused_before_reservation(ws, kwargs, message):
    with pytest.raises(HarnessError, match=message):
        call(ws, capture=["final_norm"], **kwargs)
    assert [e["event"] for e in journal(ws)] == ["open"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "ne contient aucun texte"),
        ("pas du json\n", "ligne 1 refusée"),
        ('{"text": ""}\n', "ligne 1 refusée"),
        ('{"text": "a", "token_ids": [1]}\n', "ligne 1 refusée"),
        ('{"text": "deux"}\n[1]\n', "ligne 2 refusée"),
    ],
)
def test_bad_texts_file_refused(ws, content, message):
    path = ws / "notes" / "bad.jsonl"
    path.write_text(content, "utf-8")
    with pytest.raises(HarnessError, match=message):
        call(ws, path, capture=["final_norm"])


def test_capture_option_validated():
    assert probe.parse_captures("final_norm,resid") == ["resid", "final_norm"]
    with pytest.raises(HarnessError, match="--capture 'logits' refusé"):
        probe.parse_captures("resid,logits")
    with pytest.raises(HarnessError, match="--capture obligatoire"):
        probe.parse_captures(None)
    with pytest.raises(HarnessError, match="--layers"):
        probe._int_list("1,a", "--layers")
    with pytest.raises(HarnessError, match="--layers"):
        probe._int_list("-1", "--layers")
    assert probe._int_list("3,1,3", "--layers") == [1, 3]


def test_only_probe_dir_and_journal_are_written(ws):
    def tree(root: Path) -> dict[str, bytes]:
        return {
            p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }

    texts = write_texts(ws)
    before = tree(ws)
    call(ws, texts, capture=["resid", "final_norm"])
    after = tree(ws)
    changed = {k for k in after if before.get(k) != after[k]}
    assert changed == {k for k in after if k.startswith("probes/1/")} | {
        "sessions/s1/journal.jsonl"
    }
    assert set(before) <= set(after)


def test_cli_probe_flow_and_exit_codes(ws, monkeypatch, capsys):
    monkeypatch.chdir(ws)
    write_texts(ws)
    args = [
        "probe",
        "--session",
        "s1",
        "--sessions-root",
        "sessions",
        "--texts",
        "notes/textes.jsonl",
    ]
    assert main([*args, "--capture", "final_norm", "--positions", "last"]) == 0
    out = capsys.readouterr().out
    assert "Probe 1 : 10 jetons, reste 1999990 jetons de probe" in out
    assert (ws / "probes" / "1" / "final_norm.npy").is_file()
    last = journal(ws)[-1]
    assert last["arguments"]["texts"] == "notes/textes.jsonl"
    assert len(last["arguments"]["texts_sha256"]) == 64
    assert last["arguments"]["capture"] == ["final_norm"]
    assert main([*args, "--capture", "bogus"]) == 2
    assert "--capture 'bogus' refusé" in capsys.readouterr().err
    assert main([*args, "--capture", "final_norm", "--layers", "9"]) == 2
    assert "--layers refusé" in capsys.readouterr().err
    assert main(["probe", "--session", "s1", "--sessions-root", "sessions", "--capture",
                 "final_norm", "--texts", "notes/absent.jsonl"]) == 1  # fmt: skip
    assert "--texts illisible" in capsys.readouterr().err


def test_cli_texts_outside_workspace_refused(ws, tmp_path, monkeypatch, capsys):
    outside = tmp_path / "dehors.jsonl"
    outside.write_text('{"text": "deux"}\n', "utf-8")
    monkeypatch.chdir(ws)
    code = main(["probe", "--session", "s1", "--sessions-root", "sessions", "--texts",
                 str(outside), "--capture", "final_norm"])  # fmt: skip
    assert code == 2
    assert "--texts refusé : fichier hors de" in capsys.readouterr().err
    assert [e["event"] for e in journal(ws)] == ["open"]


def test_cli_other_session_refused_from_workspace(ws, monkeypatch, capsys):
    monkeypatch.chdir(ws)
    write_texts(ws)
    code = main(["probe", "--session", "autre", "--sessions-root", "sessions",
                 "--texts", "notes/textes.jsonl", "--capture", "final_norm"])  # fmt: skip
    assert code == 2
    assert "séance refusée" in capsys.readouterr().err


def test_script_reads_probe_capture_in_mmap_mode(ws):
    from slm_weight_editing import script_runner

    call(ws, capture=["resid"], layers=[1])
    code = (
        "import numpy as np\n"
        "a = np.load('probes/1/resid_l1.npy', mmap_mode='r')\n"
        "print('SHAPE', a.shape, float(a.sum()))\n"
    )
    (ws / "scripts" / "lit.py").write_text(code, encoding="utf-8")
    entry = script_runner.run_script("scripts/lit.py", "s1", workspace=ws, timeout=60)
    assert entry["returncode"] == 0, entry
    log = (ws / entry["log"]).read_text("utf-8")
    want = np.load(ws / "probes" / "1" / "resid_l1.npy")
    assert f"SHAPE (10, 8) {float(want.sum())}" in log


def test_script_cannot_write_probes(ws):
    from slm_weight_editing import script_runner
    from slm_weight_editing._sandbox import REFUSED

    call(ws, capture=["final_norm"])
    (ws / "scripts" / "ecrit.py").write_text(
        "open('probes/1/final_norm.npy', 'wb').write(b'x')\n", encoding="utf-8"
    )
    entry = script_runner.run_script("scripts/ecrit.py", "s1", workspace=ws, timeout=60)
    assert entry["returncode"] == REFUSED
    assert np.load(ws / "probes" / "1" / "final_norm.npy").shape == (10, 8)


def test_settings_deny_probes_and_allow_probe(ws):
    perms = json.loads((ws / ".claude" / "settings.json").read_text("utf-8"))["permissions"]
    assert "Edit(./probes/**)" in perms["deny"]
    assert "Bash(uv run slm-weight-editing probe *)" in perms["allow"]
    assert "PowerShell(uv run slm-weight-editing probe *)" in perms["allow"]
    assert "Edit(./probes/**)" not in perms["allow"]


def test_audit_flags_probe_writes_and_unbound_texts(ws):
    from slm_weight_editing import audit_transcript as audit

    ctx = audit.Context.build(ws, workspace.repo_root(), "s1")
    base = "uv run slm-weight-editing probe --session s1 --sessions-root sessions"
    ok = f"{base} --texts notes/textes.jsonl --capture final_norm --positions last"
    assert audit.check_shell(ok, ctx) == []
    for texts in ("../dehors.jsonl", "C:/autre/t.jsonl", "/tmp/t.jsonl"):
        found = audit.check_shell(f"{base} --texts {texts} --capture final_norm", ctx)
        assert {r for r, _ in found} == {"hors-espace-seance"}, texts
    assert not audit._write_allowed("probes/1/final_norm.npy", ctx)


def test_audit_accepts_consigne_probe_command(ws):
    from slm_weight_editing import audit_transcript as audit

    ctx = audit.Context.build(ws, workspace.repo_root(), "s1")
    consigne = (ws / "CONSIGNE.md").read_text("utf-8")
    line = next(
        ln for ln in consigne.splitlines() if ln.startswith("uv run slm-weight-editing probe")
    )
    import re

    command = re.sub(r"\s*\[[^\]]*\]", "", line)
    assert "<" not in command and "|" not in command
    assert audit.check_shell(command, ctx) == []
    full = line.replace("[", "").replace("]", "")
    assert audit.check_shell(full, ctx) == []


def test_texts_with_utf8_bom_accepted(ws):
    path = ws / "notes" / "bom.jsonl"
    path.write_bytes(
        b"\xef\xbb\xbf" + "".join(json.dumps({"text": t}) + "\n" for t in TEXTS).encode()
    )
    assert call(ws, path, capture=["final_norm"], positions="last")["tokens"] == 10


def test_logits_never_materialised_for_all_positions(ws, monkeypatch):
    # Revue passe 1 : jamais [jetons, vocabulaire] d'un coup ; blocs de 4 positions au plus.
    monkeypatch.setattr(probe, "LOGIT_CHUNK", 4)
    sizes, keeps = [], []
    real = LlamaForCausalLM.forward

    def spy(self, *args, **kwargs):
        keeps.append(kwargs.get("logits_to_keep"))
        out = real(self, *args, **kwargs)
        sizes.append(out.logits.shape[1])
        return out

    monkeypatch.setattr(LlamaForCausalLM, "forward", spy)
    call(ws, capture=["final_norm"], positions="all", token_ids=[7, 9])
    assert keeps == [1, 1] and sizes == [1, 1]
    monkeypatch.undo()
    ref = reference(session(ws) / "work", TEXTS)
    out = ws / "probes" / "1"
    want = torch.from_numpy(np.concatenate([r["logits"] for r in ref]))
    top = torch.topk(want, len(WORDS), dim=-1)
    assert np.array_equal(np.load(out / "top_logit_values.npy"), top.values.numpy())
    assert np.array_equal(np.load(out / "top_logit_ids.npy"), top.indices.numpy())
    assert np.array_equal(np.load(out / "token_logits.npy"), want[:, [7, 9]].numpy())


def test_logits_of_a_position_do_not_depend_on_how_many_are_asked(tmp_path):
    # Revue passe 2 : une matmul d'une seule ligne (gemv) ne rend pas les mêmes bits qu'un bloc ;
    # full_vocab_logits, token_logits et top_logit_values d'une même position doivent concorder,
    # que l'on demande une position ou toutes.
    wide = tmp_path / "wide"
    fast = PreTrainedTokenizerFast.from_pretrained(build_checkpoint(tmp_path / "tok"))
    torch.manual_seed(3)
    cfg = LlamaConfig(
        vocab_size=4096, hidden_size=512, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=1, max_position_embeddings=64,
        tie_word_embeddings=True,
    )  # fmt: skip
    wide.mkdir()
    LlamaForCausalLM(cfg).save_pretrained(wide)
    fast.save_pretrained(wide)
    ws = make_ws(tmp_path, wide)
    call(ws, capture=["final_norm"], positions="all", token_ids=[7, 9], full_vocab_positions=[3])
    call(ws, capture=["final_norm"], positions="last", token_ids=[7, 9])
    every, last = ws / "probes" / "1", ws / "probes" / "2"
    fv = np.load(every / "full_vocab_logits.npy")
    texts = json.loads((every / "manifest.json").read_text("utf-8"))["texts"]
    rows = [t["offset"] + 3 for t in texts]
    top = torch.topk(torch.from_numpy(fv), 20, dim=-1)
    assert np.array_equal(np.load(every / "top_logit_values.npy")[rows], top.values.numpy())
    assert np.array_equal(np.load(every / "token_logits.npy")[rows], fv[:, [7, 9]])
    ends = [t["offset"] + len(t["positions"]) - 1 for t in texts]
    assert np.array_equal(
        np.load(last / "token_logits.npy"), np.load(every / "token_logits.npy")[ends]
    )
