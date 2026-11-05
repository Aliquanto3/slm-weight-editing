"""Harnais d'édition : budget, refus, journal et génération sans gradient (tenseurs synthétiques).

Petit modèle Llama de config aléatoire et tokenizer WordLevel construits en mémoire, sauvegardés
dans tmp_path : ni modèle réel ni téléchargement. La norme finale est mise à zéro pour rendre
la sortie déterministe : logits tous nuls, donc l'argmax glouton choisit toujours le jeton 0.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from slm_weight_editing import harness, main  # noqa: E402
from slm_weight_editing.harness import HarnessError  # noqa: E402

WORDS = ["<eos>", "<unk>", "<pad>", "combien", "font", "deux", "et", "trois", "?", "x"]


def build_checkpoint(path: Path, eos_id: int) -> Path:
    vocab = {w: i for i, w in enumerate(WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token=WORDS[eos_id], unk_token="<unk>", pad_token="<pad>"
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
        eos_token_id=eos_id,
        pad_token_id=2,
        bos_token_id=None,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    with torch.no_grad():
        model.model.norm.weight.zero_()
    path.mkdir(parents=True)
    model.save_pretrained(path)
    fast.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def stop_ckpt(tmp_path_factory) -> Path:
    """Le jeton 0 est <eos> : chaque génération s'arrête au premier jeton."""
    return build_checkpoint(tmp_path_factory.mktemp("ckpt") / "stop", eos_id=0)


@pytest.fixture
def train_file(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "gsm8k" / "train.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"id": f"g{i}", "question": "combien font deux et trois ?", "split": "train"}
        for i in range(3)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


def open_s(root: Path, ckpt: Path, budget: int = 10, sid: str = "s1"):
    return harness.open_session(sid, budget, ckpt, base_revision="rev0", sessions_root=root)


@pytest.mark.parametrize("budget", [10, 25, 50])
def test_open_writes_state_and_journal(root, stop_ckpt, budget):
    state = open_s(root, stop_ckpt, budget)
    d = root / "s1"
    raw = json.loads((d / "state.json").read_text("utf-8"))
    assert (raw["budget"], raw["used"], raw["frozen"], raw["base_revision"]) == (
        budget,
        0,
        False,
        "rev0",
    )
    assert state.remaining == budget
    assert harness.read_state(d) == state
    assert sorted(p.name for p in (d / "work").glob("*.safetensors")) == ["model.safetensors"]
    open_line = harness.read_journal(d)[0]
    norm = load_file(stop_ckpt / "model.safetensors")["model.norm.weight"]
    expected_norm = hashlib.sha256(norm.numpy().tobytes()).hexdigest()
    assert open_line["base_tensors_sha256"]["model.norm.weight"] == expected_norm
    config_sha = hashlib.sha256((stop_ckpt / "config.json").read_bytes()).hexdigest()
    assert open_line["base_files_sha256"]["config.json"] == config_sha
    assert "tokenizer.json" in open_line["base_files_sha256"]
    assert not any(k.endswith(".safetensors") for k in open_line["base_files_sha256"])
    assert harness.read_journal(d) == [
        {
            "event": "open",
            "session_id": "s1",
            "budget": budget,
            "base_checkpoint": str(stop_ckpt.resolve()),
            "base_revision": "rev0",
            "base_tensors_sha256": open_line["base_tensors_sha256"],
            "base_files_sha256": open_line["base_files_sha256"],
            "at": state.created_at,
        }
    ]


@pytest.mark.parametrize("budget", [0, 9, 11, 24, 51, 100, -10, True, 10.0, "10"])
def test_open_refuses_budget_outside_allowed(root, stop_ckpt, budget):
    with pytest.raises(HarnessError, match="10, 25 ou 50"):
        open_s(root, stop_ckpt, budget)
    assert not (root / "s1").exists()


def test_budget_cannot_change_after_open(root, stop_ckpt, train_file):
    open_s(root, stop_ckpt, 10)
    d = root / "s1"
    with pytest.raises(HarnessError, match="déjà ouverte"):
        open_s(root, stop_ckpt, 50)
    with pytest.raises(HarnessError, match="non modifiables"):
        harness.update_state(d, budget=50)
    with pytest.raises(HarnessError, match="ne se modifie pas"):
        state = harness.read_state(d)
        harness._write_state(d, harness.dataclasses.replace(state, budget=25))
    assert harness.read_state(d).budget == 10
    # Écriture à la main dans state.json : refusée au lot suivant.
    raw = json.loads((d / "state.json").read_text("utf-8"))
    raw["budget"] = 50
    (d / "state.json").write_text(json.dumps(raw), "utf-8")
    with pytest.raises(HarnessError, match="différent de celui de l'ouverture"):
        harness.run_batch("s1", "gsm8k", "train", train_file, sessions_root=root)
    raw["budget"] = 12
    (d / "state.json").write_text(json.dumps(raw), "utf-8")
    with pytest.raises(HarnessError, match="10, 25 ou 50"):
        harness.read_state(d)


def test_cli_rejects_budget_option_on_run_batch(root, stop_ckpt, train_file):
    open_s(root, stop_ckpt)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run-batch",
                "--session",
                "s1",
                "--dataset",
                "gsm8k",
                "--split",
                "train",
                "--data",
                str(train_file),
                "--sessions-root",
                str(root),
                "--budget",
                "50",
            ]
        )
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        main(
            [
                "run-batch",
                "--session",
                "s1",
                "--dataset",
                "gsm8k",
                "--split",
                "train",
                "--data",
                str(train_file),
                "--sessions-root",
                str(root),
                "--max-new-tokens",
                "10",
            ]
        )


def test_batches_decrement_and_refuse_beyond_budget(root, stop_ckpt, train_file):
    open_s(root, stop_ckpt, 10)
    d = root / "s1"
    for n in range(1, 11):
        entry = harness.run_batch(
            "s1", "gsm8k", "train", train_file, start=0, count=1, sessions_root=root
        )
        assert (entry["batch"], entry["remaining"]) == (n, 10 - n)
        assert harness.read_state(d).used == n
    journal_before = (d / "journal.jsonl").read_bytes()
    with pytest.raises(HarnessError, match="budget épuisé : 10/10"):
        harness.run_batch("s1", "gsm8k", "train", train_file, sessions_root=root)
    assert (d / "journal.jsonl").read_bytes() == journal_before
    assert harness.read_state(d).used == 10
    assert not (d / "session.lock").exists()


def test_cli_session_flow_and_exit_codes(root, stop_ckpt, train_file, capsys):
    common = ["--sessions-root", str(root)]
    assert (
        main(
            [
                "session-open",
                "--session",
                "c1",
                "--budget",
                "10",
                "--base-checkpoint",
                str(stop_ckpt),
                *common,
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "session-open",
                "--session",
                "c1",
                "--budget",
                "25",
                "--base-checkpoint",
                str(stop_ckpt),
                *common,
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "session-open",
                "--session",
                "c2",
                "--budget",
                "30",
                "--base-checkpoint",
                str(stop_ckpt),
                *common,
            ]
        )
        == 2
    )
    batch = ["run-batch", "--session", "c1", "--dataset", "gsm8k", "--data", str(train_file)]
    assert main([*batch, "--split", "train", "--count", "2", *common]) == 0
    assert main([*batch, "--split", "test", *common]) == 2
    assert main(["session-status", "--session", "c1", *common]) == 0
    out = capsys.readouterr()
    assert "Lot 1/10 : 2 problèmes, reste 9 lots" in out.out
    assert "Séance c1 : 1/10 lots, reste 9, gelée : non" in out.out
    assert "split 'test' refusé" in out.err


def test_frozen_session_refuses_batches(root, stop_ckpt, train_file):
    open_s(root, stop_ckpt)
    d = root / "s1"
    raw = json.loads((d / "state.json").read_text("utf-8"))
    raw["frozen"] = True
    (d / "state.json").write_text(json.dumps(raw), "utf-8")
    with pytest.raises(HarnessError, match="gelée"):
        harness.run_batch("s1", "gsm8k", "train", train_file, sessions_root=root)
    assert len(harness.read_journal(d)) == 1
    assert harness.read_state(d).used == 0


def test_update_state_freezes_and_never_unfreezes(root, stop_ckpt):
    open_s(root, stop_ckpt)
    d = root / "s1"
    assert harness.update_state(d, frozen=True).frozen is True
    with pytest.raises(HarnessError, match="ne se dégèle pas"):
        harness.update_state(d, frozen=False)
    with pytest.raises(HarnessError, match="non modifiables"):
        harness.update_state(d, used=0)


def test_batch_over_512_problems_refused(root, stop_ckpt, tmp_path):
    open_s(root, stop_ckpt)
    big = tmp_path / "math_train.jsonl"
    big.write_text(
        "".join(json.dumps({"problem": "x", "split": "train"}) + "\n" for _ in range(513)), "utf-8"
    )
    with pytest.raises(HarnessError, match="lot de 513 problèmes refusé : 512 au plus"):
        harness.run_batch("s1", "math", "train", big, sessions_root=root)
    with pytest.raises(HarnessError, match="lot de 513"):
        harness.run_batch("s1", "math", "train", big, count=600, sessions_root=root)
    assert harness.load_problems(big, 0, 512)[-1]["index"] == 511
    assert len(harness.load_problems(big, 1)) == 512
    assert harness.read_state(root / "s1").used == 0


@pytest.mark.parametrize(
    ("dataset", "split", "message"),
    [
        ("gsm8k", "test", "split 'test' refusé"),
        ("math", "validation", "split 'validation' refusé"),
        ("gsm8k", "Train", "split 'Train' refusé"),
        ("mmlu", "train", "jeu 'mmlu' refusé"),
        ("GSM8K", "train", "jeu 'GSM8K' refusé"),
    ],
)
def test_refuses_dataset_and_split(root, stop_ckpt, train_file, dataset, split, message):
    open_s(root, stop_ckpt)
    with pytest.raises(HarnessError, match=message):
        harness.run_batch("s1", dataset, split, train_file, sessions_root=root)


def test_refuses_heldout_paths_before_reading(tmp_path, monkeypatch):
    held = tmp_path / "heldout" / "gsm8k"
    held.mkdir(parents=True)
    (held / "train.jsonl").write_text('{"question": "x"}\n', "utf-8")
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path / "data")
    cases = [
        held / "train.jsonl",
        Path("../heldout/gsm8k/train.jsonl"),
        tmp_path / "data" / ".." / "HeldOut" / "train.jsonl",
        tmp_path / "heldout" / "absent" / "train.jsonl",  # inexistant : refusé avant tout accès
    ]
    for path in cases:
        with pytest.raises(HarnessError, match="sous heldout/"):
            harness.check_data_request("gsm8k", "train", path)


def test_refuses_symlink_into_heldout(tmp_path):
    held = tmp_path / "heldout"
    held.mkdir()
    (held / "train.jsonl").write_text('{"question": "x"}\n', "utf-8")
    link = tmp_path / "data_link"
    try:
        link.symlink_to(held, target_is_directory=True)
    except OSError:
        pytest.skip("liens symboliques non autorisés sur ce poste")
    with pytest.raises(HarnessError, match="sous heldout/"):
        harness.check_data_request("gsm8k", "train", link / "train.jsonl")


def test_refuses_non_train_file_name_and_record(tmp_path):
    test_file = tmp_path / "test.jsonl"
    with pytest.raises(HarnessError, match="ne porte pas « train »"):
        harness.check_data_request("gsm8k", "train", test_file)
    mixed = tmp_path / "train.jsonl"
    mixed.write_text(
        '{"question": "a", "split": "train"}\n{"question": "b", "split": "test"}\n', "utf-8"
    )
    with pytest.raises(HarnessError, match=r"train.jsonl:2 : split 'test' refusé"):
        harness.load_problems(mixed)
    # Un enregistrement sans champ split est refusé, pas supposé train.
    mixed.write_text('{"question": "a", "split": "train"}\n{"question": "b"}\n', "utf-8")
    with pytest.raises(HarnessError, match=r"train.jsonl:2 : split None refusé"):
        harness.load_problems(mixed)


def test_journal_line_content(root, stop_ckpt, train_file):
    open_s(root, stop_ckpt, 25)
    entry = harness.run_batch("s1", "gsm8k", "train", train_file, start=1, sessions_root=root)
    journal = harness.read_journal(root / "s1")
    assert journal[-1] == entry
    assert (entry["event"], entry["batch"], entry["budget"], entry["remaining"]) == (
        "batch",
        1,
        25,
        24,
    )
    s = entry["settings"]
    assert (s["do_sample"], s["num_beams"], s["temperature"], s["max_new_tokens"]) == (
        False,
        1,
        None,
        4096,
    )
    assert (s["inference_mode"], s["requires_grad"], s["prompt_format"]) == (True, False, "raw")
    assert [p["id"] for p in entry["problems"]] == ["g1", "g2"]
    assert [p["index"] for p in entry["problems"]] == [1, 2]
    assert entry["problems"][0]["prompt"] == "combien font deux et trois ?"
    assert [(o["id"], o["new_tokens"], o["done_reason"], o["text"]) for o in entry["outputs"]] == [
        ("g1", 1, "stop", ""),
        ("g2", 1, "stop", ""),
    ]
    assert entry["data_sha256"] == hashlib.sha256(train_file.read_bytes()).hexdigest()
    hashes = entry["tensors_sha256"]
    assert "model.norm.weight" in hashes and len(hashes) == len(
        harness.tensor_hashes(root / "s1" / "work")
    )


def test_weights_reloaded_from_work_each_batch(root, stop_ckpt, train_file):
    from safetensors.torch import load_file, save_file

    open_s(root, stop_ckpt)
    work_file = root / "s1" / "work" / "model.safetensors"
    first = harness.run_batch("s1", "gsm8k", "train", train_file, count=1, sessions_root=root)
    tensors = load_file(work_file)
    tensors["model.norm.weight"] = torch.ones_like(tensors["model.norm.weight"])
    save_file(tensors, work_file, metadata={"format": "pt"})
    second = harness.run_batch("s1", "gsm8k", "train", train_file, count=1, sessions_root=root)
    a, b = first["tensors_sha256"], second["tensors_sha256"]
    assert a["model.norm.weight"] != b["model.norm.weight"]
    expected = hashlib.sha256(tensors["model.norm.weight"].numpy().tobytes()).hexdigest()
    assert b["model.norm.weight"] == expected
    assert {k for k in a if a[k] != b[k]} == {"model.norm.weight"}


def test_no_gradient_no_backward_no_optimizer(root, stop_ckpt, train_file, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("backward ou optimiseur appelé par le harnais")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    seen = []

    def check(module, inputs, output):
        seen.append(
            (
                torch.is_inference_mode_enabled(),
                torch.is_grad_enabled(),
                any(p.requires_grad for p in module.parameters()),
            )
        )

    handle = torch.nn.modules.module.register_module_forward_hook(check)
    try:
        open_s(root, stop_ckpt)
        harness.run_batch("s1", "gsm8k", "train", train_file, sessions_root=root)
    finally:
        handle.remove()
    assert seen and set(seen) == {(True, False, False)}
    source = Path(harness.__file__).read_text("utf-8")
    for word in ("backward", "torch.optim", "loss", "zero_grad"):
        assert word not in source


def test_length_done_reason_at_fixed_cap(tmp_path, train_file):
    """Le jeton 0 n'est pas <eos> : la génération va jusqu'au plafond de 4096 jetons."""
    ckpt = build_checkpoint(tmp_path / "length", eos_id=1)
    root = tmp_path / "sessions"
    harness.open_session("L", 10, ckpt, sessions_root=root)
    entry = harness.run_batch("L", "gsm8k", "train", train_file, count=1, sessions_root=root)
    out = entry["outputs"][0]
    assert (out["new_tokens"], out["done_reason"]) == (4096, "length")
    assert out["text"].split() == ["<eos>"] * 4096


def test_lock_refuses_concurrent_batch(root, stop_ckpt, train_file):
    open_s(root, stop_ckpt)
    write_lock(root / "s1", os.getpid())
    with pytest.raises(HarnessError, match=rf"un lot tourne déjà.*{os.getpid()} est vivant"):
        harness.run_batch("s1", "gsm8k", "train", train_file, sessions_root=root)
    assert harness.read_state(root / "s1").used == 0


def write_lock(directory, pid):
    (directory / "session.lock").write_text(
        json.dumps({"pid": pid, "acquired_at": "2026-11-05T10:00:00+00:00"}), "utf-8"
    )


def dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_pid_alive_current_and_finished_child():
    assert harness.pid_alive(os.getpid()) is True
    assert harness.pid_alive(dead_pid()) is False
    assert harness.pid_alive(0) is False


def test_lock_holds_pid_and_time_then_is_removed(root, stop_ckpt):
    open_s(root, stop_ckpt)
    d = root / "s1"
    with harness._session_lock(d):
        data = json.loads((d / "session.lock").read_text("utf-8"))
        assert set(data) == {"pid", "acquired_at"} and data["pid"] == os.getpid()
        datetime.fromisoformat(data["acquired_at"])
    assert not (d / "session.lock").exists()
    harness.update_state(d, frozen=True)
    assert not (d / "session.lock").exists()


def test_lock_of_dead_process_is_refused_with_unlock_hint(root, stop_ckpt):
    open_s(root, stop_ckpt)
    pid = dead_pid()
    write_lock(root / "s1", pid)
    with pytest.raises(HarnessError, match=rf"{pid} est mort.*seul Anaël.*session-unlock"):
        harness.update_state(root / "s1", frozen=True)


@pytest.mark.parametrize("content", ["", "{", "[]", '{"pid": "x", "acquired_at": 1}'])
def test_unreadable_lock_gives_explicit_refusal(root, stop_ckpt, content):
    open_s(root, stop_ckpt)
    (root / "s1" / "session.lock").write_text(content, "utf-8")
    with pytest.raises(HarnessError, match="illisible.*seul Anaël"):
        harness.update_state(root / "s1", frozen=True)


def test_unlock_removes_dead_lock_and_journals_it(root, stop_ckpt):
    open_s(root, stop_ckpt)
    pid = dead_pid()
    write_lock(root / "s1", pid)
    entry = harness.unlock_session("s1", root)
    assert not (root / "s1" / "session.lock").exists()
    last = harness.read_journal(root / "s1")[-1]
    assert last == entry
    assert last["event"] == "unlock" and last["pid"] == pid
    assert last["lock_acquired_at"] == "2026-11-05T10:00:00+00:00"
    datetime.fromisoformat(last["at"])


def test_unlock_removes_unreadable_lock(root, stop_ckpt):
    open_s(root, stop_ckpt)
    (root / "s1" / "session.lock").write_text("", "utf-8")
    entry = harness.unlock_session("s1", root)
    assert entry["pid"] is None and not (root / "s1" / "session.lock").exists()


def test_unlock_refuses_live_process_and_removes_nothing(root, stop_ckpt):
    open_s(root, stop_ckpt)
    write_lock(root / "s1", os.getpid())
    before = harness.read_journal(root / "s1")
    with pytest.raises(HarnessError, match=rf"{os.getpid()} est vivant"):
        harness.unlock_session("s1", root)
    assert (root / "s1" / "session.lock").exists()
    assert harness.read_journal(root / "s1") == before


def test_unlock_without_lock_is_a_clear_noop(root, stop_ckpt, capsys):
    open_s(root, stop_ckpt)
    assert harness.unlock_session("s1", root) is None
    import argparse

    parser = argparse.ArgumentParser()
    harness.add_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["session-unlock", "--session", "s1", "--sessions-root", str(root)])
    assert harness.run(args) == 0
    assert "aucun verrou à retirer" in capsys.readouterr().out


def test_unlock_cli_removes_dead_lock(root, stop_ckpt, capsys):
    import argparse

    open_s(root, stop_ckpt)
    pid = dead_pid()
    write_lock(root / "s1", pid)
    parser = argparse.ArgumentParser()
    harness.add_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["session-unlock", "--session", "s1", "--sessions-root", str(root)])
    assert harness.run(args) == 0
    assert f"PID {pid} mort" in capsys.readouterr().out
    write_lock(root / "s1", os.getpid())
    assert harness.run(args) == 2


@pytest.mark.parametrize("sid", ["", "..", "../x", "a/b", "a\\b", ".cache"])
def test_session_id_validated(root, stop_ckpt, sid):
    with pytest.raises(HarnessError, match="identifiant de séance invalide"):
        open_s(root, stop_ckpt, sid=sid)


# --- Revue passe 1 : décompte, verrou, lignée, empreintes ---


def test_base_checkpoint_from_a_session_refused(root, stop_ckpt, tmp_path):
    open_s(root, stop_ckpt)
    with pytest.raises(HarnessError, match="issu d'une séance"):
        open_s(root, root / "s1" / "work", sid="s2")
    # Copie d'une séance hors du dossier des séances : toujours reconnue (state.json au-dessus).
    import shutil

    elsewhere = tmp_path / "copie"
    shutil.copytree(root / "s1", elsewhere)
    with pytest.raises(HarnessError, match="issu d'une séance"):
        open_s(root, elsewhere / "work", sid="s3")
    assert not (root / "s2").exists() and not (root / "s3").exists()


def test_deleted_session_id_cannot_be_reopened(root, stop_ckpt, train_file):
    import shutil

    open_s(root, stop_ckpt, 10)
    harness.run_batch("s1", "gsm8k", "train", train_file, count=1, sessions_root=root)
    shutil.rmtree(root / "s1")
    with pytest.raises(HarnessError, match="déjà ouverte"):
        open_s(root, stop_ckpt, 50)
    registry = (root / "registry.jsonl").read_text("utf-8").splitlines()
    assert [json.loads(line)["session_id"] for line in registry] == ["s1"]


def test_used_reset_by_hand_refused(root, stop_ckpt, train_file, capsys):
    open_s(root, stop_ckpt, 10)
    d = root / "s1"
    for _ in range(2):
        harness.run_batch("s1", "gsm8k", "train", train_file, count=1, sessions_root=root)
    raw = json.loads((d / "state.json").read_text("utf-8"))
    raw["used"] = 0
    (d / "state.json").write_text(json.dumps(raw), "utf-8")
    with pytest.raises(HarnessError, match="compteur incohérent : state.json compte 0 lots"):
        harness.run_batch("s1", "gsm8k", "train", train_file, sessions_root=root)
    assert main(["session-status", "--session", "s1", "--sessions-root", str(root)]) == 2
    assert "compteur incohérent" in capsys.readouterr().err


def test_crash_during_generation_consumes_batch(root, stop_ckpt, train_file, monkeypatch):
    open_s(root, stop_ckpt, 10)
    d = root / "s1"
    real = harness._generate

    def crash(work, problems, device):
        raise RuntimeError("plantage simulé")

    monkeypatch.setattr(harness, "_generate", crash)
    with pytest.raises(RuntimeError, match="plantage simulé"):
        harness.run_batch("s1", "gsm8k", "train", train_file, count=1, sessions_root=root)
    assert harness.read_state(d).used == 1
    assert [e["event"] for e in harness.read_journal(d)] == ["open", "reserve"]
    assert not (d / "session.lock").exists()
    monkeypatch.setattr(harness, "_generate", real)
    entry = harness.run_batch("s1", "gsm8k", "train", train_file, count=1, sessions_root=root)
    assert (entry["batch"], entry["remaining"]) == (2, 8)
    events = [(e["event"], e.get("batch")) for e in harness.read_journal(d)]
    assert events == [("open", None), ("reserve", 1), ("reserve", 2), ("batch", 2)]


def test_state_write_refused_generates_nothing(root, stop_ckpt, train_file, monkeypatch):
    open_s(root, stop_ckpt, 10)
    d = root / "s1"
    monkeypatch.setattr(harness, "_REPLACE_DELAY", 0)
    calls = []

    def locked(src, dst):
        calls.append(dst)
        raise PermissionError(13, "Accès refusé", str(dst))

    monkeypatch.setattr(harness.os, "replace", locked)
    monkeypatch.setattr(harness, "_generate", lambda *a: pytest.fail("généré sans décompte"))
    with pytest.raises(HarnessError, match="écriture de .*state.json impossible"):
        harness.run_batch("s1", "gsm8k", "train", train_file, count=1, sessions_root=root)
    assert len(calls) == harness._REPLACE_RETRIES
    assert [e["event"] for e in harness.read_journal(d)] == ["open"]
    assert harness.read_state(d).used == 0
    assert not (d / "state.tmp").exists() and not (d / "session.lock").exists()


def test_state_write_retried_while_file_held(root, stop_ckpt, train_file, monkeypatch):
    open_s(root, stop_ckpt, 10)
    monkeypatch.setattr(harness, "_REPLACE_DELAY", 0)
    real_replace = os.replace
    failures = [PermissionError(13, "Accès refusé")] * 3

    def flaky(src, dst):
        if failures:
            raise failures.pop()
        return real_replace(src, dst)

    monkeypatch.setattr(harness.os, "replace", flaky)
    entry = harness.run_batch("s1", "gsm8k", "train", train_file, count=1, sessions_root=root)
    assert entry["batch"] == 1 and harness.read_state(root / "s1").used == 1


def test_freeze_during_batch_refused_never_undone(root, stop_ckpt, train_file, monkeypatch):
    open_s(root, stop_ckpt, 10)
    d = root / "s1"
    real = harness._generate
    seen = []

    def freeze_then_generate(work, problems, device):
        with pytest.raises(HarnessError, match="un lot tourne déjà") as exc:
            harness.update_state(d, frozen=True)
        seen.append(exc.value)
        return real(work, problems, device)

    monkeypatch.setattr(harness, "_generate", freeze_then_generate)
    harness.run_batch("s1", "gsm8k", "train", train_file, count=1, sessions_root=root)
    assert seen and harness.read_state(d).frozen is False
    monkeypatch.setattr(harness, "_generate", real)
    assert harness.update_state(d, frozen=True).frozen is True
    assert not (d / "session.lock").exists()
    with pytest.raises(HarnessError, match="gelée"):
        harness.run_batch("s1", "gsm8k", "train", train_file, count=1, sessions_root=root)
    assert [e["event"] for e in harness.read_journal(d)] == ["open", "reserve", "batch"]


def test_checkpoint_generation_config_cannot_change_decoding(tmp_path, train_file):
    """generation_config.json de work/ (échantillonnage, plafond 5) ignoré : glouton, 4096."""
    ckpt = build_checkpoint(tmp_path / "length", eos_id=1)
    root = tmp_path / "sessions"
    harness.open_session("G", 10, ckpt, sessions_root=root)
    work = root / "G" / "work"
    before = harness.work_file_hashes(work)
    (work / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "temperature": 5.0, "top_k": 0, "max_new_tokens": 5}),
        "utf-8",
    )
    entry = harness.run_batch("G", "gsm8k", "train", train_file, count=1, sessions_root=root)
    out = entry["outputs"][0]
    assert (out["new_tokens"], out["done_reason"]) == (4096, "length")
    assert set(out["text"].split()) == {"<eos>"}
    files = entry["files_sha256"]
    gen_sha = hashlib.sha256((work / "generation_config.json").read_bytes()).hexdigest()
    assert files["generation_config.json"] == gen_sha
    assert files["generation_config.json"] != before.get("generation_config.json")
    assert files["config.json"] == before["config.json"]


def fake_cuda(monkeypatch, seen=None, free=8 << 30):
    """CUDA simulé : pilote, nom de GPU et génération remplacés ; aucun GPU réel."""
    monkeypatch.setattr(harness, "_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA GeForce RTX 3050")
    monkeypatch.setattr(torch.version, "cuda", "13.2")
    monkeypatch.setattr(harness, "_cuda_free_bytes", lambda: free)

    def fake_generate(work, problems, device):
        if seen is not None:
            seen.append(device)
        settings = harness._generation_settings("raw", "torch.bfloat16")
        outputs = [
            {"id": p["id"], "text": "", "new_tokens": 0, "done_reason": "stop"} for p in problems
        ]
        return settings, outputs

    monkeypatch.setattr(harness, "_generate", fake_generate)


def test_cuda_batch_journals_device_gpu_and_versions(root, stop_ckpt, train_file, monkeypatch):
    seen = []
    fake_cuda(monkeypatch, seen)
    open_s(root, stop_ckpt, 10)
    entry = harness.run_batch(
        "s1", "gsm8k", "train", train_file, count=1, sessions_root=root, device="auto"
    )
    assert seen == ["cuda"]
    assert entry["device"] == {
        "device": "cuda",
        "gpu_name": "NVIDIA GeForce RTX 3050",
        "torch_version": torch.__version__,
        "cuda_version": "13.2",
    }
    journal = harness.read_journal(root / "s1")
    assert [(e["event"], e.get("device")) for e in journal[1:]] == [
        ("reserve", "cuda"),
        ("batch", entry["device"]),
    ]
    assert harness.read_journal(root / "s1")[-1]["settings"]["max_new_tokens"] == 4096


def test_cpu_batch_journals_no_gpu(root, stop_ckpt, train_file):
    open_s(root, stop_ckpt, 10)
    entry = harness.run_batch("s1", "gsm8k", "train", train_file, count=1, sessions_root=root)
    assert entry["device"]["device"] == "cpu"
    assert entry["device"]["gpu_name"] is None
    assert entry["device"]["torch_version"] == torch.__version__


def test_auto_falls_back_to_cpu_without_cuda():
    assert harness.resolve_device("auto") == "cpu"
    assert harness.resolve_device("cpu") == "cpu"


def test_cuda_requested_without_gpu_refused_without_consuming_batch(root, stop_ckpt, train_file):
    open_s(root, stop_ckpt, 10)
    with pytest.raises(HarnessError, match="--device cuda refusé : aucun GPU CUDA disponible"):
        harness.run_batch(
            "s1", "gsm8k", "train", train_file, count=1, sessions_root=root, device="cuda"
        )
    assert harness.read_state(root / "s1").used == 0
    assert [e["event"] for e in harness.read_journal(root / "s1")] == ["open"]


def test_unknown_device_refused():
    with pytest.raises(HarnessError, match="appareil 'tpu' refusé"):
        harness.resolve_device("tpu")


def test_device_change_mid_session_refused_before_reservation(
    root, stop_ckpt, train_file, monkeypatch
):
    open_s(root, stop_ckpt, 10)
    harness.run_batch("s1", "gsm8k", "train", train_file, count=1, sessions_root=root)
    fake_cuda(monkeypatch)
    with pytest.raises(
        HarnessError, match="changement d'appareil refusé : la séance tourne sur cpu, pas sur cuda"
    ):
        harness.run_batch(
            "s1", "gsm8k", "train", train_file, count=1, sessions_root=root, device="auto"
        )
    assert harness.read_state(root / "s1").used == 1
    harness.run_batch("s1", "gsm8k", "train", train_file, count=1, sessions_root=root, device="cpu")
    assert harness.read_state(root / "s1").used == 2


def test_crashed_first_batch_still_fixes_device(root, stop_ckpt, train_file, monkeypatch):
    fake_cuda(monkeypatch)
    monkeypatch.setattr(harness, "_generate", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    open_s(root, stop_ckpt, 10)
    with pytest.raises(RuntimeError):
        harness.run_batch(
            "s1", "gsm8k", "train", train_file, count=1, sessions_root=root, device="auto"
        )
    with pytest.raises(HarnessError, match="changement d'appareil refusé"):
        harness.run_batch(
            "s1", "gsm8k", "train", train_file, count=1, sessions_root=root, device="cpu"
        )


def test_cli_device_option_default_auto_and_choices(root, stop_ckpt, train_file, capsys):
    open_s(root, stop_ckpt, 10)
    args = ["run-batch", "--session", "s1", "--sessions-root", str(root), "--dataset", "gsm8k"]
    args += ["--split", "train", "--data", str(train_file), "--count", "1"]
    assert main(args) == 0
    assert "(appareil cpu)" in capsys.readouterr().out
    assert main([*args, "--device", "cuda"]) == 2
    assert "--device cuda refusé" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main([*args, "--device", "tpu"])


def test_generate_releases_gpu_memory(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(harness, "_generate_on", lambda w, p, d: ({}, []))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    harness._generate(tmp_path, [], "cuda")
    harness._generate(tmp_path, [], "cpu")
    assert calls == ["empty"]


# Revue passe 1 : séances antérieures à --device, mémoire GPU contrôlée avant la réservation.


def test_reserve_without_device_counts_as_cpu(root, stop_ckpt, train_file, monkeypatch):
    open_s(root, stop_ckpt, 10)
    import dataclasses

    state = harness.read_state(root / "s1")
    harness._write_state(root / "s1", dataclasses.replace(state, used=1))
    harness._append_journal(root / "s1", {"event": "reserve", "batch": 1, "budget": 10})
    fake_cuda(monkeypatch)
    with pytest.raises(HarnessError, match="la séance tourne sur cpu, pas sur cuda"):
        harness.run_batch(
            "s1", "gsm8k", "train", train_file, count=1, sessions_root=root, device="auto"
        )


def write_fake_weights(work: Path, dtype: str, numel: int) -> None:
    """En-tête safetensors d'un seul tenseur de `numel` éléments (octets non écrits)."""
    import struct

    stored = {"F32": 4, "BF16": 2}[dtype]
    header = {"w": {"dtype": dtype, "shape": [numel], "data_offsets": [0, numel * stored]}}
    raw = json.dumps(header).encode()
    with (work / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        f.truncate(8 + len(raw) + numel * stored)


@pytest.mark.parametrize(
    ("config_dtype", "stored", "expected"),
    [("bfloat16", "BF16", 2), ("float32", "F32", 4), ("bfloat16", "F32", 2), (None, "F32", 4)],
)
def test_weights_bytes_follow_loaded_dtype(tmp_path, config_dtype, stored, expected):
    work = tmp_path / "work"
    work.mkdir()
    config = {} if config_dtype is None else {"dtype": config_dtype}
    (work / "config.json").write_text(json.dumps(config), "utf-8")
    write_fake_weights(work, stored, 1000)
    assert harness.weights_bytes(work) == (1000 * expected, config_dtype)


def test_fp32_checkpoint_too_big_for_gpu_refused_without_consuming_batch(
    root, stop_ckpt, train_file, monkeypatch
):
    seen = []
    fake_cuda(monkeypatch, seen, free=8 << 30)
    open_s(root, stop_ckpt, 10)
    work = root / "s1" / "work"
    (work / "config.json").write_text(json.dumps({"dtype": "float32"}), "utf-8")
    for f in work.glob("*.safetensors"):
        f.unlink()
    write_fake_weights(work, "F32", 2_000_000_000)  # 2 Md de paramètres fp32 : 8 Go
    with pytest.raises(HarnessError, match="mémoire GPU insuffisante") as exc:
        harness.run_batch(
            "s1", "gsm8k", "train", train_file, count=1, sessions_root=root, device="auto"
        )
    assert "pas de conversion en bf16" in str(exc.value)
    assert seen == []
    assert harness.read_state(root / "s1").used == 0
    assert [e["event"] for e in harness.read_journal(root / "s1")] == ["open"]


def test_bf16_checkpoint_fits_gpu(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    (work / "config.json").write_text(json.dumps({"dtype": "bfloat16"}), "utf-8")
    write_fake_weights(work, "BF16", 2_000_000_000)  # 4 Go
    monkeypatch.setattr(harness, "_cuda_free_bytes", lambda: 7 << 30)
    harness.check_cuda_memory(work)
    monkeypatch.setattr(harness, "_cuda_free_bytes", lambda: 4 << 30)
    with pytest.raises(HarnessError, match="mémoire GPU insuffisante") as exc:
        harness.check_cuda_memory(work)
    assert "conversion" not in str(exc.value)


def test_cpu_batch_never_queries_gpu_memory(root, stop_ckpt, train_file):
    open_s(root, stop_ckpt, 10)
    harness.run_batch("s1", "gsm8k", "train", train_file, count=1, sessions_root=root)
