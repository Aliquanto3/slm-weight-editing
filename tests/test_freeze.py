"""Gel de séance, diff vérifiable et rejeu bit à bit (tenseurs synthétiques bf16, f16, f32)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from conftest import TARGET, build, write_safetensors

import slm_weight_editing
from slm_weight_editing import freeze, harness
from slm_weight_editing.compare_edits import CompareError, compare
from slm_weight_editing.freeze import DiffError, FreezeError
from slm_weight_editing.harness import HarnessError
from slm_weight_editing.report_gguf import report_gguf
from slm_weight_editing.safetensors_file import open_checkpoint

REV = "rev-synthetique"


def run_batch(*args, **kwargs):
    """Lot en mode raisonnement `off` sauf précision (le mode est obligatoire)."""
    kwargs.setdefault("thinking", "off")
    return harness.run_batch(*args, **kwargs)


def sha(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def base_tensors() -> dict[str, tuple[str, np.ndarray]]:
    rng = np.random.default_rng(1)
    return {
        "bf.safetensors": ("BF16", {"w.bf": rng.integers(0, 2**16, (4, 8)).astype("<u2")}),
        "f16.safetensors": ("F16", {"w.f16": rng.normal(size=(5, 6)).astype("<f2")}),
        "f32.safetensors": ("F32", {"w.f32": rng.normal(size=(40,)).astype("<f4")}),
    }


def write_checkpoint(path: Path, spec) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for fname, (dtype, tensors) in spec.items():
        write_safetensors(path / fname, tensors, dtype=dtype)
    (path / "config.json").write_text('{"a": 1}', "utf-8")
    return path


@pytest.fixture
def sess(tmp_path):
    """Séance ouverte sur une base à trois dtypes, avec une édition déjà faite dans work/."""
    spec = base_tensors()
    base = write_checkpoint(tmp_path / "base", spec)
    root = tmp_path / "sessions"
    harness.open_session("s1", 10, base, base_revision=REV, sessions_root=root)
    edited = {k: (d, {n: a.copy() for n, a in t.items()}) for k, (d, t) in spec.items()}
    edited["bf.safetensors"][1]["w.bf"][1, 2] = 0x7FC1  # motif NaN : pas d'arrondi flottant
    edited["bf.safetensors"][1]["w.bf"][3, 7] ^= 1
    edited["f16.safetensors"][1]["w.f16"][0, 0] += np.float16(1)
    edited["f32.safetensors"][1]["w.f32"][39] = np.float32(-0.0)
    work = root / "s1" / "work"
    for fname, (dtype, tensors) in edited.items():
        write_safetensors(work / fname, tensors, dtype=dtype)
    return {"base": base, "root": root, "dir": root / "s1", "edited": edited, "spec": spec}


def freeze_it(s, **kw):
    return freeze.freeze_session("s1", s["base"], sessions_root=s["root"], **kw)


def test_freeze_writes_exact_diff_and_freezes(sess):
    diff = freeze_it(sess)
    assert diff["format"] == "slm-weight-editing/edit-diff-v2"
    assert (diff["session_id"], diff["base_revision"], diff["budget"], diff["batches_used"]) == (
        "s1",
        REV,
        10,
        0,
    )
    by_name = {t["name"]: t for t in diff["tensors"]}
    assert sorted(by_name) == ["w.bf", "w.f16", "w.f32"]
    bf = by_name["w.bf"]
    old_bf, new_bf = sess["spec"]["bf.safetensors"][1]["w.bf"], None
    new_bf = sess["edited"]["bf.safetensors"][1]["w.bf"]
    assert (bf["shape"], bf["dtype"], bf["numel"], bf["changed"]) == ([4, 8], "BF16", 32, 2)
    assert (bf["sha256_before"], bf["sha256_after"]) == (sha(old_bf), sha(new_bf))
    assert by_name["w.f16"]["changed"] == 1 and by_name["w.f16"]["dtype"] == "F16"
    # -0.0 diffère de la valeur d'origine par le bit de signe seulement s'il était > 0 : bit à bit
    assert by_name["w.f32"]["changed"] == 1
    assert diff["files_changed"] == []
    d = sess["dir"]
    state = harness.read_state(d)
    assert state.frozen is True
    out = d / "freeze"
    assert json.loads((out / "diff.json").read_text("utf-8")) == diff
    values = (out / "diff.bin").read_bytes()
    assert diff["values_file"] == {
        "name": "diff.bin",
        "sha256": hashlib.sha256(values).hexdigest(),
        "bytes": len(values),
    }
    # indices u4 puis valeurs brutes u2 : (2 * 4) + (2 * 2) octets pour w.bf
    assert bf["index_dtype"] == "u4" and bf["indices_offset"] == 0 and bf["values_offset"] == 8
    assert np.frombuffer(values, "<u4", 2, 0).tolist() == [10, 31]
    assert np.frombuffer(values, "<u2", 2, 8).tolist() == [0x7FC1, int(new_bf[3, 7])]
    events = harness.read_journal(d)
    assert [e["event"] for e in events] == ["open", "freeze"]
    copy_sha = hashlib.sha256((out / "journal.jsonl").read_bytes()).hexdigest()
    assert diff["journal"] == {"file": "journal.jsonl", "sha256": copy_sha}
    assert events[1]["journal_sha256"] == copy_sha
    assert events[1]["diff_sha256"] == hashlib.sha256((out / "diff.json").read_bytes()).hexdigest()
    assert events[1]["tensors_changed"] == 3


def test_freeze_is_final_and_harness_refuses_batches(sess, tmp_path):
    freeze_it(sess)
    data = tmp_path / "data" / "train.jsonl"
    data.parent.mkdir()
    data.write_text(json.dumps({"id": "1", "question": "q", "split": "train"}) + "\n", "utf-8")
    with pytest.raises(HarnessError, match="séance gelée"):
        run_batch("s1", "gsm8k", "train", data, sessions_root=sess["root"])
    assert harness.used_batches(sess["dir"]) == 0
    with pytest.raises(HarnessError, match="ne se dégèle pas"):
        harness.update_state(sess["dir"], frozen=False)
    with pytest.raises(FreezeError, match="déjà gelée"):
        freeze_it(sess)
    assert harness.read_state(sess["dir"]).frozen is True


def test_wrong_original_refuses_and_writes_nothing(sess, tmp_path):
    other = write_checkpoint(tmp_path / "autre", base_tensors())
    arr = np.zeros((4, 8), "<u2")
    write_safetensors(other / "bf.safetensors", {"w.bf": arr})
    before = (sess["dir"] / "journal.jsonl").read_bytes()
    with pytest.raises(FreezeError, match=r"ne correspond pas.*\['w.bf'\]"):
        freeze.freeze_session("s1", other, sessions_root=sess["root"])
    assert harness.read_state(sess["dir"]).frozen is False
    assert not (sess["dir"] / "freeze").exists()
    assert (sess["dir"] / "journal.jsonl").read_bytes() == before


def test_original_with_other_config_is_refused(sess):
    (sess["base"] / "config.json").write_text('{"a": 2}', "utf-8")
    with pytest.raises(FreezeError, match="fichiers de l'original"):
        freeze_it(sess)
    assert harness.read_state(sess["dir"]).frozen is False


def test_ceiling_refuses_a_near_complete_tensor_without_freezing(sess):
    work = sess["dir"] / "work"
    arr = sess["edited"]["f32.safetensors"][1]["w.f32"].copy()
    arr[:5] += np.float32(1)  # 5 + 1 déjà changé = 6/40 = 15 % > 10 %
    write_safetensors(work / "f32.safetensors", {"w.f32": arr}, dtype="F32")
    with pytest.raises(
        FreezeError, match=r"w\.f32 : 6/40 .*[(]15.0%[)], plafond 10%.*tenseur complet"
    ):
        freeze_it(sess)
    assert harness.read_state(sess["dir"]).frozen is False
    assert not (sess["dir"] / "freeze").exists()
    arr[:4] = sess["spec"]["f32.safetensors"][1]["w.f32"][:4]  # 2 changés, sous le plafond
    write_safetensors(work / "f32.safetensors", {"w.f32": arr}, dtype="F32")
    assert freeze_it(sess)["tensors"]


def test_shape_dtype_or_names_changes_are_refused(sess):
    work = sess["dir"] / "work"
    write_safetensors(work / "f32.safetensors", {"w.f32": np.zeros(41, "<f4")}, dtype="F32")
    with pytest.raises(FreezeError, match=r"w\.f32 : forme \[41\]"):
        freeze_it(sess)
    write_safetensors(work / "f32.safetensors", {"w.autre": np.zeros(40, "<f4")}, dtype="F32")
    with pytest.raises(FreezeError, match=r"en trop : \['w.autre'\]"):
        freeze_it(sess)


def test_changed_non_tensor_files_are_reported(sess):
    work = sess["dir"] / "work"
    (work / "config.json").write_text('{"a": 3}', "utf-8")
    (work / "extra.txt").write_text("x", "utf-8")
    diff = freeze_it(sess)
    cfg = hashlib.sha256(b'{"a": 1}').hexdigest()
    assert diff["files_changed"] == [
        {
            "path": "config.json",
            "sha256_before": cfg,
            "sha256_after": hashlib.sha256(b'{"a": 3}').hexdigest(),
        },
        {
            "path": "extra.txt",
            "sha256_before": None,
            "sha256_after": hashlib.sha256(b"x").hexdigest(),
        },
    ]


def test_tensor_changed_during_freeze_is_refused(sess, monkeypatch):
    real = freeze._freeze_state

    def tampering(directory):
        real(directory)
        arr = np.zeros(40, "<f4")
        write_safetensors(directory / "work" / "f32.safetensors", {"w.f32": arr}, dtype="F32")

    monkeypatch.setattr(freeze, "_freeze_state", tampering)
    with pytest.raises(FreezeError, match="work/ a changé pendant le gel ; séance déjà gelée"):
        freeze_it(sess)
    assert not (sess["dir"] / "freeze").exists()


def test_running_batch_lock_gives_a_bounded_clear_error(sess, monkeypatch):
    monkeypatch.setattr(freeze, "_FREEZE_RETRIES", 3)
    monkeypatch.setattr(freeze, "_FREEZE_DELAY", 0)
    (sess["dir"] / harness.LOCK_FILE).write_text("", "utf-8")
    with pytest.raises(
        FreezeError, match="gel non posé après 3 essais : un lot tourne peut-être.*session-unlock"
    ):
        freeze_it(sess)
    assert harness.read_state(sess["dir"]).frozen is False


def test_lock_released_during_retries_lets_the_freeze_through(sess, monkeypatch):
    lock = sess["dir"] / harness.LOCK_FILE
    lock.write_text("", "utf-8")
    monkeypatch.setattr(freeze, "_FREEZE_DELAY", 0)
    sleeps = []
    monkeypatch.setattr(freeze.time, "sleep", lambda s: (sleeps.append(s), lock.unlink()))
    freeze_it(sess)
    assert len(sleeps) == 1 and harness.read_state(sess["dir"]).frozen is True


def test_interrupted_freeze_can_be_resumed(sess):
    harness.update_state(sess["dir"], frozen=True)  # gel posé, aucun diff ni événement freeze
    assert freeze_it(sess)["tensors"]


def test_replay_gives_exact_published_sha(sess, tmp_path):
    diff = freeze_it(sess)
    out = tmp_path / "models" / "rejoue"
    meta = freeze.apply_diff(sess["dir"] / "freeze" / "diff.json", sess["base"], out)
    assert meta["tensors"] == diff["tensors"]
    replayed = open_checkpoint(out)
    work = open_checkpoint(sess["dir"] / "work")
    assert {n: t.read_bytes() for n, t in replayed.items()} == {
        n: t.read_bytes() for n, t in work.items()
    }
    for t in diff["tensors"]:
        assert hashlib.sha256(replayed[t["name"]].read_bytes()).hexdigest() == t["sha256_after"]
    assert (out / "config.json").read_bytes() == b'{"a": 1}'
    assert sorted(p.name for p in out.glob("*.safetensors")) == sorted(
        p.name for p in sess["base"].glob("*.safetensors")
    )


def test_replay_refuses_a_tampered_values_file(sess, tmp_path):
    freeze_it(sess)
    bin_path = sess["dir"] / "freeze" / "diff.bin"
    raw = bytearray(bin_path.read_bytes())
    raw[-1] ^= 0xFF
    bin_path.write_bytes(raw)
    out = tmp_path / "out"
    with pytest.raises(DiffError, match="diff.bin : sha256 différent"):
        freeze.apply_diff(sess["dir"] / "freeze" / "diff.json", sess["base"], out)
    assert not out.exists()


def test_replay_refuses_a_forged_published_sha(sess, tmp_path):
    freeze_it(sess)
    path = sess["dir"] / "freeze" / "diff.json"
    meta = json.loads(path.read_text("utf-8"))
    meta["tensors"][0]["sha256_after"] = "0" * 64
    path.write_text(json.dumps(meta), "utf-8")
    out = tmp_path / "out"
    with pytest.raises(DiffError, match=r"w\.bf : le rejeu ne redonne pas le sha256 après"):
        freeze.apply_diff(path, sess["base"], out)
    assert not out.exists()


def test_replay_refuses_another_original(sess, tmp_path):
    freeze_it(sess)
    other = write_checkpoint(tmp_path / "autre", base_tensors())
    write_safetensors(other / "f32.safetensors", {"w.f32": np.ones(40, "<f4")}, dtype="F32")
    with pytest.raises(DiffError, match=r"w\.f32 : l'original n'a pas le sha256 avant"):
        freeze.apply_diff(sess["dir"] / "freeze" / "diff.json", other, tmp_path / "out")


def test_replay_refuses_non_empty_output_and_unknown_format(sess, tmp_path):
    freeze_it(sess)
    diff_path = sess["dir"] / "freeze" / "diff.json"
    full = tmp_path / "plein"
    full.mkdir()
    (full / "x").write_text("x", "utf-8")
    with pytest.raises(DiffError, match="sortie non vide"):
        freeze.apply_diff(diff_path, sess["base"], full)
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "autre"}', "utf-8")
    with pytest.raises(DiffError, match="format 'autre'"):
        freeze.load_diff(bad)
    with pytest.raises(DiffError, match="diff illisible"):
        freeze.load_diff(tmp_path / "absent.json")


def test_cli_freeze_then_apply_diff(sess, tmp_path, capsys):
    root, out = str(sess["root"]), tmp_path / "rejoue"
    (sess["dir"] / "work" / "notes.txt").write_text("n", "utf-8")
    code = slm_weight_editing.main(
        ["session-freeze", "--session", "s1", "--original", str(sess["base"])]
        + ["--sessions-root", root]
    )
    captured = capsys.readouterr().out
    assert code == 0 and "3 tenseurs modifiés, 4 éléments" in captured
    assert "Attention : notes.txt diffère de la base" in captured
    code = slm_weight_editing.main(
        ["apply-diff", "--diff", str(sess["dir"] / "freeze" / "diff.json")]
        + ["--original", str(sess["base"]), "--out", str(out)]
    )
    captured = capsys.readouterr().out
    assert code == 0 and "3 tenseurs vérifiés" in captured and "notes.txt" in captured
    # séance déjà gelée : code 2 du harnais
    code = slm_weight_editing.main(
        ["session-freeze", "--session", "s1", "--original", str(sess["base"])]
        + ["--sessions-root", root]
    )
    assert code == 2 and "déjà gelée" in capsys.readouterr().err
    code = slm_weight_editing.main(
        ["apply-diff", "--diff", str(tmp_path / "nul.json")]
        + ["--original", str(sess["base"]), "--out", str(tmp_path / "o2")]
    )
    assert code == 1 and "diff illisible" in capsys.readouterr().err


# --- report-gguf sur le checkpoint rejoué ---------------------------------------------------


def gguf_session(tmp_path):
    s = build(tmp_path)
    root = tmp_path / "sessions"
    harness.open_session("g1", 25, s.base_checkpoint, base_revision=REV, sessions_root=root)
    work = root / "g1" / "work"
    write_safetensors(work / "model.safetensors", s.edited)
    return s, root


def test_report_gguf_accepts_the_replayed_checkpoint_via_edit_diff(tmp_path, capsys):
    s, root = gguf_session(tmp_path)
    diff = freeze.freeze_session("g1", s.base_checkpoint, sessions_root=root)
    assert [t["name"] for t in diff["tensors"]] == [TARGET]
    replayed = tmp_path / "models" / "rejoue"
    freeze.apply_diff(root / "g1" / "freeze" / "diff.json", s.base_checkpoint, replayed)
    before = {p.name: p.read_bytes() for p in replayed.iterdir()}
    out = tmp_path / "gguf"
    args = ["report-gguf", "--checkpoint", str(replayed), "--base-checkpoint"]
    args += [str(s.base_checkpoint), "--ollama-models", str(s.store), "--out", str(out)]
    args += ["--edit-diff", str(root / "g1" / "freeze" / "diff.json")]
    assert slm_weight_editing.main(args) == 1  # pas de tag : refus
    assert "--edit-diff demande --tag" in capsys.readouterr().err
    assert slm_weight_editing.main(args + ["--tag", "perso:edit"]) == 0
    assert {p.name: p.read_bytes() for p in replayed.iterdir()} == before  # inchangé
    rep = json.loads((out / "diff.json").read_text("utf-8"))
    assert [t["hf_name"] for t in rep["tensors"]] == [TARGET]
    assert rep["tensors"][0]["hf_bf16_sha256_after"] == diff["tensors"][0]["sha256_after"]
    assert rep["base"] == {"repo": None, "revision": REV}
    assert rep["edit"]["repo"] is None and rep["edit"]["session_id"] == "g1"
    diff_sha = hashlib.sha256((root / "g1" / "freeze" / "diff.json").read_bytes()).hexdigest()
    assert rep["edit"]["diff_sha256"] == diff_sha
    assert rep["edit"]["journal_sha256"] == diff["journal"]["sha256"]
    assert "MIT" not in json.dumps(rep) and "RikiyaT" not in json.dumps(rep)
    assert (out / "perso-edit.gguf").exists()


def test_report_gguf_edit_diff_refuses_a_checkpoint_that_is_not_the_edit(tmp_path, capsys):
    s, root = gguf_session(tmp_path)
    freeze.freeze_session("g1", s.base_checkpoint, sessions_root=root)
    out = tmp_path / "gguf"
    code = slm_weight_editing.main(
        ["report-gguf", "--checkpoint", str(s.base_checkpoint)]
        + ["--base-checkpoint", str(s.base_checkpoint), "--ollama-models", str(s.store)]
        + ["--edit-diff", str(root / "g1" / "freeze" / "diff.json"), "--tag", "t:1"]
        + ["--out", str(out)]
    )
    assert code == 3 and "sha256" in capsys.readouterr().err and not out.exists()


def test_report_gguf_edit_diff_unreadable_diff(tmp_path, capsys):
    s = build(tmp_path)
    code = slm_weight_editing.main(
        ["report-gguf", "--checkpoint", str(s.checkpoint), "--tag", "t:1"]
        + ["--base-checkpoint", str(s.base_checkpoint), "--edit-diff", str(tmp_path / "x.json")]
    )
    assert code == 1 and "diff invalide (diff illisible" in capsys.readouterr().err


def test_default_report_still_describes_the_mit_edit(synth, tmp_path, monkeypatch):
    from slm_weight_editing import report_gguf as module

    monkeypatch.setattr(module, "WSP_EDITED_SHA256", synth.expected_sha)
    result = report_gguf(synth.checkpoint, tmp_path / "out", store=synth.store)
    rep = json.loads(result.diff.read_text("utf-8"))
    assert rep["base"] == {"repo": module.WSP_BASE_REPO, "revision": module.WSP_BASE_REVISION}
    assert rep["edit"] == {"repo": module.WSP_EDIT_REPO}


def test_custom_sha_without_provenance_does_not_claim_the_mit_edit(synth, tmp_path):
    result = report_gguf(
        synth.checkpoint, tmp_path / "out", store=synth.store, expected_sha256=synth.expected_sha
    )
    rep = json.loads(result.diff.read_text("utf-8"))
    assert rep["base"] == {"repo": None, "revision": None} and rep["edit"] == {"repo": None}


# --- compare-edits avec un diff -----------------------------------------------------------------


def test_compare_with_diff_equals_compare_with_checkpoint(sess, tmp_path):
    freeze_it(sess)
    diff_path = sess["dir"] / "freeze" / "diff.json"
    replayed = tmp_path / "rejoue"
    freeze.apply_diff(diff_path, sess["base"], replayed)
    by_ckpt, _ = compare(sess["base"], replayed, sess["dir"] / "work")
    by_diff, md = compare(sess["base"], None, sess["dir"] / "work", diff_a=diff_path)
    assert json.dumps(by_diff["weights"]) == json.dumps(by_ckpt["weights"])  # NaN : pas de ==
    assert by_diff["weights"]["a"]["tensors"] == {"w.bf": 2, "w.f16": 1, "w.f32": 1}
    both_diff, _ = compare(sess["base"], None, None, diff_a=diff_path, diff_b=diff_path)
    assert both_diff["weights"]["a"] == both_diff["weights"]["b"]
    assert both_diff["weights"]["only_a"] == [] and md


def test_compare_needs_exactly_one_source_per_side(sess):
    freeze_it(sess)
    diff_path = sess["dir"] / "freeze" / "diff.json"
    work = sess["dir"] / "work"
    with pytest.raises(CompareError, match="édition A : un checkpoint édité ou un diff"):
        compare(sess["base"], work, work, diff_a=diff_path)
    with pytest.raises(CompareError, match="édition B : un checkpoint édité ou un diff"):
        compare(sess["base"], work, None)


def test_compare_with_a_diff_for_another_original_fails_cleanly(sess, tmp_path):
    freeze_it(sess)
    other = write_checkpoint(tmp_path / "autre", base_tensors())
    write_safetensors(other / "f32.safetensors", {"w.f32": np.ones(40, "<f4")}, dtype="F32")
    work = sess["dir"] / "work"
    with pytest.raises(CompareError, match="édition A : w.f32 : l'original n'a pas"):
        compare(other, None, work, diff_a=sess["dir"] / "freeze" / "diff.json")


def test_compare_cli_accepts_diffs(sess, tmp_path):
    freeze_it(sess)
    diff_path = sess["dir"] / "freeze" / "diff.json"
    out = tmp_path / "r" / "rapport.md"
    code = slm_weight_editing.main(
        ["compare-edits", "--original", str(sess["base"]), "--diff-a", str(diff_path)]
        + ["--diff-b", str(diff_path), "--out", str(out)]
    )
    assert code == 0
    weights = json.loads(out.with_suffix(".json").read_text("utf-8"))["weights"]
    assert weights["a"]["modified_weights"] == 4
    assert weights["total_weights"] == 32 + 30 + 40
    shutil.rmtree(out.parent)
    code = slm_weight_editing.main(
        ["compare-edits", "--original", str(sess["base"]), "--diff-a", str(diff_path)]
        + ["--out", str(out)]
    )
    assert code == 1 and not out.exists()


# --- Revue passe 1 ----------------------------------------------------------------------------


def test_published_journal_has_no_local_path(sess, tmp_path):
    data = tmp_path / "donnees" / "train.jsonl"
    harness._append_journal(sess["dir"], {"event": "note", "data_path": str(data)})
    diff = freeze_it(sess)
    raw = (sess["dir"] / harness.JOURNAL_FILE).read_text("utf-8")
    assert json.loads(raw.splitlines()[0])["base_checkpoint"] == str(sess["base"].resolve())
    published = (sess["dir"] / "freeze" / harness.JOURNAL_FILE).read_bytes()
    assert diff["journal"]["sha256"] == hashlib.sha256(published).hexdigest()
    home = Path.home()
    for f in (sess["dir"] / "freeze").iterdir():
        text = f.read_bytes().decode("utf-8", "replace").replace("\\\\", "\\")
        for local in (tmp_path, tmp_path.resolve(), home, home.resolve()):
            assert str(local) not in text and local.as_posix() not in text, f.name
    lines = [json.loads(line) for line in published.decode("utf-8").splitlines()]
    assert lines[0]["base_checkpoint"] == ".../base"
    assert lines[1]["data_path"] == ".../train.jsonl"
    assert lines[0]["base_tensors_sha256"]  # contenu toujours identifié


def test_public_path_keeps_paths_under_the_working_directory_relative():
    assert freeze.public_path(str(Path.cwd() / "models" / "hf" / "q")) == "models/hf/q"
    assert freeze.public_path("models/hf/q") == "models/hf/q"


def test_public_path_hides_user_name_when_cwd_is_above_home(monkeypatch):
    home = Path.home().resolve()
    monkeypatch.chdir(home.anchor)
    assert freeze.public_path(str(home / "models" / "q")) == ".../q"


def test_whole_freeze_runs_under_the_session_lock(sess, monkeypatch):
    real = freeze.compute_tensor_diff
    seen = []

    def spy(base, work, *args):
        seen.append((sess["dir"] / harness.LOCK_FILE).exists())
        with pytest.raises(HarnessError, match="un lot tourne déjà"):
            harness.update_state(sess["dir"], frozen=True)  # second gel concurrent
        return real(base, work, *args)

    monkeypatch.setattr(freeze, "compute_tensor_diff", spy)
    freeze_it(sess)
    assert seen == [True] and not (sess["dir"] / harness.LOCK_FILE).exists()
    events = [e["event"] for e in harness.read_journal(sess["dir"])]
    assert events.count("freeze") == 1


def test_failed_write_after_freeze_says_how_to_resume(sess, monkeypatch):
    def broken(path, data):
        raise OSError("disque plein")

    monkeypatch.setattr(freeze, "_write_atomic", broken)
    with pytest.raises(FreezeError, match="séance déjà gelée, relancer session-freeze"):
        freeze_it(sess)
    assert harness.read_state(sess["dir"]).frozen is True
    monkeypatch.undo()
    assert freeze_it(sess)["tensors"]  # reprise


def _rewrite(diff_path: Path, mutate) -> None:
    meta = json.loads(diff_path.read_text("utf-8"))
    values = bytearray((diff_path.parent / freeze.VALUES_FILE).read_bytes())
    mutate(meta, values)
    (diff_path.parent / freeze.VALUES_FILE).write_bytes(bytes(values))
    meta["values_file"] |= {"sha256": hashlib.sha256(values).hexdigest(), "bytes": len(values)}
    diff_path.write_text(json.dumps(meta), "utf-8")


def _set(key, value):
    def mutate(meta, values):
        meta["tensors"][0][key] = value

    return mutate


def _swap_indices(meta, values):
    t = meta["tensors"][0]  # w.bf, deux indices u4 : on les inverse
    o = t["indices_offset"]
    values[o : o + 8] = values[o + 4 : o + 8] + values[o : o + 4]


def _out_of_range_index(meta, values):
    t = meta["tensors"][0]
    o = t["indices_offset"] + 4
    values[o : o + 4] = (t["numel"]).to_bytes(4, "little")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_set("numel", 10**6), "numel ou changed incohérents"),
        (_set("changed", 10**6), "numel ou changed incohérents"),
        (_set("index_dtype", "f8"), "index_dtype ou dtype non géré"),
        (_set("index_dtype", "i4"), "index_dtype ou dtype non géré"),
        (_set("indices_offset", 10**6), "indices_offset hors de diff.bin"),
        (_set("values_offset", -1), "values_offset hors de diff.bin"),
        (_set("shape", "x"), "forme invalide"),
        (_swap_indices, "indices non triés ou en double"),
        (_out_of_range_index, "indice hors du tenseur"),
    ],
)
def test_malformed_diff_is_a_clean_diff_error(sess, tmp_path, capsys, mutate, message):
    freeze_it(sess)
    diff_path = sess["dir"] / "freeze" / "diff.json"
    _rewrite(diff_path, mutate)
    with pytest.raises(DiffError, match=message):
        freeze.apply_diff(diff_path, sess["base"], tmp_path / "out")
    assert not (tmp_path / "out").exists()
    code = slm_weight_editing.main(
        ["apply-diff", "--diff", str(diff_path), "--original", str(sess["base"])]
        + ["--out", str(tmp_path / "out")]
    )
    assert code == 1 and message in capsys.readouterr().err
    with pytest.raises(CompareError, match=message):
        compare(sess["base"], None, sess["dir"] / "work", diff_a=diff_path)


def test_report_gguf_refuses_a_diff_with_changed_files(tmp_path, capsys):
    s, root = gguf_session(tmp_path)
    (root / "g1" / "work" / "tokenizer_config.json").write_text("{}", "utf-8")
    freeze.freeze_session("g1", s.base_checkpoint, sessions_root=root)
    replayed = tmp_path / "models" / "rejoue"
    freeze.apply_diff(root / "g1" / "freeze" / "diff.json", s.base_checkpoint, replayed)
    out = tmp_path / "gguf"
    code = slm_weight_editing.main(
        ["report-gguf", "--checkpoint", str(replayed), "--tag", "t:1"]
        + ["--base-checkpoint", str(s.base_checkpoint), "--ollama-models", str(s.store)]
        + ["--edit-diff", str(root / "g1" / "freeze" / "diff.json"), "--out", str(out)]
    )
    err = capsys.readouterr().err
    assert code == 1 and "tokenizer_config.json" in err and "ne les rejouent" in err
    assert not out.exists()


def test_session_open_and_status_show_the_freeze_ceiling(tmp_path, capsys):
    base = write_checkpoint(tmp_path / "base", base_tensors())
    root = ["--sessions-root", str(tmp_path / "sessions")]
    assert (
        slm_weight_editing.main(
            ["session-open", "--session", "p1", "--budget", "10", "--base-checkpoint", str(base)]
            + root
        )
        == 0
    )
    assert "Plafond du gel : au plus 10%" in capsys.readouterr().out
    assert slm_weight_editing.main(["session-status", "--session", "p1", *root]) == 0
    assert "Plafond du gel : au plus 10%" in capsys.readouterr().out


def cache_session(sess):
    """Séance s2 ouverte sur une base qui contient `.cache/` (absent de work/), édition copiée."""
    cache = sess["base"] / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "x.metadata").write_text("meta", "utf-8")
    harness.open_session("s2", 10, sess["base"], base_revision=REV, sessions_root=sess["root"])
    work = sess["root"] / "s2" / "work"
    assert not (work / ".cache").exists()
    for f in (sess["dir"] / "work").glob("*.safetensors"):
        shutil.copy(f, work / f.name)
    return lambda: freeze.freeze_session("s2", sess["base"], sessions_root=sess["root"])


def test_freeze_ignores_hf_cache_files_absent_from_work(sess):
    freeze_s2 = cache_session(sess)
    (sess["root"] / "s2" / "work" / "config.json").write_text('{"a": 3}', "utf-8")
    diff = freeze_s2()
    assert [f["path"] for f in diff["files_changed"]] == ["config.json"]
    assert freeze.changed_files({".cache/huggingface/x": "aa"}, {}) == []
    assert freeze.changed_files({".cache/a": "aa"}, {".cache/a": "bb", "cache.txt": "cc"}) == [
        {"path": "cache.txt", "sha256_before": None, "sha256_after": "cc"}
    ]


def test_freeze_with_only_cache_files_gives_empty_files_changed(sess):
    assert cache_session(sess)()["files_changed"] == []


def cache_only_diff(tmp_path):
    """Diff de séance dont files_changed ne contient que des entrées `.cache/` (comme s1)."""
    s, root = gguf_session(tmp_path)
    freeze.freeze_session("g1", s.base_checkpoint, sessions_root=root)
    path = root / "g1" / "freeze" / "diff.json"
    meta = json.loads(path.read_text("utf-8"))
    meta["files_changed"] = [
        {"path": f".cache/huggingface/download/{n}.metadata", "sha256_before": "ab" * 32}
        | {"sha256_after": None}
        for n in ("a", "b")
    ]
    path.write_text(json.dumps(meta), "utf-8")
    return s, path


def test_cache_only_files_changed_do_not_block_apply_diff_and_report(tmp_path, capsys):
    s, diff_path = cache_only_diff(tmp_path)
    replayed = tmp_path / "models" / "rejoue"
    assert (
        slm_weight_editing.main(
            ["apply-diff", "--diff", str(diff_path), "--original", str(s.base_checkpoint)]
            + ["--out", str(replayed)]
        )
        == 0
    )
    assert "Attention" not in capsys.readouterr().out
    out = tmp_path / "gguf"
    code = slm_weight_editing.main(
        ["report-gguf", "--checkpoint", str(replayed), "--tag", "t:1"]
        + ["--base-checkpoint", str(s.base_checkpoint), "--ollama-models", str(s.store)]
        + ["--edit-diff", str(diff_path), "--out", str(out)]
    )
    assert code == 0, capsys.readouterr().err
    assert (out / "t-1.gguf").exists()


def test_cache_entry_does_not_hide_another_changed_file(tmp_path, capsys):
    s, diff_path = cache_only_diff(tmp_path)
    meta = json.loads(diff_path.read_text("utf-8"))
    meta["files_changed"].append({"path": "config.json", "sha256_before": None})
    diff_path.write_text(json.dumps(meta), "utf-8")
    code = slm_weight_editing.main(
        ["report-gguf", "--checkpoint", str(s.base_checkpoint), "--tag", "t:1"]
        + ["--base-checkpoint", str(s.base_checkpoint)]
        + ["--ollama-models", str(s.store), "--edit-diff", str(diff_path)]
        + ["--out", str(tmp_path / "gguf")]
    )
    err = capsys.readouterr().err
    assert code == 1 and "(config.json)" in err and "ne les rejouent" in err


@pytest.mark.parametrize(
    ("path", "ignored"),
    [
        (".cache/huggingface/download/x.metadata", True),
        (r".cache\huggingface\x.lock", True),
        (".cachefoo/x", False),
        ("a/.cache/x", False),
        (r"a\.cache\x", False),
        (".cache/../config.json", False),
        (r".cache\..\config.json", False),
        ("config.json", False),
        (None, False),
    ],
)
def test_is_ignored_file_only_ignores_the_top_level_cache(path, ignored):
    assert freeze.is_ignored_file(path) is ignored
