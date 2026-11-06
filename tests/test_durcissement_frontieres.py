"""Story 2.9 : durcissement des frontières 2.5/2.6/2.8 (constats A-12 ; CAP-6, CAP-9, CAP-10).

Un test (ou un groupe paramétré) par critère de la fiche `story-2-9-durcissement-frontieres.md`.
Tenseurs synthétiques, vrais sous-processus pour le crochet, aucun modèle ni téléchargement.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import COLS, ROWS, to_bf16, write_safetensors
from test_audit_transcript import script_event

from slm_weight_editing import _refusal_hook as hook
from slm_weight_editing import audit_transcript as audit
from slm_weight_editing import freeze, harness, lm_head, main, probe, workspace
from slm_weight_editing.harness import HarnessError

REPO = Path(__file__).resolve().parents[1]
EMBED = "model.embed_tokens.weight"
NORM = "model.norm.weight"


def tied_base(path: Path) -> Path:
    """Checkpoint synthétique aux embeddings liés (`tie_word_embeddings: true`)."""
    rng = np.random.default_rng(3)
    path.mkdir(parents=True)
    write_safetensors(
        path / "model.safetensors",
        {
            EMBED: to_bf16(rng.normal(0, 0.05, (ROWS, COLS)).astype(np.float32)),
            NORM: to_bf16(rng.normal(1, 0.05, (COLS,)).astype(np.float32)),
        },
    )
    (path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "tie_word_embeddings": True}), "utf-8"
    )
    return path


@pytest.fixture(scope="module")
def base(tmp_path_factory) -> Path:
    return tied_base(tmp_path_factory.mktemp("base") / "ckpt")


@pytest.fixture
def ws(tmp_path, base) -> Path:
    data = tmp_path / "src-data" / "gsm8k-train.jsonl"
    data.parent.mkdir()
    data.write_text('{"split": "train", "question": "1+1"}\n', encoding="utf-8")
    return workspace.create_workspace(tmp_path / "ws", "s1", 10, base, [data], base_revision="r0")


def session(ws: Path) -> Path:
    return ws / "sessions" / "s1"


def payload(name: str, tool_input: dict, cwd: Path) -> dict:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": name,
        "tool_input": tool_input,
        "cwd": str(cwd),
    }


def rules(found) -> set[str]:
    return {r for r, _ in found}


def hook_rules(ws: Path, name: str, tool_input: dict) -> set[str]:
    """Règles du refus rendu par le crochet (vide si l'appel est laissé aux réglages)."""
    out = hook.decide(payload(name, tool_input, ws), hook.context(ws))
    if out is None:
        return set()
    text = out["hookSpecificOutput"]["permissionDecisionReason"]
    return {line.split(" : ", 1)[0] for line in text.splitlines()[:-1]}


def call_hook(ws: Path, data: bytes) -> subprocess.CompletedProcess:
    settings = json.loads((ws / ".claude" / "settings.json").read_text(encoding="utf-8"))
    entry = settings["hooks"]["PreToolUse"][0]["hooks"][0]
    return subprocess.run(
        [entry["command"], *entry["args"]],
        input=data,
        capture_output=True,
        cwd=ws,
        timeout=60,
        check=False,
    )


# --- Point 1 : état de séance écrit sans verbe reconnu (critères 1 à 3) -----------------------

TARGETS = {
    "state.json": "sessions/s1/state.json",
    "journal.jsonl": "sessions/s1/journal.jsonl",
    "registry.jsonl": "sessions/s1/registry.jsonl",
    "session.lock": "sessions/s1/session.lock",
    "workspace.json": "workspace.json",
}
FORMS = {
    "tofile": 'import numpy as np\nnp.zeros(3).tofile("{p}")\n',
    "pickle": 'import pickle\npickle.dump({{}}, open("{p}", "wb"))\n',
    "os.open": 'import os\nos.close(os.open("{p}", os.O_WRONLY))\n',
    "memmap": 'import numpy as np\nm = np.memmap("{p}", mode="w+", shape=(4,))\n',
    "truncate": 'import os\nos.truncate("{p}", 0)\n',
}


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("target", sorted(TARGETS))
def test_c1_state_write_without_known_verb_is_refused(ws, form, target):
    code = FORMS[form].format(p=TARGETS[target])
    tool_input = {"file_path": str(ws / "scripts" / "x.py"), "content": code}
    found = audit.check_tool_use("Write", tool_input, hook.context(ws))
    assert ("etat-seance", f"écriture ou mention visant « {target} »") in found, found
    assert "etat-seance" in hook_rules(ws, "Write", tool_input)


def test_c1_state_write_forms_are_refused_by_audit_scripts(ws):
    s = session(ws)
    codes = [FORMS[f].format(p=TARGETS["state.json"]) for f in sorted(FORMS)]
    with (s / "journal.jsonl").open("a", encoding="utf-8") as f:
        for code in codes:
            f.write(json.dumps(script_event(s, code)) + "\n")
    first = len(harness.read_journal(s)) - len(codes) + 1
    flagged = {v.line for v in audit.audit_scripts(s) if v.rule == "etat-seance"}
    assert flagged == set(range(first, first + len(codes)))


@pytest.mark.parametrize(
    "code",
    [
        'p = "sessions/s1/state.json"\nkeep(p)\n',
        'def keep(x):\n    return x\n\nPATH = "sessions/s1/journal.jsonl"\nkeep(PATH)\n',
        'target = "workspace.json"\n',
    ],
)
def test_c2_mention_without_any_write_call_is_refused(ws, code):
    tool_input = {"file_path": str(ws / "scripts" / "x.py"), "content": code}
    assert "etat-seance" in rules(audit.check_tool_use("Write", tool_input, hook.context(ws)))
    assert "etat-seance" in rules(audit.check_code_content(code, "run-script"))


@pytest.mark.parametrize(
    ("name", "tool_input"),
    [
        ("Read", {"file_path": "sessions/s1/journal.jsonl"}),
        ("Grep", {"pattern": "batch", "path": "sessions/s1/journal.jsonl"}),
        ("Glob", {"pattern": "sessions/s1/journal.jsonl"}),
    ],
)
def test_c3_reading_the_journal_by_tool_has_no_finding(ws, name, tool_input):
    assert audit.check_tool_use(name, tool_input, hook.context(ws)) == []
    assert hook_rules(ws, name, tool_input) == set()


# --- Point 2 : crochet en échec, délai, `_SHADOWING` (critères 4 à 8) --------------------------


@pytest.fixture
def stdio(monkeypatch):
    """Entrée et sortie standard du crochet en mémoire (frontière : flux du processus)."""

    def feed(data: bytes) -> io.BytesIO:
        out = io.BytesIO()
        monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(data)))
        monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(out))
        return out

    return feed


def denied(out: io.BytesIO) -> str:
    decision = json.loads(out.getvalue())["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    assert decision["hookEventName"] == "PreToolUse"
    return decision["permissionDecisionReason"]


@pytest.mark.parametrize(
    ("data", "detail"),
    [
        (b"", "entrée illisible (JSONDecodeError)"),
        (b"\xff\xfe{", "entrée illisible (UnicodeDecodeError)"),
        (b'"texte"', "entrée illisible (objet JSON attendu, reçu str)"),
    ],
)
def test_c4_unreadable_input_is_denied_with_failure_rule(ws, stdio, data, detail):
    out = stdio(data)
    assert hook.main(["hook", str(ws)]) == 0
    text = denied(out)
    assert text == "\n".join(
        [audit.refusal_message("crochet-en-echec", detail), hook.BLOCAGES_HINT]
    )


def test_c4_exception_in_context_is_denied(ws, stdio):
    (ws / "workspace.json").write_text('{"repo": "x"}', encoding="utf-8")  # session_id absent
    out = stdio(json.dumps(payload("Read", {"file_path": "CONSIGNE.md"}, ws)).encode())
    assert hook.main(["hook", str(ws)]) == 0
    text = denied(out)
    assert text.startswith("crochet-en-echec : exception KeyError : 'session_id' ; permis : ")
    assert text.endswith(hook.BLOCAGES_HINT)


def test_c4_exception_in_decide_is_denied(ws, stdio):
    # Entrée imbriquée au-delà de la pile Python : l'audit lève RecursionError, l'appel est refusé.
    nested = "[" * 3000 + "]" * 3000
    data = '{"tool_name": "Read", "tool_input": {"file_path": "x", "v": ' + nested + "}}"
    out = stdio(data.encode())
    assert hook.main(["hook", str(ws)]) == 0
    assert denied(out).startswith("crochet-en-echec : exception RecursionError")


def test_c4_hook_never_answers_allow_or_ask(ws, stdio):
    for data in (b"", b"[]", json.dumps(payload("Bash", {"command": "cat x"}, ws)).encode()):
        out = stdio(data)
        hook.main(["hook", str(ws)])
        decision = json.loads(out.getvalue())["hookSpecificOutput"]["permissionDecision"]
        assert decision == "deny"


def test_c5_settings_block_on_failure_and_deadline_under_timeout(ws):
    settings = json.loads((ws / ".claude" / "settings.json").read_text(encoding="utf-8"))
    entry = settings["hooks"]["PreToolUse"][0]["hooks"][0]
    assert entry["onFailure"] == "block"
    assert entry["timeout"] == workspace.HOOK_TIMEOUT == 10
    assert 0 < hook.DEADLINE < workspace.HOOK_TIMEOUT


def test_c6_analysis_past_the_deadline_is_denied(ws, stdio):
    content = "x = 1\n" * 40_000  # analyse de plusieurs dizaines de ms, bien au-delà de 0 s
    tool_input = {"file_path": str(ws / "scripts" / "x.py"), "content": content}
    out = stdio(json.dumps(payload("Write", tool_input, ws)).encode())
    start = time.perf_counter()
    assert hook.main(["hook", str(ws)], deadline=0.0) == 0
    assert time.perf_counter() - start < 1.0
    text = denied(out)
    assert text.startswith("crochet-en-echec : analyse trop longue (plus de 0 s) ; permis : ")


def test_c6_hook_process_denies_failure_in_a_real_subprocess(ws):
    done = call_hook(ws, b"{pas du json")
    assert done.returncode == 0, done.stderr
    reason = json.loads(done.stdout)["hookSpecificOutput"]["permissionDecisionReason"]
    assert reason.startswith("crochet-en-echec : entrée illisible")


SHADOW_WORST = {
    "for-sans-in": lambda n: "for " + "a, " * (n // 3),
    "for-lignes": lambda n: ("for " + "a, b, c, " * 10 + "\n") * (n // 95),
    "for-open": lambda n: "for " + "open, " * (n // 6),
    "for-for": lambda n: "for " * (n // 4),
    "lambda-sans-deux-points": lambda n: "lambda " + "open " * (n // 5),
    "lambda-lambda": lambda n: "lambda " * (n // 7),
    "virgule-sans-egal": lambda n: "open, " * (n // 6),
    "crochet-ouvert": lambda n: "open[" * (n // 5),
    "global": lambda n: "global " * (n // 7),
    "import": lambda n: "import " * (n // 7),
}


# Une mention de l'état de séance fait passer tout le contenu par l'allègement des lectures.
_MENTION = 'print(open("sessions/s1/journal.jsonl").read())\n'


def _timed(text: str) -> float:
    start = time.perf_counter()
    audit.check_code_content(text, "Write")
    return time.perf_counter() - start


@pytest.mark.parametrize("kind", sorted(SHADOW_WORST))
def test_c7_shadowing_worst_case_is_linear(kind):
    make = SHADOW_WORST[kind]
    small, large = _MENTION + make(100_000), _MENTION + make(200_000)
    assert 190_000 <= len(large) <= 200_100
    # Meilleure de trois mesures : seuils larges, tolérants à une CI chargée.
    t_small = min(_timed(small) for _ in range(3))
    t_large = min(_timed(large) for _ in range(3))
    # 0,5 s (fiche) sur un poste ; x3 en CI (machines partagées). L'ancienne forme quadratique
    # prenait des dizaines de secondes à 200 ko : les deux seuils l'attrapent.
    limit = 0.5 * (3 if os.environ.get("CI") else 1)
    assert t_large < limit, f"{kind} : {t_large:.3f} s pour 200 ko"
    assert t_large < 3 * max(t_small, 0.01), f"{kind} : {t_small:.3f} s puis {t_large:.3f} s"


@pytest.mark.parametrize(
    "code",
    [
        'print = lambda p: p\nprint(open("sessions/s1/journal.jsonl").read())\n',
        'def open(p, m="r"):\n    return p\nprint(open("sessions/s1/journal.jsonl").read())\n',
        'for open in [print]:\n    pass\nprint(open("sessions/s1/journal.jsonl").read())\n',
        'f = lambda Path: Path\nprint(Path("sessions/s1/journal.jsonl").read_text())\n',
        'open, x = print, 1\nprint(open("sessions/s1/journal.jsonl").read())\n',
        'open[0] = 1\nprint(open("sessions/s1/journal.jsonl").read())\n',
        'def f(a, Path):\n    pass\nprint(Path("sessions/s1/journal.jsonl").read_text())\n',
        'global open\nprint(open("sessions/s1/journal.jsonl").read())\n',
        'from x import open\nprint(open("sessions/s1/journal.jsonl").read())\n',
        'from x import Path\nprint(Path("sessions/s1/journal.jsonl").read_text())\n',
    ],
)
def test_c8_shadowing_cases_stay_refused(code):
    assert audit._shadows(code)
    assert "etat-seance" in rules(audit.check_code_content(code, "run-script"))


def test_c8_plain_read_without_shadowing_is_still_exempt():
    code = 'print(open("sessions/s1/journal.jsonl").read())\n'
    assert not audit._shadows(code)
    assert audit.check_code_content(code, "run-script") == []


# --- Point 3 : prose de EDITION.md et notes/ (critères 9 à 11) ---------------------------------

PROSE = (
    "J'ai lu sessions/s1/state.json et le journal ; pas de rm ni de git. Rien de heldout.\n"
    'Le script fait open(p, "w") puis chr(65) ; base64 et subprocess interdits.\n'
)


@pytest.mark.parametrize("rel", ["EDITION.md", "notes/x.md", "notes/sous/y.txt"])
@pytest.mark.parametrize("tool", ["Write", "Edit"])
def test_c9_prose_is_not_audited_as_code(ws, rel, tool):
    key = "content" if tool == "Write" else "new_string"
    tool_input = {"file_path": str(ws / rel), key: PROSE}
    if tool == "Edit":
        tool_input["old_string"] = "x"
    assert audit.check_tool_use(tool, tool_input, hook.context(ws)) == []
    assert hook_rules(ws, tool, tool_input) == set()


@pytest.mark.parametrize(
    ("path", "rule"),
    [
        ("notes/state.json", "etat-seance"),
        ("sessions/s1/notes.md", "etat-seance"),
        ("CLAUDE.md", "harnais-modifie"),
        ("README.md", "ecriture-hors-liste"),
        ("../x.md", "ecriture-hors-liste"),
    ],
)
def test_c9_path_rules_still_apply_to_prose(ws, path, rule):
    tool_input = {"file_path": str(ws / path), "content": PROSE}
    assert rule in rules(audit.check_tool_use("Write", tool_input, hook.context(ws)))
    assert rule in hook_rules(ws, "Write", tool_input)


def test_c10_same_text_in_a_script_keeps_its_findings(ws):
    tool_input = {"file_path": str(ws / "scripts" / "x.py"), "content": PROSE}
    found = rules(audit.check_tool_use("Write", tool_input, hook.context(ws)))
    assert {"etat-seance", "lecture-interdite", "obfuscation", "python-hors-lanceur"} <= found


@pytest.mark.parametrize(
    "rel", ["notes/x.py", "notes/x.PY", "notes/x.py.", "notes/x.pyw", "notes/a.pth", "EDITION.py"]
)
def test_c11_python_file_outside_scripts_is_refused(ws, rel):
    tool_input = {"file_path": str(ws / rel), "content": "x = 1\n"}
    assert "ecriture-hors-liste" in rules(
        audit.check_tool_use("Write", tool_input, hook.context(ws))
    )
    ok = {"file_path": str(ws / "scripts" / "x.py"), "content": "x = 1\n"}
    assert audit.check_tool_use("Write", ok, hook.context(ws)) == []


@pytest.mark.parametrize(
    "code",
    [
        'exec(open("notes/x.md").read())',
        'import runpy\nrunpy.run_path("notes/x.md")',
        'import importlib\nimportlib.import_module("notes.x")',
        'import sys\nsys.path.insert(0, "notes")\nimport x',
        'import sys\nsys.path.append("notes")',
        'from sys import path\npath.append("notes")',
    ],
)
def test_c11_script_executing_notes_is_refused(ws, code):
    tool_input = {"file_path": str(ws / "scripts" / "x.py"), "content": code}
    found = rules(audit.check_tool_use("Write", tool_input, hook.context(ws)))
    assert found & {"obfuscation", "api-harnais"}, found


# --- Point 4 : LSP et `filePath` (critères 12 et 13) -------------------------------------------


def _outside(ws: Path) -> dict[str, str]:
    held = ws.parent / "heldout"
    held.mkdir(exist_ok=True)
    (held / "math.jsonl").write_text("{}\n", encoding="utf-8")
    return {
        "depot": str(REPO / "pyproject.toml"),
        "heldout": str(held / "math.jsonl"),
        "parent": "../x.py",
    }


@pytest.mark.parametrize("target", ["depot", "heldout", "parent"])
def test_c12_lsp_file_path_outside_the_workspace_is_refused(ws, target):
    tool_input = {"operation": "hover", "filePath": _outside(ws)[target], "line": 1}
    assert "lecture-hors-espace" in rules(audit.check_tool_use("LSP", tool_input, hook.context(ws)))
    assert "lecture-hors-espace" in hook_rules(ws, "LSP", tool_input)


def test_c12_lsp_through_a_junction_in_the_workspace_is_refused(ws):
    link = ws / "notes" / "lien"
    try:
        if sys.platform == "win32":
            done = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(REPO)],
                capture_output=True,
                check=False,
            )
            made = done.returncode == 0 and link.exists()
        else:
            link.symlink_to(REPO, target_is_directory=True)
            made = True
    except OSError:
        made = False
    if not made:
        pytest.skip("jonction ou lien symbolique non créable sur ce poste")
    tool_input = {"operation": "hover", "filePath": str(link / "pyproject.toml"), "line": 1}
    assert "lecture-hors-espace" in hook_rules(ws, "LSP", tool_input)


@pytest.mark.parametrize("key", audit.READ_PATH_KEYS)
def test_c13_every_read_path_key_is_checked_the_same_way(ws, key):
    assert key in {"file_path", "path", "notebook_path", "filePath"}
    ctx = hook.context(ws)
    inside = {key: str(ws / "scripts" / "x.py")}
    outside = {key: str(REPO / "pyproject.toml")}
    for tool in ("Read", "Grep", "LSP"):
        assert audit.check_tool_use(tool, inside, ctx) == []
        assert ("lecture-hors-espace", f"{key} {outside[key]}") in audit.check_tool_use(
            tool, outside, ctx
        )


# --- Point 5 : reprise de `detach-bak` et ligne de journal (critères 14 à 17) -----------------


@pytest.fixture
def sess(tmp_path):
    base = tied_base(tmp_path / "base")
    root = tmp_path / "sessions"
    harness.open_session("s1", 10, base, base_revision="rev", sessions_root=root)
    return {"base": base, "root": root, "dir": root / "s1", "work": root / "s1" / "work"}


def kill_detach_before_the_journal_line(directory: Path) -> None:
    """État laissé par un `detach-lm-head` tué juste avant sa ligne de journal."""
    work = directory / harness.WORK_DIR
    backups = directory / lm_head.BACKUP_DIR
    backups.mkdir()
    config = work / lm_head.CONFIG_FILE
    raw = config.read_bytes()
    lm_head.write_config_bytes(backups / lm_head.CONFIG_FILE, raw)
    st = lm_head.open_checkpoint(work)[EMBED]
    lm_head.append_tensor(
        st.path,
        lm_head.LM_HEAD,
        st.dtype,
        list(st.shape),
        st.read_bytes(),
        backup=backups / st.path.name,
    )
    lm_head.write_config(config, lm_head.transform_config(raw.decode("utf-8"))[0])


def _work_bytes(work: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(work.iterdir())}


def _dead_lock(directory: Path) -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    (directory / harness.LOCK_FILE).write_text(
        json.dumps({"pid": proc.pid, "acquired_at": "2026-11-06T10:00:00+00:00"}), "utf-8"
    )


RESTORED = "détachement de lm_head interrompu"


def _unlock(s):
    assert harness.unlock_session("s1", s["root"]) is None


def _unlock_dead_lock(s):
    _dead_lock(s["dir"])
    entry = harness.unlock_session("s1", s["root"])
    assert entry is not None and entry["event"] == "unlock"


def _run_batch(s):
    # Données absentes : refus après la reprise, faite avant toute autre action.
    with pytest.raises(HarnessError, match="données illisibles"):
        data = s["root"] / "x-train.jsonl"
        harness.run_batch("s1", "gsm8k", "train", data, thinking="off", sessions_root=s["root"])


def _probe(s):
    with pytest.raises(HarnessError):
        probe.run_probe(
            "s1", s["root"] / "absent.jsonl", capture=["resid"], sessions_root=s["root"]
        )


COMMANDS = {"unlock": _unlock, "unlock-verrou-mort": _unlock_dead_lock, "run-batch": _run_batch}
COMMANDS["probe"] = _probe


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_c14_killed_detach_is_rolled_back_by_the_next_command(sess, capsys, command):
    before = _work_bytes(sess["work"])
    kill_detach_before_the_journal_line(sess["dir"])
    assert _work_bytes(sess["work"]) != before
    COMMANDS[command](sess)
    assert _work_bytes(sess["work"]) == before
    assert not (sess["dir"] / lm_head.BACKUP_DIR).exists()
    assert f"Reprise : {RESTORED}" in capsys.readouterr().err
    assert lm_head.detach_event(harness.read_journal(sess["dir"])) is None


def test_c14_freeze_rolls_back_and_hashes_work_as_before_the_attempt(sess, capsys):
    hashes = harness.work_file_hashes(sess["work"])
    kill_detach_before_the_journal_line(sess["dir"])
    diff = freeze.freeze_session("s1", sess["base"], sessions_root=sess["root"])
    assert f"Reprise : {RESTORED}" in capsys.readouterr().err
    assert harness.work_file_hashes(sess["work"]) == hashes
    assert not (sess["dir"] / lm_head.BACKUP_DIR).exists()
    assert (diff["detached_lm_head"], diff["files_changed"], diff["tensors"]) == (None, [], [])


def test_c14_without_backup_nothing_is_said(sess, capsys):
    assert harness.recover_detach(sess["dir"]) is None
    _unlock(sess)
    assert "Reprise" not in capsys.readouterr().err


def test_c15_backup_after_the_journal_line_is_dropped_and_freeze_accepts(sess, capsys):
    lm_head.detach_lm_head("s1", sessions_root=sess["root"])
    detached = _work_bytes(sess["work"])
    backups = sess["dir"] / lm_head.BACKUP_DIR
    backups.mkdir()
    (backups / "model.safetensors").write_bytes((sess["base"] / "model.safetensors").read_bytes())
    (backups / "config.json").write_bytes((sess["base"] / "config.json").read_bytes())
    diff = freeze.freeze_session("s1", sess["base"], sessions_root=sess["root"])
    err = capsys.readouterr().err
    assert "Reprise : sauvegarde detach-bak/ d'un détachement de lm_head journalisé" in err
    assert _work_bytes(sess["work"]) == detached and not backups.exists()
    assert diff["detached_lm_head"]["lm_head"] == lm_head.LM_HEAD
    assert diff["files_changed"] == []
    assert [(t["name"], t["added"], t["changed"]) for t in diff["tensors"]] == [
        (lm_head.LM_HEAD, True, 0)
    ]


@pytest.mark.parametrize("error", [OSError("disque plein"), KeyboardInterrupt()])
def test_c16_error_after_the_journal_line_keeps_work_detached(sess, monkeypatch, error):
    real = harness._append_journal

    def write_then_fail(directory, entry):
        real(directory, entry)
        raise error

    monkeypatch.setattr(harness, "_append_journal", write_then_fail)
    with pytest.raises(type(error)):
        lm_head.detach_lm_head("s1", sessions_root=sess["root"])
    monkeypatch.undo()
    entry = lm_head.detach_event(harness.read_journal(sess["dir"]))
    assert entry is not None
    model = sess["work"] / "model.safetensors"
    assert harness._sha256_file(model) == entry["file_sha256_after"]
    assert lm_head.LM_HEAD in lm_head.open_checkpoint(sess["work"])
    config = (sess["work"] / "config.json").read_bytes()
    assert lm_head._sha(config) == entry["config"]["sha256_after"]
    assert not (sess["dir"] / lm_head.BACKUP_DIR).exists()
    assert sorted(p.name for p in sess["work"].iterdir()) == ["config.json", "model.safetensors"]
    with pytest.raises(HarnessError, match="déjà détaché"):
        lm_head.detach_lm_head("s1", sessions_root=sess["root"])


@pytest.mark.parametrize("error", [OSError("disque plein"), KeyboardInterrupt()])
def test_c17_error_before_the_journal_line_restores_work(sess, monkeypatch, error):
    before = _work_bytes(sess["work"])

    def fail(directory, entry):
        raise error

    monkeypatch.setattr(harness, "_append_journal", fail)
    with pytest.raises(type(error)):
        lm_head.detach_lm_head("s1", sessions_root=sess["root"])
    monkeypatch.undo()
    assert _work_bytes(sess["work"]) == before
    assert lm_head.detach_event(harness.read_journal(sess["dir"])) is None
    assert not (sess["dir"] / lm_head.BACKUP_DIR).exists()


# --- Point 6 : `detach-lm-head` dans un espace réel (critère 18) -------------------------------


def test_c18_detach_in_a_created_workspace_end_to_end(ws, base, tmp_path, monkeypatch, capsys):
    s = session(ws)
    prefix = next(c for c in workspace.allowed_commands(REPO) if c.endswith("detach-lm-head"))
    command = f"{prefix} --session s1 --sessions-root sessions"
    assert hook.decide(payload("Bash", {"command": command}, ws), hook.context(ws)) is None
    done = call_hook(ws, json.dumps(payload("Bash", {"command": command}, ws)).encode())
    assert (done.returncode, done.stdout) == (0, b""), done.stderr

    monkeypatch.chdir(ws)
    assert main(["detach-lm-head", "--session", "s1", "--sessions-root", "sessions"]) == 0
    assert "lm_head détaché" in capsys.readouterr().out
    assert lm_head.detach_event(harness.read_journal(s)) is not None
    assert not (s / lm_head.BACKUP_DIR).exists()
    assert sorted(p.name for p in (s / "work").iterdir()) == ["config.json", "model.safetensors"]

    monkeypatch.chdir(tmp_path)
    freeze.freeze_session("s1", base, sessions_root=ws / "sessions")
    assert harness.read_state(s).frozen

    transcript = tmp_path / "t.jsonl"
    call = {
        "type": "assistant",
        "cwd": ws.as_posix(),
        "message": {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t0", "name": "Bash", "input": {"command": command}}
            ],
        },
    }
    transcript.write_text(json.dumps(call) + "\n", encoding="utf-8")
    capsys.readouterr()
    args = ["audit-session", str(transcript), "--workspace", str(ws), "--session-dir", str(s)]
    args += ["--reference-from", str(base), "--base-revision", "r0"]
    code = main(args)
    out = capsys.readouterr().out
    assert code == 0, out
    for rule in ("lien", "etat-seance", "base-differente", "script-copie", "hors-espace-seance"):
        assert f"{rule} ·" not in out, out
