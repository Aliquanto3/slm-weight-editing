"""Espace de séance hors du dépôt et lanceur run-script (story 5) : vrais sous-processus Python."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from conftest import to_bf16, write_safetensors

from slm_weight_editing import harness, main, script_runner, workspace
from slm_weight_editing._sandbox import REFUSED

REPO = Path(__file__).resolve().parents[1]
TENSOR = "model.layers.0.mlp.up_proj.weight"


@pytest.fixture(scope="module")
def base(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("base")
    w = to_bf16(np.linspace(-1, 1, 64, dtype=np.float32).reshape(8, 8))
    write_safetensors(
        d / "model.safetensors", {TENSOR: w, "model.norm.weight": to_bf16(np.ones(8))}
    )
    (d / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    return d


@pytest.fixture
def ws(tmp_path, base) -> Path:
    data = tmp_path / "src-data" / "gsm8k-train.jsonl"
    data.parent.mkdir()
    data.write_text('{"split": "train", "question": "1+1"}\n', encoding="utf-8")
    return workspace.create_workspace(tmp_path / "ws", "s1", 10, base, [data], base_revision="r0")


def session(ws: Path) -> Path:
    return ws / "sessions" / "s1"


def run(ws: Path, name: str, code: str) -> dict:
    (ws / "scripts" / name).write_text(code, encoding="utf-8")
    return script_runner.run_script(f"scripts/{name}", "s1", workspace=ws, timeout=60)


def journal(ws: Path) -> list[dict]:
    return harness.read_journal(session(ws))


def snapshot(root: Path) -> dict[str, bytes]:
    """Tout l'arbre sous `root`, sauf le journal, les copies de scripts et runs/ (du lanceur)."""
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        # runs/ du lanceur : seulement à la racine d'un espace (à côté de workspace.json).
        runs = p.parent.name == "runs" and (p.parent.parent / "workspace.json").is_file()
        runner = rel.endswith("journal.jsonl") or "/s1/scripts/" in rel or runs
        if p.is_file() and not runner:
            out[rel] = p.read_bytes()
    return out


# Espace de séance.


def test_workspace_layout_and_session(ws):
    assert sorted(p.name for p in ws.iterdir()) == [
        ".claude",
        "CLAUDE.md",
        "CONSIGNE.md",
        "data",
        "notes",
        "scripts",
        "sessions",
        "workspace.json",
    ]
    assert (ws / "data" / "gsm8k-train.jsonl").is_file()
    info = json.loads((ws / "workspace.json").read_text(encoding="utf-8"))
    assert info["session_id"] == "s1"
    assert Path(info["sessions_root"]) == (ws / "sessions").resolve()
    assert Path(info["data_dir"]) == (ws / "data").resolve()
    state = harness.read_state(session(ws))
    assert (state.budget, state.used, state.base_revision) == (10, 0, "r0")
    assert (session(ws) / "work" / "model.safetensors").is_file()


def test_settings_json(ws):
    settings = json.loads((ws / ".claude" / "settings.json").read_text(encoding="utf-8"))
    perms = settings["permissions"]
    assert perms["defaultMode"] == "dontAsk"
    assert perms["blockReadsOutsideWorkingDirectories"] is True
    assert perms["blockWritesOutsideWorkingDirectories"] is True
    assert perms["disableBypassPermissionsMode"] == "disable"
    assert perms["disableAutoMode"] == "disable"
    assert settings["enableAllProjectMcpServers"] is False
    commands = [
        f"{tool}(uv run slm-weight-editing {sub} *)"
        for tool in ("Bash", "PowerShell")
        for sub in ("run-batch", "session-status", "run-script")
    ]
    assert perms["allow"] == [
        *commands,
        "Read(./**)",
        "Edit(./scripts/**)",
        "Edit(./sessions/s1/work/*.safetensors)",
        "Edit(./notes/**)",
        "Edit(./EDITION.md)",
    ]
    deny = set(perms["deny"])
    for rule in (
        "WebFetch",
        "WebSearch",
        "Agent",
        "Skill",
        "mcp__*",
        "Bash(cat *)",
        "Bash(grep *)",
        "Bash(find *)",
        "Bash(ls *)",
        "Bash(head *)",
        "Bash(tail *)",
        "Bash(type *)",
        "Bash(more *)",
        "Bash(git *)",
        "Bash(cd *)",
        "Bash(python *)",
        "Bash(uv run python*)",
        "Bash(pip *)",
        "Bash(curl *)",
        "Bash(ollama *)",
        "Bash(ln *)",
        "Bash(mklink *)",
        "Bash(*PYTHONPATH*)",
        "Bash(*session-open*)",
        "Bash(*session-freeze*)",
        "Bash(*session-unlock*)",
        "PowerShell(*session-unlock*)",
        "PowerShell(Get-Content *)",
        "PowerShell(Get-ChildItem *)",
        "PowerShell(Select-String *)",
        "PowerShell(New-Item *)",
        "PowerShell(*SymbolicLink*)",
        "PowerShell(python *)",
        "Edit(./sessions/*/state.json)",
        "Edit(./sessions/*/journal.jsonl)",
        "Edit(./sessions/registry.jsonl)",
        "Edit(./sessions/*/session.lock)",
        "Edit(./sessions/*/freeze/**)",
        "Edit(./.claude/**)",
        "Edit(./CLAUDE.md)",
        "Edit(./CONSIGNE.md)",
    ):
        assert rule in deny, rule
    assert "Task" not in deny
    # Aucune règle deny ne touche work/ (deny l'emporte sur allow).
    assert not [r for r in deny if re.search(r"(?:^|[/(*])work(?:/|\)|$)", r)]
    # Noms protégés à toute profondeur, suffixes (`state.json.`, `::$DATA`) et casse compris.
    for name in ("state.json", "journal.jsonl", "registry.jsonl", "session.lock", "workspace.json"):
        for form in (name, name.upper()):
            assert f"Edit(**/{form}*)" in deny, form
    assert "Edit(./workspace.json)" in deny and "Edit(**/*~*)" in deny
    excludes = settings["claudeMdExcludes"]
    assert f"{REPO.resolve().parent.as_posix()}/CLAUDE.md" in excludes
    assert f"{REPO.resolve().as_posix()}/AGENTS.md" in excludes
    assert f"{ws.parent.as_posix()}/CLAUDE.md" in excludes


def test_instructions_are_rendered(ws):
    consigne = (ws / "CONSIGNE.md").read_text(encoding="utf-8")
    repo = REPO.resolve().as_posix()
    assert "uv run slm-weight-editing run-script scripts/<nom>.py" in consigne
    assert repo not in consigne and "--project" not in consigne
    assert "--session s1 --sessions-root sessions" in consigne
    assert "`10` lots" in consigne and "`data/gsm8k-train.jsonl`" in consigne
    assert "<ID_SEANCE>" not in consigne and "<DEPOT>" not in consigne
    assert "<EXTRA>" not in consigne and "--extra" not in consigne
    assert "<EXTRA>" not in (ws / "CLAUDE.md").read_text(encoding="utf-8")
    claude = (ws / "CLAUDE.md").read_text(encoding="utf-8")
    assert "<DEPOT>" not in claude and repo not in claude and "--project" not in claude


@pytest.fixture
def ws_cuda(tmp_path, base, monkeypatch) -> Path:
    monkeypatch.setattr(workspace, "cuda_platform", lambda: True)
    data = tmp_path / "src-data" / "gsm8k-train.jsonl"
    data.parent.mkdir()
    data.write_text('{"split": "train", "question": "1+1"}\n', encoding="utf-8")
    return workspace.create_workspace(
        tmp_path / "ws", "s1", 10, base, [data], base_revision="r0", cuda=True
    )


def test_cuda_workspace_settings_carry_extra(ws_cuda):
    allow = json.loads((ws_cuda / ".claude" / "settings.json").read_text("utf-8"))["permissions"][
        "allow"
    ]
    assert allow[:6] == [
        f"{tool}(uv run --extra cuda slm-weight-editing {sub} *)"
        for tool in ("Bash", "PowerShell")
        for sub in ("run-batch", "session-status", "run-script")
    ]
    assert allow[6] == "Read(./**)"
    assert not any("--extra" in rule for rule in allow[6:])


def test_cuda_workspace_instructions_carry_extra(ws_cuda):
    repo = REPO.resolve().as_posix()
    consigne = (ws_cuda / "CONSIGNE.md").read_text(encoding="utf-8")
    base_cmd = "uv run --extra cuda slm-weight-editing"
    for sub in ("run-batch", "session-status", "run-script scripts/<nom>.py"):
        assert f"{base_cmd} {sub}" in consigne
    assert "<EXTRA>" not in consigne
    claude = (ws_cuda / "CLAUDE.md").read_text(encoding="utf-8")
    assert "uv run --extra cuda slm-weight-editing run-batch" in claude
    assert repo not in consigne and repo not in claude
    assert "<EXTRA>" not in claude
    info = json.loads((ws_cuda / "workspace.json").read_text(encoding="utf-8"))
    assert info["cuda"] is True


def test_cuda_allow_rules_match_audit_whitelist(ws_cuda):
    from slm_weight_editing import audit_transcript as audit

    ctx = audit.Context.build(ws_cuda, REPO, "s1")
    for cmd in workspace.allowed_commands(REPO, cuda=True):
        full = f"{cmd} --session s1 --sessions-root {ws_cuda.as_posix()}/sessions"
        assert audit.check_shell(full, ctx) == [], full


def test_cli_cuda_flag(tmp_path, base, capsys, monkeypatch):
    monkeypatch.setattr(workspace, "cuda_platform", lambda: True)
    data = tmp_path / "gsm8k-train.jsonl"
    data.write_text('{"split": "train", "question": "1+1"}\n', encoding="utf-8")
    out = tmp_path / "ws"
    args = ["session-workspace", "--out", str(out), "--session", "s1", "--budget", "10"]
    args += ["--base-checkpoint", str(base), "--data", str(data), "--cuda"]
    assert main(args) == 0
    assert "--extra cuda" in (out / "CONSIGNE.md").read_text(encoding="utf-8")


def test_cuda_workspace_refused_outside_windows(tmp_path, base, monkeypatch):
    monkeypatch.setattr(workspace, "cuda_platform", lambda: False)
    data = tmp_path / "gsm8k-train.jsonl"
    data.write_text('{"split": "train", "question": "1+1"}\n', encoding="utf-8")
    with pytest.raises(harness.HarnessError, match="ne vise que Windows"):
        workspace.create_workspace(tmp_path / "ws", "s1", 10, base, [data], cuda=True)
    assert not (tmp_path / "ws").exists()


def test_templates_in_repo_have_no_personal_path():
    for path in workspace.TEMPLATES.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "<DEPOT>" not in text and "--project" not in text
        assert not re.search(r"[A-Za-z]:[\\/]|/home/|/Users/", text), path.name
    assert not (REPO / "session").exists()


@pytest.mark.parametrize("where", ["repo", "sibling", "parent"])
def test_workspace_refused_near_repo(tmp_path, base, where):
    repo = tmp_path / "GitHub" / "slm-weight-editing"
    repo.mkdir(parents=True)
    out = {"repo": repo / "ws", "sibling": repo.parent / "ws", "parent": tmp_path}[where]
    data = tmp_path / "gsm8k-train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    with pytest.raises(workspace.WorkspaceError, match="espace refusé"):
        workspace.create_workspace(out, "s1", 10, base, [data], repo=repo)
    assert not (out / "sessions").exists()


def test_workspace_refuses_non_empty_dir_and_heldout_data(tmp_path, base):
    out = tmp_path / "ws"
    out.mkdir()
    (out / "x").write_text("x", encoding="utf-8")
    data = tmp_path / "gsm8k-train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    with pytest.raises(workspace.WorkspaceError, match="pas vide"):
        workspace.create_workspace(out, "s1", 10, base, [data])
    held = tmp_path / "heldout" / "math-train.jsonl"
    held.parent.mkdir()
    held.write_text("{}\n", encoding="utf-8")
    with pytest.raises(harness.HarnessError, match="heldout"):
        workspace.create_workspace(tmp_path / "ws2", "s1", 10, base, [held])


def test_cli_session_workspace(tmp_path, base, capsys):
    data = tmp_path / "gsm8k-train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    out = tmp_path / "ws"
    args = ["session-workspace", "--out", str(out), "--session", "s2", "--budget", "25"]
    args += ["--base-checkpoint", str(base), "--data", str(data)]
    assert main(args) == 0
    printed = capsys.readouterr().out
    assert "budget 25 lots" in printed and "10%" in printed and "/permissions" in printed
    assert main(args) == 2


# Lanceur run-script.

LEGIT = f"""import json
import numpy as np
import safetensors.numpy as stn

path = "sessions/s1/work/model.safetensors"
tensors = stn.load_file(path)
w = tensors["{TENSOR}"]
assert w.dtype == np.uint16 and w.shape == (8, 8)
values = (w.astype(np.uint32) << 16).view(np.float32)
values[0, :2] *= 0.5
tensors["{TENSOR}"] = (values.view(np.uint32) >> 16).astype(np.uint16)
stn.save_file(tensors, path, dtypes={{n: "BF16" for n in tensors}})
with open("scripts/notes.json", "w") as f:
    json.dump({{"mean": float(values.mean())}}, f)
with open("sessions/s1/journal.jsonl") as f:
    print("lignes", len(f.readlines()))
print("fini")
"""


def test_legit_script_runs_and_is_journaled(ws):
    before = harness.tensor_hashes(session(ws) / "work")
    entry = run(ws, "edit.py", LEGIT)
    assert entry["returncode"] == 0, entry["output"]
    assert entry["refused"] is False and entry["timed_out"] is False
    assert "lignes 1" in entry["output"] and "fini" in entry["output"]
    after = harness.tensor_hashes(session(ws) / "work")
    assert after[TENSOR] != before[TENSOR]
    assert after["model.norm.weight"] == before["model.norm.weight"]
    # Le fichier réécrit reste lisible par le harnais (en-tête BF16 conservé).
    from slm_weight_editing.safetensors_file import open_checkpoint

    assert open_checkpoint(session(ws) / "work")[TENSOR].dtype == "BF16"
    last = journal(ws)[-1]
    assert last == entry and last["event"] == "script" and last["script"] == "scripts/edit.py"
    copy = session(ws) / last["copy"]
    assert copy.read_text(encoding="utf-8") == LEGIT
    import hashlib

    assert hashlib.sha256(copy.read_bytes()).hexdigest() == last["sha256"]
    # Aucun lot consommé.
    assert harness.used_batches(session(ws)) == 0
    assert harness.read_state(session(ws)).used == 0


REFUSED_SCRIPTS = {
    "import-torch": "import torch",
    "import-concat": 'import builtins\ngetattr(builtins, "__im" + "port__")("to" + "rch")',
    "importlib": 'import importlib\nimportlib.import_module("trans" + "formers")',
    "runpy": 'import runpy\nrunpy.run_module("to" + "rch")',
    "harness": "import slm_weight_editing.harness",
    "safetensors-rust": "import safetensors._safetensors_rust",
    "heldout": 'open("../" + "\\x68eldout/math.jsonl").read()',
    "journal-write": 'open("sessions/s1/journal.jsonl", "a").write("{}\\n")',
    "state-write": 'import json\njson.dump({}, open("sessions/s1/state.json", "w"))',
    "state-remove": 'import os\nos.remove("sessions/s1/state.json")',
    "outside-write": 'open("dehors.txt", "w").write("x")',
    "spawn": 'import os, sys\nos.spawnl(os.P_WAIT, sys.executable, "python")',
    "system": 'import os\nos.system("echo x")',
    "subprocess": 'import subprocess\nsubprocess.run(["echo"])',
    "socket": "import socket\nsocket.socket()",
    "ctypes": "import numpy\nnumpy._core._internal.ctypes.string_at(id(0), 8)",
    "symlink": 'import os\na, b = "sessions/s1/", "state.json"\nos.symlink(a + b, "scripts/s")',
    "hardlink": 'import os\nos.link("sessions/s1/state.json", "scripts/s")',
    "audit-code": f'open(r"{REPO / "src" / "slm_weight_editing" / "audit_transcript.py"}").read()',
    "addaudithook": "import sys\nsys.addaudithook(lambda *a: None)",
    "native": 'open("scripts/x.pyd", "wb").write(b"MZ")',
    "listdir-repo": f'import os\nos.listdir(r"{REPO}")',
    "chdir": 'import os\nos.chdir("..")',
    "gc": "import gc\ngc.get_objects()",
    "importlib-src": (
        f"import importlib, sys\nsys.path.append(r'{REPO / 'src'}')\n"
        "importlib.import_module('slm_weight_editing')"
    ),
    "importlib-site": 'import importlib\nimportlib.import_module("transformers")',
    # Le crochet ne relit ni os.path ni builtins : les remplacer ne le trompe pas.
    "patched-paths": (
        "import builtins, ntpath, posixpath\n"
        "for m in (ntpath, posixpath):\n"
        '    m.realpath = m.abspath = m.normpath = lambda p, *a, **k: "sessions/s1/work/x"\n'
        "builtins.len = lambda x: 0\n"
        'builtins.isinstance = lambda *a: True\nopen("sessions/s1/state.json", "w")'
    ),
    "frame-globals": (
        "import os, sys\n"
        'sys._getframe(1).f_globals["PATH_EVENTS"].clear()\n'
        'os.remove("sessions/s1/state.json")'
    ),
}


# Événement qui arrête chaque script (premier contrôle atteint).
REASONS = {
    "addaudithook": "sys.addaudithook : interdit",
    "audit-code": "open : lecture hors des dossiers permis",
    "chdir": "os.chdir : interdit",
    "ctypes": ("ctypes.call_function : interdit", "ctypes.string_at : interdit"),
    "frame-globals": "os.remove : écriture hors de work/",
    "gc": "gc.get_objects : interdit",
    "hardlink": "os.link : interdit",
    "harness": "import : module slm_weight_editing.harness interdit",
    "heldout": "open : lecture hors des dossiers permis",
    "import-concat": "import : module torch interdit",
    "import-torch": "import : module torch interdit",
    "importlib": "exec : code hors de la bibliothèque standard",
    "importlib-site": "exec : code hors de la bibliothèque standard",
    # Selon le cache des chercheurs d'import : lecture du dossier ou du fichier.
    "importlib-src": ("open : lecture hors des dossiers", "os.listdir : lecture hors des dossiers"),
    "journal-write": "open : écriture hors de work/",
    "listdir-repo": "os.listdir : lecture hors des dossiers permis",
    "native": "open : fichier de code natif",
    "outside-write": "open : écriture hors de work/",
    "patched-paths": "open : écriture hors de work/",
    "runpy": "import : module torch interdit",
    "safetensors-rust": "import : module safetensors._safetensors_rust interdit",
    "socket": "import : module socket interdit",
    # Sous Linux et macOS, os.spawnl passe par os.fork.
    "spawn": ("os.spawn : interdit", "os.fork : interdit", "os.posix_spawn : interdit"),
    "state-remove": "os.remove : écriture hors de work/",
    "state-write": "open : écriture hors de work/",
    "subprocess": "import : module subprocess interdit",
    "symlink": "os.symlink : interdit",
    "system": "os.system : interdit",
}


def test_every_forbidden_script_has_a_reason():
    assert sorted(REASONS) == sorted(REFUSED_SCRIPTS)


@pytest.mark.parametrize("name", sorted(REFUSED_SCRIPTS))
def test_forbidden_script_is_refused(ws, name):
    held = ws.parent / "heldout"
    held.mkdir(exist_ok=True)
    (held / "math.jsonl").write_text("{}\n", encoding="utf-8")
    code = REFUSED_SCRIPTS[name].replace("\n", "\n    ")
    # Une exception attrapée n'efface pas le refus : le crochet arrête le processus.
    script = f"try:\n    {code}\nexcept BaseException:\n    pass\nprint('passé')\n"
    (ws / "scripts" / "t.py").write_text(script, encoding="utf-8")
    before = snapshot(ws.parent)
    raw_journal = (session(ws) / "journal.jsonl").read_bytes()
    entry = script_runner.run_script("scripts/t.py", "s1", workspace=ws, timeout=60)
    assert entry["returncode"] == REFUSED, entry["output"]
    assert entry["refused"] is True
    reasons = REASONS[name] if isinstance(REASONS[name], tuple) else (REASONS[name],)
    assert any(f"run-script : refus ({r}" in entry["output"] for r in reasons), entry["output"]
    assert "passé" not in entry["output"]
    assert snapshot(ws.parent) == before
    # Journal : inchangé sauf l'événement script ajouté par le lanceur.
    lines = (session(ws) / "journal.jsonl").read_bytes()
    assert lines.startswith(raw_journal)
    assert [e["event"] for e in journal(ws)[-1:]] == ["script"]
    assert harness.read_state(session(ws)).used == 0


def test_environment_and_cwd_modules_do_not_reach_the_child(ws, monkeypatch):
    # Scénario 4 : PYTHONPATH vers un faux json et json.py dans l'espace.
    fake = ws / "scripts" / "fake"
    fake.mkdir()
    (fake / "json.py").write_text("raise SystemExit(42)", encoding="utf-8")
    (ws / "json.py").write_text("raise SystemExit(43)", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(fake))
    monkeypatch.setenv("PYTHONSTARTUP", str(fake / "json.py"))
    monkeypatch.setenv("PYTHONHOME", str(fake))
    assert "PYTHONPATH" not in script_runner.child_env()
    entry = run(ws, "env.py", "import json, os, sys\nprint(json.__file__, sys.flags.isolated)\n")
    assert entry["returncode"] == 0, entry["output"]
    assert "fake" not in entry["output"] and entry["output"].strip().endswith("1")
    assert journal(ws)[-1]["event"] == "script"


def test_script_outside_scripts_dir_refused(ws):
    (ws / "notes" / "x.py").write_text("print(1)", encoding="utf-8")
    for bad in ("notes/x.py", "scripts/../notes/x.py", "scripts/absent.py"):
        with pytest.raises(harness.HarnessError, match="script refusé"):
            script_runner.run_script(bad, "s1", workspace=ws)
    with pytest.raises(harness.HarnessError, match="--sessions-root refusé"):
        (ws / "scripts" / "ok.py").write_text("print(1)", encoding="utf-8")
        script_runner.run_script("scripts/ok.py", "s1", sessions_root=ws.parent, workspace=ws)


def test_frozen_or_locked_session_refuses_scripts(ws):
    (ws / "scripts" / "ok.py").write_text("print(1)", encoding="utf-8")
    (session(ws) / harness.LOCK_FILE).write_text(
        json.dumps({"pid": os.getpid(), "acquired_at": "2026-11-05T10:00:00+00:00"}),
        encoding="utf-8",
    )
    with pytest.raises(harness.HarnessError, match="un lot tourne déjà"):
        script_runner.run_script("scripts/ok.py", "s1", workspace=ws)
    (session(ws) / harness.LOCK_FILE).unlink()
    harness.update_state(session(ws), frozen=True)
    with pytest.raises(harness.HarnessError, match="gelée"):
        script_runner.run_script("scripts/ok.py", "s1", workspace=ws)


def test_output_is_bounded_and_timeout_kills(ws):
    entry = run(ws, "loud.py", "print('x' * 100000)")
    assert entry["returncode"] == 0
    assert len(entry["output"]) == script_runner.MAX_OUTPUT and entry["output_bytes"] > 100000
    (ws / "scripts" / "slow.py").write_text("import time\ntime.sleep(30)", encoding="utf-8")
    slow = script_runner.run_script("scripts/slow.py", "s1", workspace=ws, timeout=1)
    assert slow["timed_out"] is True and slow["returncode"] != 0


def test_cli_run_script(ws, monkeypatch, capsys):
    monkeypatch.chdir(ws)
    (ws / "scripts" / "ok.py").write_text("print('bonjour')", encoding="utf-8")
    args = ["run-script", "scripts/ok.py", "--session", "s1", "--sessions-root", "sessions"]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert "bonjour" in captured.out and "code 0" in captured.err
    (ws / "scripts" / "bad.py").write_text("import torch", encoding="utf-8")
    assert main([args[0], "scripts/bad.py", *args[2:]]) == REFUSED
    assert "refusé" in capsys.readouterr().err


def test_sandbox_needs_isolated_interpreter(ws):
    import subprocess

    cfg = {"script": "x", "write_roots": ["a", "b"]}
    proc = subprocess.run(
        [sys.executable, str(script_runner._BOOT), json.dumps(cfg)],
        capture_output=True,
        encoding="utf-8",
        env=script_runner.child_env(),
        cwd=ws,
    )
    assert proc.returncode == REFUSED and "non isolé" in proc.stderr


# Revue passe 1 : séance, racine et données imposées par workspace.json ; noms protégés.


def forge_session(ws: Path, root: Path, budget: int = 50) -> None:
    """Séance forgée par l'éditeur hors de sessions/ (dans notes/, écriture permise)."""
    real, fake = session(ws), root / "s1"
    (fake / "work").mkdir(parents=True)
    for f in (real / "work").iterdir():
        (fake / "work" / f.name).write_bytes(f.read_bytes())
    state = json.loads((real / "state.json").read_text("utf-8"))
    (fake / "state.json").write_text(json.dumps({**state, "budget": budget}), encoding="utf-8")
    (fake / "journal.jsonl").write_text('{"event": "open"}\n', encoding="utf-8")


@pytest.fixture
def fake_generate(monkeypatch):
    """Génération remplacée (frontière : le modèle réel n'est jamais chargé par les tests)."""
    calls = []

    def generate(work, problems, device):
        calls.append(work)
        return {"model": "test"}, [{"text": "2"} for _ in problems]

    monkeypatch.setattr(harness, "_generate", generate)
    return calls


def test_run_batch_is_bound_to_the_workspace(ws, monkeypatch, capsys, fake_generate):
    monkeypatch.chdir(ws)
    forge_session(ws, ws / "notes" / "fake")
    leak = ws.parent / "src-data" / "gsm8k-train.jsonl"
    common = ["--dataset", "gsm8k", "--split", "train"]
    data = ["--data", "data/gsm8k-train.jsonl"]
    for bad, reason in (
        (["--session", "s1", "--sessions-root", "notes/fake", *data], "--sessions-root refusé"),
        (["--session", "s1", "--sessions-root", str(ws.parent), *data], "--sessions-root refusé"),
        (["--session", "s2", *data], "séance refusée"),
        (["--session", "s1", "--data", str(leak)], "--data refusé"),
        (["--session", "s1", "--data", "data/../../src-data/gsm8k-train.jsonl"], "--data refusé"),
    ):
        assert main(["run-batch", *bad, *common]) == 2, bad
        assert reason in capsys.readouterr().err
    assert fake_generate == []
    ok = ["run-batch", "--session", "s1", *data, *common]
    assert main([*ok, "--sessions-root", "sessions"]) == 0
    assert main(ok) == 0
    assert len(fake_generate) == 2 and harness.read_state(session(ws)).used == 2
    assert harness.read_state(ws / "notes" / "fake" / "s1").used == 0


def test_session_status_is_bound_to_the_workspace(ws, monkeypatch, capsys):
    other = ws.parent / "other"
    other.mkdir()
    monkeypatch.chdir(ws / "scripts")  # sous-dossier : l'espace est trouvé plus haut
    assert main(["session-status", "--session", "s1", "--sessions-root", str(other)]) == 2
    assert "--sessions-root refusé" in capsys.readouterr().err
    assert main(["session-status", "--session", "s1"]) == 0
    assert "0/10 lots" in capsys.readouterr().out


def test_moved_workspace_file_is_refused(ws, monkeypatch, capsys):
    (ws / "notes" / "workspace.json").write_bytes((ws / "workspace.json").read_bytes())
    monkeypatch.chdir(ws / "notes")
    assert main(["session-status", "--session", "s1"]) == 2
    assert "ne désigne pas" in capsys.readouterr().err


@pytest.mark.parametrize(
    "target",
    [
        "fake/s1/state.json.",
        "fake/s1/STATE.JSON",
        "fake/s1/state.json ",
        "fake/s1/state.json::$DATA",
        "fake/workspace.json",
        "fake/x.py:flux",
    ],
)
def test_protected_name_variants_refused_in_scripts(ws, target):
    (ws / "scripts" / "fake" / "s1").mkdir(parents=True)
    before = snapshot(ws)
    entry = run(ws, "forge.py", f"open('scripts/{target}', 'w').write('{{}}')\n")
    assert entry["refused"], entry["output"]
    after = snapshot(ws)
    after.pop("scripts/forge.py")
    assert after == before
    assert not list((ws / "scripts" / "fake").rglob("*.*"))


def test_launcher_runs_the_verified_copy(ws):
    import hashlib

    (ws / "scripts" / "ok.py").write_text("print('original')", encoding="utf-8")
    sha = hashlib.sha256(b"print('original')").hexdigest()
    copies = session(ws) / "scripts"
    copies.mkdir(exist_ok=True)
    (copies / f"{sha}.py").write_text("print('altere')", encoding="utf-8")
    entry = script_runner.run_script("scripts/ok.py", "s1", workspace=ws, timeout=60)
    assert entry["output"].strip() == "original" and entry["sha256"] == sha
    assert (copies / f"{sha}.py").read_bytes() == b"print('original')"


def test_sandbox_executes_the_copy_not_the_mutable_file(ws):
    import hashlib
    import subprocess

    (ws / "scripts" / "x.py").write_text("print('mutable')", encoding="utf-8")
    copy = ws.parent / "copy.py"
    copy.write_text("print('copie')", encoding="utf-8")
    cfg = {
        "script": str(ws / "scripts" / "x.py"),
        "source": str(copy),
        "sha256": hashlib.sha256(b"print('copie')").hexdigest(),
        "workspace": str(ws),
        "read_roots": [str(ws / "scripts")],
        "read_files": [],
        "write_roots": [str(ws / "scripts"), str(ws / "scripts")],
        "python_roots": script_runner._python_roots(),
    }

    def boot(cfg):
        cmd = [sys.executable, "-I", "-B", "-X", "utf8", str(script_runner._BOOT), json.dumps(cfg)]
        env = script_runner.child_env()
        return subprocess.run(cmd, capture_output=True, encoding="utf-8", env=env, cwd=ws)

    proc = boot(cfg)
    assert proc.returncode == 0 and proc.stdout.strip() == "copie", proc.stderr
    proc = boot({**cfg, "sha256": "00" * 32})
    assert proc.returncode == REFUSED and "altérée" in proc.stderr


def test_settings_pin_repo_through_uv_project_env(ws):
    """Le dépôt n'est cité que dans `env.UV_PROJECT` (et workspace.json) : jamais dans une règle."""
    settings = json.loads((ws / ".claude" / "settings.json").read_text("utf-8"))
    repo = REPO.resolve().as_posix()
    assert settings["env"] == {"UV_PROJECT": repo}
    perms = settings["permissions"]
    assert not any(repo in rule for rule in [*perms["allow"], *perms["deny"]])
    assert not any("--project" in rule for rule in perms["allow"])


@pytest.mark.parametrize("tool", ["Bash", "PowerShell"])
def test_settings_deny_uv_redirection(ws, tool):
    deny = json.loads((ws / ".claude" / "settings.json").read_text("utf-8"))["permissions"]["deny"]
    for needle in ("UV_", "--project", "--directory"):
        assert f"{tool}(*{needle}*)" in deny, needle


# CAP-8 : threads plafonnés, délai par script, sortie dans un fichier.

THREAD_VARS = script_runner._THREAD_VARS


def make_ws(tmp_path, base, **kwargs) -> Path:
    data = tmp_path / "src-data" / "gsm8k-train.jsonl"
    data.parent.mkdir()
    data.write_text('{"split": "train", "question": "1+1"}\n', encoding="utf-8")
    return workspace.create_workspace(tmp_path / "ws", "s1", 10, base, [data], **kwargs)


def test_threads_default_four_in_child_before_numpy(ws):
    # threadpoolctl n'est pas une dépendance (et son import est refusé dans l'enfant) : la valeur
    # effective des bibliothèques natives vient de ces variables, lues au chargement de numpy.
    code = (
        "import os\nimport numpy as np\n"
        "print(sorted((k, v) for k, v in os.environ.items() if k.endswith('_NUM_THREADS')))\n"
        "print((np.ones((64, 64)) @ np.ones((64, 64))).sum())"
    )
    entry = run(ws, "t.py", code)
    assert entry["returncode"] == 0
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        assert f"('{var}', '4')" in entry["output"]
    assert "262144.0" in entry["output"]


def test_threads_env_value_comes_from_workspace_config(tmp_path, base):
    ws = make_ws(tmp_path, base, script_threads=2)
    assert json.loads((ws / "workspace.json").read_text("utf-8"))["script_threads"] == 2
    entry = run(ws, "t.py", "import os\nprint(os.environ['OPENBLAS_NUM_THREADS'])")
    assert entry["output"].strip() == "2"


def test_script_cannot_raise_threads(ws):
    code = (
        "import os\nos.environ['OPENBLAS_NUM_THREADS'] = '64'\n"
        "import numpy as np\nprint(os.environ['OPENBLAS_NUM_THREADS'])\n"
        "import threadpoolctl\n"
    )
    entry = run(ws, "t.py", code)
    assert entry["refused"] is True and "threadpoolctl" in entry["output"]
    code = 'import numpy\nc = numpy._core._internal.ctypes\nc.CDLL("x.dll").f(64)'
    entry = run(ws, "c.py", code)
    assert entry["refused"] is True, entry["output"]
    assert "ctypes" in entry["output"]
    for flag in ("--threads", "--script-threads"):
        with pytest.raises(SystemExit):
            main(["run-script", "scripts/t.py", "--session", "s1", flag, "64"])


def test_limits_defaults_and_creation(tmp_path, base):
    ws = make_ws(tmp_path, base)
    info = json.loads((ws / "workspace.json").read_text("utf-8"))
    assert (info["script_threads"], info["script_timeout"], info["script_max_timeout"]) == (
        4,
        900,
        3600,
    )
    consigne = (ws / "CONSIGNE.md").read_text("utf-8")
    assert "15 min" in consigne and "60 min" in consigne and "4 threads" in consigne
    assert "runs/<horodatage>-<script>.log" in consigne and "<SCRIPT_" not in consigne
    assert "<SCRIPT_" not in (ws / "CLAUDE.md").read_text("utf-8")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"script_threads": 0}, "--script-threads refusé"),
        ({"script_timeout": 0}, "--script-timeout refusé"),
        ({"script_max_timeout": -1}, "--script-max-timeout refusé"),
        ({"script_timeout": 7200}, "dépasse le délai maximum 3600 s"),
        ({"script_max_timeout": float("inf")}, "--script-max-timeout refusé"),
        ({"script_max_timeout": 1e300}, "au plus 86400 s"),
        ({"script_timeout": float("nan")}, "--script-timeout refusé"),
        ({"script_threads": 2.5}, "--script-threads refusé"),
    ],
)
def test_invalid_limits_refused_at_creation(tmp_path, base, kwargs, message):
    with pytest.raises(harness.HarnessError, match=message):
        make_ws(tmp_path, base, **kwargs)
    assert not (tmp_path / "ws").exists()


def test_timeout_default_and_max(tmp_path, base):
    ws = make_ws(tmp_path, base, script_timeout=1, script_max_timeout=5)
    (ws / "scripts" / "slow.py").write_text("import time\ntime.sleep(30)", encoding="utf-8")
    slow = script_runner.run_script("scripts/slow.py", "s1", workspace=ws)
    assert slow["timed_out"] is True and slow["timeout"] == 1
    assert journal(ws)[-1]["timed_out"] is True
    n = len(journal(ws))
    with pytest.raises(harness.HarnessError, match=r"délai maximum.*\(5 s\).*au plus 5 s"):
        script_runner.run_script("scripts/slow.py", "s1", workspace=ws, timeout=6)
    for bad in (0, -3):
        with pytest.raises(harness.HarnessError, match="strictement positif"):
            script_runner.run_script("scripts/slow.py", "s1", workspace=ws, timeout=bad)
    assert len(journal(ws)) == n  # refus sans exécution ni journal


def test_default_limits_for_workspace_without_limits_keys(ws):
    info = json.loads((ws / "workspace.json").read_text("utf-8"))
    for k in ("script_threads", "script_timeout", "script_max_timeout"):
        del info[k]
    (ws / "workspace.json").write_text(json.dumps(info), encoding="utf-8")
    assert script_runner.script_limits(harness.find_workspace(ws)[1]) == (4, 900.0, 3600.0)


@pytest.mark.parametrize("bad", [float("inf"), "x", 0])
def test_damaged_limits_in_workspace_refused(ws, bad):
    # Un délai infini relu tel quel ferait lever OverflowError au Timer : script sans délai.
    info = json.loads((ws / "workspace.json").read_text("utf-8"))
    info["script_max_timeout"] = bad
    (ws / "workspace.json").write_text(json.dumps(info), encoding="utf-8")
    (ws / "scripts" / "t.py").write_text("print(1)", encoding="utf-8")
    with pytest.raises(harness.HarnessError):
        script_runner.run_script("scripts/t.py", "s1", workspace=ws)


def test_timeout_kills_process_tree():
    # Le petit-enfant hérite du tube : sans arrêt de l'arbre, la lecture attend 60 s.
    import subprocess

    grandchild = "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import subprocess,sys,time;{grandchild};print('ok',flush=True);time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        start_new_session=sys.platform != "win32",
    )
    assert proc.stdout.readline().strip() == b"ok"
    script_runner._kill_tree(proc)
    done = threading.Event()
    threading.Thread(target=lambda: (proc.stdout.read(), done.set()), daemon=True).start()
    assert done.wait(15), "un descendant tient encore le tube"
    assert proc.wait(timeout=10) != 0
    proc.stdout.close()


_BLAS_PROBE = """
import ctypes, glob, os, sys
import numpy as np
os.environ.update({k: "64" for k in %r})
root = os.path.dirname(np.__file__)
libs = glob.glob(os.path.join(root, os.pardir, "numpy.libs", "*openblas*"))
libs += glob.glob(os.path.join(root, ".dylibs", "*openblas*"))
for f in libs:
    fn = getattr(ctypes.CDLL(f), "openblas_get_num_threads", None)
    if fn is not None:
        print(fn())
        break
else:
    print("absent")
"""


def test_blas_threads_measured_equal_cap_even_if_env_changed():
    # Mesure effective (hors bac à sable, où ctypes est refusé) : l'OpenBLAS de numpy, chargé
    # avec child_env(), garde le plafond même si le script change ensuite os.environ.
    import subprocess

    code = _BLAS_PROBE % (THREAD_VARS,)
    for cap in (script_runner.DEFAULT_THREADS, 2):
        out = subprocess.run(
            [sys.executable, "-I", "-c", code],
            env=script_runner.child_env(cap),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if out == "absent":
            pytest.skip("numpy sans OpenBLAS embarqué")
        assert out == str(cap)


def test_full_output_goes_to_log_file(ws):
    import hashlib

    entry = run(ws, "loud.py", "import sys\nprint('x' * 100000)\nprint('err', file=sys.stderr)")
    log = ws / entry["log"]
    assert re.fullmatch(r"runs/\d{8}T\d{12}Z-loud\.log", entry["log"])
    data = log.read_bytes()
    assert b"x" * 100000 in data and b"err" in data
    assert len(data) == entry["output_bytes"] == entry["log_bytes"] > script_runner.MAX_OUTPUT
    assert entry["log_sha256"] == hashlib.sha256(data).hexdigest()
    assert entry["log_truncated"] is False
    assert len(entry["output"]) == script_runner.MAX_OUTPUT
    assert journal(ws)[-1]["log_sha256"] == entry["log_sha256"]
    # Deux exécutions : deux fichiers, aucun écrasé.
    second = run(ws, "loud.py", "print('y')")
    assert second["log"] != entry["log"] and log.read_bytes() == data


def test_log_file_is_capped(ws):
    (ws / "scripts" / "big.py").write_text("print('z' * 50000)", encoding="utf-8")
    entry = script_runner.run_script("scripts/big.py", "s1", workspace=ws, max_log_bytes=1000)
    assert entry["log_bytes"] == 1000 and entry["log_truncated"] is True
    assert entry["output_bytes"] > 50000 and (ws / entry["log"]).stat().st_size == 1000
    data = (ws / entry["log"]).read_bytes()
    assert entry["log_sha256"] == hashlib.sha256(data).hexdigest()


def test_popen_failure_leaves_no_log(ws, monkeypatch):
    (ws / "scripts" / "t.py").write_text("print(1)", encoding="utf-8")

    def boom(*a, **k):
        raise OSError("lancement impossible")

    monkeypatch.setattr(script_runner.subprocess, "Popen", boom)
    with pytest.raises(OSError):
        script_runner.run_script("scripts/t.py", "s1", workspace=ws)
    assert not list((ws / "runs").glob("*.log"))


def test_interrupted_read_kills_script(ws, monkeypatch):
    # Ctrl-C pendant la lecture : le délai est annulé, l'arbre doit être arrêté quand même.
    (ws / "scripts" / "t.py").write_text("import time; time.sleep(60)", encoding="utf-8")
    seen = []

    def interrupted(stream, *a, **k):
        raise KeyboardInterrupt

    real_kill = script_runner._kill_tree
    monkeypatch.setattr(script_runner, "_bounded_read", interrupted)
    monkeypatch.setattr(
        script_runner, "_kill_tree", lambda proc: (seen.append(proc), real_kill(proc))
    )
    with pytest.raises(KeyboardInterrupt):
        script_runner.run_script("scripts/t.py", "s1", workspace=ws)
    assert len(seen) == 1 and seen[0].poll() is not None


def test_runs_dir_not_writable_by_editor(ws):
    deny = json.loads((ws / ".claude" / "settings.json").read_text("utf-8"))["permissions"]["deny"]
    assert "Edit(./runs/**)" in deny
    entry = run(ws, "w.py", "open('runs/x.log', 'w')")
    assert entry["refused"] is True and not (ws / "runs" / "x.log").exists()


def test_cli_prints_log_path(ws, monkeypatch, capsys):
    monkeypatch.chdir(ws)
    (ws / "scripts" / "ok.py").write_text("print('a' * 30000)", encoding="utf-8")
    args = ["run-script", "scripts/ok.py", "--session", "s1", "--sessions-root", "sessions"]
    assert main(args + ["--timeout", "99999"]) != 0
    assert "délai maximum" in capsys.readouterr().err
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "sortie complète dans runs/" in out
