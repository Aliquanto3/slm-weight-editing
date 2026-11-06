"""Story 2.5 : lecture libre de l'espace (CAP-5), OPERATEUR.md hors de l'espace (CAP-6), refus
expliqués par le crochet PreToolUse, notes/BLOCAGES.md, décodage autorégressif interdit."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import to_bf16, write_safetensors
from test_audit_transcript import CLEAN_CALLS, CTX_S1, SCENARIOS, VIOLATIONS, WS

from slm_weight_editing import _refusal_hook as hook
from slm_weight_editing import _sandbox, freeze, harness, main, script_runner, workspace
from slm_weight_editing import audit_transcript as audit

REPO = Path(__file__).resolve().parents[1]
REFUSED = _sandbox.REFUSED
TENSOR = "model.layers.0.mlp.up_proj.weight"


@pytest.fixture(scope="module")
def base(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("base")
    w = to_bf16(np.linspace(-1, 1, 64, dtype=np.float32).reshape(8, 8))
    write_safetensors(d / "model.safetensors", {TENSOR: w})
    (d / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    return d


def _data(tmp_path: Path) -> Path:
    data = tmp_path / "src-data" / "gsm8k-train.jsonl"
    data.parent.mkdir(exist_ok=True)
    data.write_text('{"split": "train", "question": "1+1"}\n', encoding="utf-8")
    return data


@pytest.fixture
def ws(tmp_path, base) -> Path:
    return workspace.create_workspace(
        tmp_path / "ws", "s1", 10, base, [_data(tmp_path)], base_revision="r0"
    )


def run(ws: Path, code: str) -> dict:
    (ws / "scripts" / "t.py").write_text(code, encoding="utf-8")
    return script_runner.run_script("scripts/t.py", "s1", workspace=ws, timeout=60)


def make_link(link: Path, target: Path) -> bool:
    """Jonction NTFS (`mklink /J`) sous Windows, lien symbolique ailleurs ; False si impossible."""
    try:
        if sys.platform == "win32":
            done = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True,
                check=False,
            )
            return done.returncode == 0 and link.exists()
        os.symlink(target, link, target_is_directory=target.is_dir())
        return True
    except OSError:
        return False


def payload(name: str, tool_input: dict, cwd: str | Path) -> dict:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": name,
        "tool_input": tool_input,
        "cwd": str(cwd),
    }


def hook_command(ws: Path) -> list[str]:
    settings = json.loads((ws / ".claude" / "settings.json").read_text(encoding="utf-8"))
    entry = settings["hooks"]["PreToolUse"][0]["hooks"][0]
    return [entry["command"], *entry["args"]]


def call_hook(ws: Path, data: bytes) -> subprocess.CompletedProcess:
    return subprocess.run(
        hook_command(ws), input=data, capture_output=True, cwd=ws, timeout=60, check=False
    )


# Critère 1 : un script lit tout l'espace, rien au-delà.


READ_ALL = """import mmap, os
for p in ("notes/BLOCAGES.md", "EDITION.md", "CONSIGNE.md", "CLAUDE.md", "workspace.json",
          ".claude/settings.json", "sessions/s1/journal.jsonl", "sessions/s1/state.json",
          "probes/0/manifest.json", "runs/x.log", "data/gsm8k-train.jsonl"):
    with open(p, "rb") as f:
        assert f.read(), p
for d in (".", "notes", "sessions", "sessions/s1", "probes", "runs", ".claude"):
    os.listdir(d)
with open("probes/0/manifest.json", "rb") as f:
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    assert m[:1] == b"{"
    m.close()
print("lu")
"""


def test_script_reads_the_whole_workspace(ws):
    (ws / "EDITION.md").write_text("édition\n", encoding="utf-8")
    (ws / "probes" / "0").mkdir(parents=True)
    (ws / "probes" / "0" / "manifest.json").write_text("{}", encoding="utf-8")
    (ws / "runs").mkdir(exist_ok=True)
    (ws / "runs" / "x.log").write_text("sortie\n", encoding="utf-8")
    entry = run(ws, READ_ALL)
    assert entry["returncode"] == 0, entry["output"]
    assert "lu" in entry["output"]


def _outside_targets(ws: Path) -> dict[str, str]:
    held = ws.parent / "heldout"
    held.mkdir(exist_ok=True)
    (held / "math.jsonl").write_text("{}\n", encoding="utf-8")
    return {
        "depot": f'open(r"{REPO / "pyproject.toml"}").read()',
        "heldout": f'open(r"{held / "math.jsonl"}").read()',
        "parent": 'open("../heldout/math.jsonl").read()',
        "parent-listdir": 'import os\nos.listdir("..")',
        "operateur": f'open(r"{workspace.operator_dir(REPO) / "s1.md"}").read()',
    }


@pytest.mark.parametrize("target", ["depot", "heldout", "parent", "parent-listdir", "operateur"])
def test_script_reading_outside_the_workspace_is_refused(ws, target):
    entry = run(ws, _outside_targets(ws)[target])
    assert entry["returncode"] == REFUSED, entry["output"]
    assert "lecture hors des dossiers permis" in entry["output"]
    assert (
        "règle lecture-hors-espace, permis : tout l'espace de séance est lisible"
        in (entry["output"])
    )


def test_script_reading_through_a_link_to_the_repo_is_refused(ws):
    if not make_link(ws / "notes" / "lien", REPO):
        pytest.skip("jonction ou lien symbolique non créable sur ce poste")
    entry = run(ws, 'open("notes/lien/pyproject.toml").read()')
    assert entry["returncode"] == REFUSED, entry["output"]
    assert "règle lecture-hors-espace" in entry["output"]


@pytest.mark.skipif(sys.platform != "win32", reason="préfixes de chemin Windows")
@pytest.mark.parametrize("form", ["device", "unc"])
def test_script_reading_repo_through_windows_prefixes_is_refused(ws, form):
    target = REPO / "pyproject.toml"
    if form == "device":
        path = "\\\\?\\" + str(target)
    else:
        path = "\\\\localhost\\" + str(target)[0].lower() + "$" + str(target)[2:]
    entry = run(ws, f"open(r{path!r}).read()")
    assert entry["returncode"] == REFUSED, entry["output"]
    assert "lecture hors des dossiers permis" in entry["output"]


def test_script_writing_protected_name_variant_is_refused(ws):
    entry = run(ws, 'open("scripts/Workspace.JSON.", "w").write("{}")')
    assert entry["returncode"] == REFUSED, entry["output"]
    assert "fichier de séance protégé" in entry["output"]
    assert "règle etat-seance, permis : état par session-status" in entry["output"]


# Critère 2 : OPERATEUR.md hors de l'espace.


def test_operator_file_is_written_outside_the_workspace(ws, base):
    path = workspace.operator_dir(REPO) / "s1.md"
    text = path.read_text(encoding="utf-8")
    assert not workspace._inside(path, ws)
    assert not [p for p in ws.rglob("*") if "operateur" in p.name.lower()]
    assert not [p for p in ws.rglob("*.md") if p.read_text(encoding="utf-8") == text]
    sessions = str(ws / "sessions")
    for command in (
        f"session-freeze --session s1 --sessions-root {sessions} --original {base.resolve()}",
        f"audit-session <transcription>.jsonl --workspace {ws}",
        f"--session-dir {sessions}/s1 --reference-from {base.resolve()} --base-revision r0",
        f"session-unlock --session s1 --sessions-root {sessions}",
        "apply-diff --diff",
        "report-gguf --checkpoint models/rejoue/s1",
        f"cd {ws}\n   {workspace.LAUNCH_COMMAND}",
    ):
        assert command in text, command
    for check in ("/hooks", "/memory", "/mcp", "/plugin", "Aucun mod"):
        assert check in text, check
    for marker in ("<ID_SEANCE>", "<ESPACE>", "<SEANCES>", "<BASE>", "<REVISION>", "<LANCEMENT>"):
        assert marker not in text, marker


def test_launch_command_has_every_option():
    launch = workspace.LAUNCH_COMMAND.split()
    assert launch[0] == "claude"
    assert "--strict-mcp-config" in launch and "--disable-slash-commands" in launch
    assert launch[launch.index("--setting-sources") + 1] == "project"
    tools = launch[launch.index("--tools") + 1].split(",")
    want = ["Read", "Grep", "Glob", "Edit", "Write", "Bash", "PowerShell", "TodoWrite"]
    assert tools == want
    assert "--bare" not in launch and "--safe-mode" not in launch


def test_cli_prints_operator_path_and_launch(tmp_path, base, capsys):
    args = ["session-workspace", "--out", str(tmp_path / "ws"), "--session", "s9"]
    args += ["--budget", "10", "--base-checkpoint", str(base), "--data", str(_data(tmp_path))]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert f"{workspace.operator_dir(REPO) / 's9'}.md" in out
    assert workspace.LAUNCH_COMMAND in out and "/hooks" in out


def test_operator_file_is_never_overwritten(tmp_path, base, ws):
    path = workspace.operator_dir(REPO) / "s1.md"
    before = path.read_bytes()
    with pytest.raises(workspace.WorkspaceError, match="instructions déjà écrites"):
        workspace.create_workspace(tmp_path / "ws2", "s1", 10, base, [_data(tmp_path)])
    assert path.read_bytes() == before
    assert not (tmp_path / "ws2").exists()


def test_default_operator_dir_is_under_models(monkeypatch):
    monkeypatch.setattr(workspace, "OPERATOR_DIR", None)
    assert workspace.operator_dir(REPO) == (REPO / "models" / "operateur").resolve()


def test_operator_read_is_denied_by_settings_audit_and_hook(ws):
    settings = json.loads((ws / ".claude" / "settings.json").read_text(encoding="utf-8"))
    rule = f"Read({workspace.rule_path(workspace.operator_dir(REPO))}/**)"
    assert rule in settings["permissions"]["deny"]
    assert rule.startswith("Read(//")
    path = str(workspace.operator_dir(REPO) / "s1.md")
    ctx = hook.context(ws)
    found = audit.check_tool_use("Read", {"file_path": path}, ctx)
    assert {"lecture-hors-espace", "lecture-interdite"} <= {r for r, _ in found}
    out = hook.decide(payload("Read", {"file_path": path}, ws), ctx)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize(
    ("path", "want"),
    [
        (Path("C:/Users/Anael/depot/models/operateur"), "//c/Users/Anael/depot/models/operateur"),
        (Path("D:/x"), "//d/x"),
    ],
)
@pytest.mark.skipif(sys.platform != "win32", reason="lecteurs Windows")
def test_rule_path_windows(path, want):
    assert workspace.rule_path(path) == want


@pytest.mark.skipif(sys.platform == "win32", reason="chemins POSIX")
def test_rule_path_posix():
    # Chemin résolu (macOS : /home -> /System/Volumes/Data/home), préfixé de « / ».
    expected = "/" + Path("/home/anael/depot").resolve().as_posix()
    assert workspace.rule_path(Path("/home/anael/depot")) == expected


# Critères 3 et 4 : crochet de refus, familles de refus.


def test_allowed_ways_cover_every_rule():
    assert set(audit.ALLOWED_WAYS) == set(audit.RULES)
    assert set(_sandbox.WAYS) <= set(audit.RULES)
    for rule in ("commande-hors-liste", "python-hors-lanceur", "modele-hors-harnais"):
        assert any(c in audit.ALLOWED_WAYS[rule] for c in ("run-batch", "run-script", "probe"))
    assert audit.refusal_message("web", "outil WebFetch") == (
        "web : outil WebFetch ; permis : aucun accès réseau ; note le besoin dans notes/BLOCAGES.md"
    )


FAMILIES = [
    ("Bash", {"command": "cat CONSIGNE.md"}, "commande-hors-liste", "seules run-batch"),
    ("Bash", {"command": "python -c 1"}, "python-hors-lanceur", "Python par run-script"),
    ("Bash", {"command": "cd .."}, "hors-espace-seance", "reste dans l'espace"),
    ("Bash", {"command": "X=1 uv run slm-weight-editing session-status"}, "environnement", ""),
    ("Read", {"file_path": "../x.txt"}, "lecture-hors-espace", "tout l'espace de séance"),
    ("Write", {"file_path": "x.py", "content": "1"}, "ecriture-hors-liste", "scripts/, notes/"),
    ("Write", {"file_path": "scripts/state.json", "content": "{}"}, "etat-seance", "seul work/"),
    ("WebFetch", {"url": "https://example.org"}, "web", "notes/BLOCAGES.md"),
    ("Agent", {"prompt": "x"}, "sous-agent", "aucun sous-agent"),
    ("Skill", {"skill": "x"}, "outil-non-autorise", "notes/BLOCAGES.md"),
    ("mcp__x__y", {}, "web", "aucun accès réseau"),
]


@pytest.mark.parametrize(("name", "tool_input", "rule", "way"), FAMILIES)
def test_hook_family_message_names_rule_and_way(ws, name, tool_input, rule, way):
    out = hook.decide(payload(name, tool_input, ws), hook.context(ws))
    decision = out["hookSpecificOutput"]
    assert decision["hookEventName"] == "PreToolUse"
    assert decision["permissionDecision"] == "deny"
    text = decision["permissionDecisionReason"]
    assert f"{rule} : " in text and f"permis : {audit.ALLOWED_WAYS[rule]}" in text
    assert way in text
    assert text.endswith(hook.BLOCAGES_HINT)


def test_launcher_import_family_message(ws):
    entry = run(ws, "import torch")
    assert entry["returncode"] == REFUSED
    assert (
        "règle import-interdit, permis : numpy, safetensors.numpy, bibliothèque standard ; "
        "modèle par run-batch ou probe" in entry["output"]
    )


def test_harness_family_message(ws, monkeypatch, capsys):
    monkeypatch.chdir(ws)
    assert main(["session-status", "--session", "autre"]) == 2
    err = capsys.readouterr().err
    assert "Erreur : séance refusée" in err and harness.EDITOR_WAY in err
    assert main(["run-script", "../x.py", "--session", "s1"]) != 0
    assert harness.EDITOR_WAY in capsys.readouterr().err


@pytest.mark.parametrize(
    ("name", "tool_input"),
    [
        ("Bash", {"command": "cat x"}),
        ("Bash", {"command": "python -c 1"}),
        ("Read", {"file_path": "C:/Windows/win.ini" if os.name == "nt" else "/etc/passwd"}),
        ("WebFetch", {"url": "https://example.org", "prompt": "x"}),
    ],
)
def test_hook_process_denies_with_reason(ws, name, tool_input):
    done = call_hook(ws, json.dumps(payload(name, tool_input, ws)).encode("utf-8"))
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert " ; permis : " in out["hookSpecificOutput"]["permissionDecisionReason"]


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"{pas du json",
        b"[1, 2]",
        "\u00e9".encode("latin-1"),
        json.dumps({"tool_name": "Read", "tool_input": {"file_path": "CONSIGNE.md"}}).encode(),
    ],
)
def test_hook_process_permitted_or_malformed_is_silent(ws, data):
    done = call_hook(ws, data)
    assert (done.returncode, done.stdout) == (0, b""), done.stderr


def test_hook_process_without_workspace_file_is_silent(ws):
    (ws / "workspace.json").unlink()
    done = call_hook(ws, json.dumps(payload("Bash", {"command": "cat x"}, ws)).encode())
    assert (done.returncode, done.stdout) == (0, b"")


def test_settings_hook_block(ws):
    settings = json.loads((ws / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert settings["hooks"] == {
        "PreToolUse": [
            {
                "matcher": "*",
                "hooks": [
                    {
                        "type": "command",
                        "command": workspace.hook_python(REPO).as_posix(),
                        "args": [
                            "-I",
                            "-B",
                            workspace.posix(REPO / workspace.HOOK_SCRIPT),
                            workspace.posix(ws),
                        ],
                        "timeout": 10,
                    }
                ],
            }
        ]
    }


def test_hook_python_falls_back_to_running_venv(tmp_path, monkeypatch):
    """Sans `.venv` dans le dépôt (CI) : environnement virtuel courant, chemin non résolu."""
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    rel = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
    exe = tmp_path.joinpath("venv", *rel)
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    assert workspace.hook_python(tmp_path / "depot") == exe.absolute()
    local = tmp_path.joinpath("depot", ".venv", *rel)
    local.parent.mkdir(parents=True)
    local.write_bytes(b"")
    assert workspace.hook_python(tmp_path / "depot") == local.absolute()


def test_workspace_refused_without_hook_interpreter(tmp_path, base, monkeypatch):
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    repo = tmp_path / "a" / "GitHub" / "depot"
    (repo / "src" / "slm_weight_editing").mkdir(parents=True)
    with pytest.raises(workspace.WorkspaceError, match="interpréteur du crochet introuvable"):
        workspace.create_workspace(tmp_path / "ws", "s1", 10, base, [_data(tmp_path)], repo=repo)
    assert not (tmp_path / "ws").exists()


def _permitted_calls(ws: Path) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []
    tail = {
        "run-batch": "--session s1 --sessions-root sessions --dataset gsm8k --split train "
        "--data data/gsm8k-train.jsonl --thinking on",
        "session-status": "--session s1 --sessions-root sessions",
        "run-script": "scripts/edit.py --session s1 --sessions-root sessions --timeout 600",
        "probe": "--session s1 --sessions-root sessions --texts notes/textes.jsonl "
        "--capture resid --positions all",
        "detach-lm-head": "--session s1 --sessions-root sessions",
    }
    for prefix in workspace.allowed_commands(REPO):
        sub = prefix.rsplit(" ", 1)[1]
        for tool in ("Bash", "PowerShell"):
            calls.append((tool, {"command": f"{prefix} {tail[sub]}"}))
    for p in sorted(ws.rglob("*")):
        if p.is_file():
            calls.append(("Read", {"file_path": str(p)}))
            calls.append(("Grep", {"pattern": "x", "path": str(p)}))
    calls.append(("Glob", {"pattern": "**/*", "path": str(ws)}))
    for rel in ("scripts/edit.py", "notes/BLOCAGES.md", "EDITION.md"):
        calls.append(("Write", {"file_path": rel, "content": "x = 1\n"}))
        calls.append(("Edit", {"file_path": str(ws / rel), "old_string": "a", "new_string": "b"}))
    calls.append(("Edit", {"file_path": "sessions/s1/work/model.safetensors", "new_string": "b"}))
    return calls


def test_hook_permits_the_closed_list(ws):
    ctx = hook.context(ws)
    calls = _permitted_calls(ws)
    for rel in (".claude/settings.json", "workspace.json", "sessions/s1/journal.jsonl"):
        assert ("Read", {"file_path": str(ws / rel)}) in calls, rel
    for name, tool_input in calls:
        assert hook.decide(payload(name, tool_input, ws), ctx) is None, (name, tool_input)


def test_hook_permits_cuda_commands_on_a_cuda_workspace(ws):
    ctx = dataclasses.replace(hook.context(ws), cuda=True)
    for prefix in workspace.allowed_commands(REPO, cuda=True):
        for tool in ("Bash", "PowerShell"):
            command = f"{prefix} --session s1 --sessions-root sessions"
            assert hook.decide(payload(tool, {"command": command}, ws), ctx) is None, command


@pytest.mark.parametrize(
    ("name", "tool_input"),
    [(n, i) for n, i, _ in VIOLATIONS + SCENARIOS] + [(n, i) for n, i in CLEAN_CALLS],
)
def test_hook_denies_iff_audit_flags(name, tool_input):
    # Amendement 2 : le crochet refuse si et seulement si l'audit signale une violation.
    flagged = audit.check_tool_use(name, tool_input, CTX_S1)
    out = hook.decide(payload(name, tool_input, WS), CTX_S1)
    assert (out is not None) == bool(flagged), (flagged, out)
    if out is not None:
        decision = out["hookSpecificOutput"]
        assert decision["permissionDecision"] == "deny"
        for rule, _ in flagged:
            assert f"{rule} : " in decision["permissionDecisionReason"]


def test_hook_never_allows(ws):
    ctx = hook.context(ws)
    calls = [(n, i) for n, i, _ in VIOLATIONS + SCENARIOS] + _permitted_calls(ws)
    for name, tool_input in calls:
        out = hook.decide(payload(name, tool_input, ws), ctx)
        assert out is None or out["hookSpecificOutput"]["permissionDecision"] == "deny"


# Amendements 4, 11 et 12 : chemins canoniques, casse, préfixes Windows.


def test_hook_denies_read_through_a_junction_to_the_repo(ws):
    if not make_link(ws / "notes" / "lien", REPO):
        pytest.skip("jonction ou lien symbolique non créable sur ce poste")
    ctx = hook.context(ws)
    for path in ("notes/lien/pyproject.toml", str(ws / "notes" / "lien" / "pyproject.toml")):
        out = hook.decide(payload("Read", {"file_path": path}, ws), ctx)
        assert out is not None, path
        assert "lecture-hors-espace" in out["hookSpecificOutput"]["permissionDecisionReason"]
    assert [v.rule for v in audit.audit_workspace_links(ws)] == ["lien"]


@pytest.mark.parametrize(
    "path",
    [
        "\\\\?\\C:\\depot\\pyproject.toml",
        "\\\\.\\C:\\depot\\pyproject.toml",
        "\\\\localhost\\c$\\depot\\pyproject.toml",
        "C:/PROGRA~1/x.txt",
        "notes/a.txt:flux",
        "C:x.txt",
        "HELDOUT/x.jsonl",
        "notes/OPERATEUR.md",
    ],
)
def test_hook_denies_indeterminate_or_forbidden_paths(ws, path):
    out = hook.decide(payload("Read", {"file_path": path}, ws), hook.context(ws))
    assert out is not None, path
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_hook_root_comes_from_argument_and_relative_paths_from_cwd(ws):
    ctx = hook.context(ws)
    # Un dossier courant hors de l'espace est refusé, même pour une lecture relative.
    out = hook.decide(payload("Read", {"file_path": "x.txt"}, ws.parent), ctx)
    assert "hors-espace-seance" in out["hookSpecificOutput"]["permissionDecisionReason"]
    assert hook.decide(payload("Read", {"file_path": "CONSIGNE.md"}, ws), ctx) is None


def _short_name(path: Path) -> str | None:
    """Nom court 8.3 de `path` (Windows), None s'il n'en a pas."""
    if sys.platform != "win32":
        return None
    import ctypes

    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, len(buf))
    short = buf.value if 0 < n < len(buf) else None
    return short if short and "~" in short else None


def test_hook_accepts_short_name_cwd_of_the_workspace(ws):
    # Le cwd vient de Claude Code (donnée de confiance) : un nom court 8.3 de l'espace n'est pas
    # un dossier hors de l'espace (l'audit des transcriptions l'accepte aussi).
    long_dir = ws / "un-dossier-au-nom-long"
    long_dir.mkdir()
    short = _short_name(long_dir)
    if short is None:
        pytest.skip("pas de nom court 8.3 sur ce volume")
    (long_dir / "workspace.json").write_text((ws / "workspace.json").read_text("utf-8"), "utf-8")
    ctx = hook.context(long_dir)
    assert hook.decide(payload("Read", {"file_path": "workspace.json"}, short), ctx) is None
    out = hook.decide(payload("Read", {"file_path": "../CONSIGNE.md"}, short), ctx)
    assert "lecture-hors-espace" in out["hookSpecificOutput"]["permissionDecisionReason"]
    assert hook.decide(payload("Read", {"file_path": "x"}, "sous-dossier"), ctx) is not None


def test_relative_path_resolved_on_received_cwd():
    ctx = dataclasses.replace(CTX_S1, cwd=str(WS / "notes"))
    assert audit._absolute("a.txt", ctx) == f"{CTX_S1.workspace}/notes/a.txt"
    assert audit._absolute("../../x", ctx) == audit._norm_path(str(WS.parent / "x"))


# Amendement 1 : l'audit accepte la lecture libre, refuse toujours l'écriture de l'état.


@pytest.mark.parametrize(
    "code",
    [
        'with open("sessions/s1/journal.jsonl") as f:\n    print(len(f.readlines()))',
        'open("workspace.json").read()',
        'open("sessions/s1/state.json", "rb").read()',
        'open("sessions/s1/state.json", mode="r").read()',
        'import pathlib\npathlib.Path("sessions/s1/session.lock").open().read()',
        'import os\nos.open("sessions/s1/state.json", os.O_RDONLY)',
    ],
)
def test_reading_protected_names_is_not_flagged(code):
    found = audit.check_code_content(code, "run-script")
    assert "etat-seance" not in {r for r, _ in found}, found


@pytest.mark.parametrize(
    "code",
    [
        'open("sessions/s1/journal.jsonl", "a").write("{}")',
        'open("sessions/s1/state.json", "w")',
        'open("sessions/s1/state.json", "x")',
        'open("sessions/s1/state.json", "r+")',
        'open("workspace.json", mode="wb")',
        'm = "w"\nopen("sessions/s1/state.json", m)',
        'import pathlib\npathlib.Path("sessions/s1/state.json").open("w")',
        'import pathlib\npathlib.Path("workspace.json").write_text("{}")',
        'import json\njson.dump({}, f)\np = "sessions/s1/state.json"',
        'import os\nos.open("sessions/s1/state.json", os.O_WRONLY)',
    ],
)
def test_writing_protected_names_stays_flagged(code):
    found = audit.check_code_content(code, "run-script")
    assert "etat-seance" in {r for r, _ in found}, found


def test_reading_claude_dir_inside_workspace_only():
    inside = audit.check_tool_use("Read", {"file_path": f"{WS}/.claude/settings.json"}, CTX_S1)
    assert inside == []
    assert audit.check_tool_use("Glob", {"pattern": ".claude/*"}, CTX_S1) == []
    outside = audit.check_tool_use("Read", {"file_path": f"{REPO}/.claude/settings.json"}, CTX_S1)
    assert {"lecture-hors-espace", "lecture-interdite"} <= {r for r, _ in outside}
    # Sans espace (audit d'appoint), `.claude/` reste un chemin interdit.
    bare = audit.check_tool_use("Read", {"file_path": ".claude/settings.json"})
    assert "lecture-interdite" in {r for r, _ in bare}


def test_operator_term_is_forbidden():
    assert audit.forbidden_reference("models/operateur/s1.md") == "terme interdit « operateur »"
    assert audit.forbidden_reference("OPERATEUR.md") == "terme interdit « operateur »"


# Amendement 10 : aucun lien dans l'espace.


def test_fresh_workspace_has_no_link(ws):
    assert audit.audit_workspace_links(ws) == []


def test_hard_link_in_workspace_is_flagged(ws):
    try:
        os.link(ws / "CONSIGNE.md", ws / "notes" / "copie.md")
    except OSError:
        pytest.skip("lien physique non créable sur ce système de fichiers")
    found = audit.audit_workspace_links(ws)
    assert {v.rule for v in found} == {"lien"}
    assert {v.detail.split(" (")[0] for v in found} == {
        "lien physique dans l'espace CONSIGNE.md",
        "lien physique dans l'espace notes/copie.md",
    }


def test_audit_session_cli_reports_workspace_links(ws, tmp_path, capsys):
    if not make_link(ws / "scripts" / "lien", REPO):
        pytest.skip("jonction ou lien symbolique non créable sur ce poste")
    t = tmp_path / "t.jsonl"
    t.write_text("", encoding="utf-8")
    args = ["audit-session", str(t), "--workspace", str(ws), "--session-dir"]
    args += [str(ws / "sessions" / "s1"), "--reference-from", str(ws / "sessions" / "s1" / "work")]
    assert main(args) == 1
    assert "lien · - · lien dans l'espace scripts/lien" in capsys.readouterr().out


# Critères 5 et 6 : BLOCAGES.md, décodage autorégressif.


def test_blocages_file_exists_and_is_writable(ws):
    text = (ws / "notes" / "BLOCAGES.md").read_text(encoding="utf-8")
    assert text == workspace.BLOCAGES_HEADER and text.startswith("# Blocages de la séance\n")
    settings = json.loads((ws / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert "Edit(./notes/**)" in settings["permissions"]["allow"]
    edit = {"file_path": "notes/BLOCAGES.md", "old_string": "x", "new_string": "y"}
    assert audit.check_tool_use("Edit", edit, hook.context(ws)) == []
    consigne = (ws / "CONSIGNE.md").read_text(encoding="utf-8")
    assert "Note ce qui te bloque dans `notes/BLOCAGES.md`" in consigne
    assert "`notes/BLOCAGES.md`" in (ws / "CLAUDE.md").read_text(encoding="utf-8")


def test_consigne_forbids_autoregressive_decoding(ws):
    consigne = (ws / "CONSIGNE.md").read_text(encoding="utf-8")
    assert "Décodage autorégressif interdit dans `run-script`" in consigne
    assert "réinjecte une sortie du modèle" in consigne and "échantillonnage" in consigne
    assert "`run-batch --thinking on`" in consigne and "`probe`" in consigne
    claude = (ws / "CLAUDE.md").read_text(encoding="utf-8")
    assert "Aucun décodage autorégressif dans `run-script`" in claude
    assert "lit tout ce dossier (et rien au-delà)" in consigne
    assert "Les instructions d'Anaël sont hors de l'espace." in consigne


# Amendement 8 : BLOCAGES.md publié avec le gel.


def test_freeze_publishes_blocages(ws, base):
    note = f"# Blocages\n\n- lecture de {Path.home()} refusée\n"
    (ws / "notes" / "BLOCAGES.md").write_text(note, encoding="utf-8")
    diff = freeze.freeze_session("s1", base, sessions_root=ws / "sessions")
    copy = ws / "sessions" / "s1" / "freeze" / "BLOCAGES.md"
    data = copy.read_bytes()
    assert data == "# Blocages\n\n- lecture de ~ refusée\n".encode()
    import hashlib

    assert diff["blocages"] == {"file": "BLOCAGES.md", "sha256": hashlib.sha256(data).hexdigest()}
    saved = json.loads((copy.parent / "diff.json").read_text(encoding="utf-8"))
    assert saved["blocages"] == diff["blocages"]


def test_freeze_outside_a_workspace_has_no_blocages(tmp_path, base):
    harness.open_session("s1", 10, base, sessions_root=tmp_path / "sessions")
    diff = freeze.freeze_session("s1", base, sessions_root=tmp_path / "sessions")
    assert "blocages" not in diff
    assert not (tmp_path / "sessions" / "s1" / "freeze" / "BLOCAGES.md").exists()
