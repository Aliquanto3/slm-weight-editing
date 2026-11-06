"""Espace de séance hors du dépôt (story 5, CAP-6), créé par Anaël : `session-workspace`.

`<out>/` reçoit `.claude/settings.json` (dontAsk, liste blanche de quatre commandes, lectures et
écritures bloquées hors de l'espace), `CLAUDE.md`, `CONSIGNE.md`, `scripts/`, `data/` (copie des
fichiers train) et `sessions/` (dossier des séances du harnais, avec `<id>/work/`). Le chemin du
dépôt n'est écrit qu'ici, à la création, hors du dépôt. L'agent éditeur lancé dans `<out>/` ne
voit ni le dépôt, ni l'audit, ni `heldout/`, ni les CLAUDE.md et AGENTS.md du dépôt.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from slm_weight_editing import harness
from slm_weight_editing.probe import DEFAULT_PROBE_BUDGET, DEFAULT_PROBE_DISK_MAX
from slm_weight_editing.script_runner import (
    DATA_DIR,
    DEFAULT_MAX_TIMEOUT,
    DEFAULT_THREADS,
    DEFAULT_TIMEOUT,
    PROBES_DIR,
    RUNS_DIR,
    SCRIPTS_DIR,
    _inside,
    _python_roots,
    check_limits,
)

SESSIONS_DIR = harness.WORKSPACE_SESSIONS_DIR
NOTES_DIR = "notes"
BLOCAGES_FILE = "BLOCAGES.md"
BLOCAGES_HEADER = (
    "# Blocages de la séance\n\n"
    "Note ici ce qui te bloque : refus incompris, outil ou donnée qui manque, voie permise qui ne\n"
    "suffit pas. Une ligne par blocage : ce que tu voulais faire, le refus reçu, ce que tu as\n"
    "fait à la place. Ce fichier est publié avec le gel de la séance.\n\n"
)
# Instructions d'Anaël (CAP-6) : dans le dépôt, sous `models/` (ignoré par git), hors de l'espace.
OPERATOR_SUBDIR = Path("models") / "operateur"
OPERATOR_TEMPLATE = "OPERATEUR.md"
# Dossier des instructions d'Anaël ; None : `<dépôt>/models/operateur` (les tests le redirigent).
OPERATOR_DIR: Path | None = None
HOOK_SCRIPT = Path("src") / "slm_weight_editing" / "_refusal_hook.py"
HOOK_TIMEOUT = 10
# Lancement de l'éditeur (fiche 2.5, amendement 3) : réglages utilisateur, MCP, plugins et
# commandes slash exclus ; `--bare` et `--safe-mode` couperaient le crochet de l'espace.
LAUNCH_COMMAND = (
    "claude --strict-mcp-config --setting-sources project "
    "--tools Read,Grep,Glob,Edit,Write,Bash,PowerShell,TodoWrite --disable-slash-commands"
)
TEMPLATES = Path(__file__).with_name("seance")
ALLOWED_SUBCOMMANDS = ("run-batch", "session-status", "run-script", "probe", "detach-lm-head")
# Commandes lues ou exécutées sans règle en mode dontAsk (lecture seule) ou génériques : refusées.
DENIED_COMMANDS = (
    "cat head tail less more grep egrep fgrep rg find ls dir tree type file stat wc du df od xxd "
    "hexdump strings sed awk cut sort uniq diff cmp echo printf pwd cd pushd popd env printenv "
    "set export which where whereis realpath readlink basename dirname jq git gh python python3 "
    "py pip pip3 uvx curl wget ollama ln mklink cmd cmd.exe powershell pwsh bash sh zsh node npm "
    "npx perl ruby cp mv rm touch mkdir chmod tee dd source eval exec xargs nohup start sudo ssh "
    "scp nc claude subst fsutil junction"
).split()
DENIED_CMDLETS = (
    "Get-Content Get-ChildItem Select-String Get-Item Get-ItemProperty Get-ItemPropertyValue "
    "Test-Path Resolve-Path Get-Location Set-Location Push-Location Pop-Location Get-FileHash "
    "Format-Hex Get-Acl Out-File Set-Content Add-Content Clear-Content New-Item Copy-Item "
    "Move-Item Remove-Item Rename-Item Invoke-WebRequest Invoke-RestMethod Invoke-Expression "
    "Invoke-Command Invoke-Item Start-Process Start-Job Import-Module Get-Command Get-Help "
    "Get-Variable Write-Output Write-Host Get-PSDrive New-PSDrive Get-Process Expand-Archive "
    "Compress-Archive Get-Clipboard cmd python py git gh ollama pip uvx mklink fsutil"
).split()
# Sous-chaînes refusées n'importe où dans une commande (une règle deny matche après une
# affectation de variable en tête, contrairement à une règle allow).
DENIED_ANYWHERE = (
    "PYTHONPATH PYTHONHOME PYTHONSTARTUP UV_ --project --directory heldout slm-quality-bench "
    "session-open session-freeze session-unlock session-workspace apply-diff audit-session "
    "state.json "
    "journal.jsonl "
    "registry.jsonl session.lock workspace.json SymbolicLink Junction HardLink"
).split()
DENIED_UV = (
    "uv run python*",
    "uv run --with *",
    "uv run --script *",
    "uv pip *",
    "uv add *",
    "uv sync*",
    "uv tool *",
    "uv lock*",
)
PROTECTED_FILES = (
    harness.STATE_FILE,
    harness.JOURNAL_FILE,
    harness.REGISTRY_FILE,
    harness.LOCK_FILE,
    harness.WORKSPACE_FILE,
)
_SAFE_PATH = re.compile(r"^[A-Za-z0-9_:/.\-]+$")


class WorkspaceError(harness.HarnessError):
    """Espace de séance refusé."""


def posix(path: Path) -> str:
    return path.resolve().as_posix()


def rule_path(path: Path) -> str:
    """Chemin absolu d'une règle de permission : `//c/Users/...` (Windows), `//home/...` sinon.

    Claude Code ramène les chemins Windows à la forme POSIX (`C:\\Users` devient `/c/Users`)
    avant de les comparer, et `//` marque un chemin absolu depuis la racine.
    """
    text = posix(path)
    if len(text) > 1 and text[1] == ":":
        text = "/" + text[0].lower() + text[2:]
    return "/" + text


def hook_python(repo: Path) -> Path:
    """Interpréteur qui lance le crochet en forme exec (sans shell).

    Celui du `.venv` du dépôt, sinon celui de l'environnement virtuel qui fait tourner le harnais
    (`uv run` hors `.venv` local, CI). Chemin absolu jamais résolu : sous POSIX, `bin/python` est un
    lien vers l'interpréteur de base, qui perdrait les `site-packages` (et le `.pth` du dépôt).
    """
    rel = Path("Scripts", "python.exe") if sys.platform == "win32" else Path("bin", "python")
    local = (repo / ".venv" / rel).absolute()
    if local.is_file():
        return local
    if sys.prefix != sys.base_prefix and (Path(sys.prefix) / rel).is_file():
        return (Path(sys.prefix) / rel).absolute()
    return local


def operator_dir(repo: Path) -> Path:
    return (repo / OPERATOR_SUBDIR if OPERATOR_DIR is None else Path(OPERATOR_DIR)).resolve()


def hook_settings(workspace: Path, repo: Path) -> dict:
    """Bloc `hooks.PreToolUse` : crochet de refus en forme exec (`command` + `args`), motif `*`.

    Pas d'`onFailure` : un crochet en échec laisse l'appel aux deny des réglages (décision 2).
    """
    command = {
        "type": "command",
        "command": hook_python(repo).as_posix(),
        "args": ["-I", "-B", posix(repo / HOOK_SCRIPT), posix(workspace)],
        "timeout": HOOK_TIMEOUT,
    }
    return {"PreToolUse": [{"matcher": "*", "hooks": [command]}]}


def repo_root() -> Path:
    """Racine du dépôt du harnais (installation éditable, disposition src/)."""
    return Path(__file__).resolve().parents[2]


CUDA_EXTRA = " --extra cuda"


def allowed_commands(repo: Path, cuda: bool = False) -> list[str]:
    """Préfixes exacts des quatre commandes de l'éditeur (forme vérifiée par l'audit).

    Le dépôt n'y figure pas : `env.UV_PROJECT` des réglages le fixe (un chemin hors de l'espace
    cité par `--project` est refusé en `dontAsk` avec `blockReadsOutsideWorkingDirectories`).
    Avec `cuda`, `--extra cuda` suit `run` : `uv run` resynchroniserait sinon l'environnement
    sans l'extra et retirerait torch CUDA.
    """
    extra = CUDA_EXTRA if cuda else ""
    return [f"uv run{extra} slm-weight-editing {s}" for s in ALLOWED_SUBCOMMANDS]


def memory_excludes(workspace: Path, repo: Path) -> list[str]:
    """CLAUDE.md, CLAUDE.local.md et AGENTS.md des dossiers parents de l'espace et du dépôt."""
    dirs = [*workspace.resolve().parents, repo.resolve(), repo.resolve().parent]
    names = ("CLAUDE.md", "CLAUDE.local.md", "AGENTS.md", ".claude/CLAUDE.md")
    out: list[str] = []
    for d in dirs:
        for n in names:
            glob = f"{d.as_posix().rstrip('/')}/{n}"
            if glob not in out:
                out.append(glob)
    return out


def build_settings(
    workspace: Path,
    repo: Path,
    session_id: str,
    cuda: bool = False,
    operator: Path | None = None,
) -> dict:
    allow = [
        f"{tool}({cmd} *)"
        for tool in ("Bash", "PowerShell")
        for cmd in allowed_commands(repo, cuda)
    ]
    allow += [
        "Read(./**)",
        "Edit(./scripts/**)",
        f"Edit(./{SESSIONS_DIR}/{session_id}/work/*.safetensors)",
        f"Edit(./{NOTES_DIR}/**)",
        "Edit(./EDITION.md)",
    ]
    deny = ["WebFetch", "WebSearch", "Agent", "Skill", "mcp__*"]
    # Instructions d'Anaël : déjà hors de l'espace, refusées en plus par une règle explicite.
    deny.append(f"Read({rule_path(operator or operator_dir(repo))}/**)")
    for c in DENIED_COMMANDS:
        deny += [f"Bash({c})", f"Bash({c} *)"]
    deny += [f"Bash({c})" for c in DENIED_UV]
    for c in DENIED_CMDLETS:
        deny += [f"PowerShell({c})", f"PowerShell({c} *)"]
    deny += [f"PowerShell({c})" for c in DENIED_UV]
    for s in DENIED_ANYWHERE:
        deny += [f"Bash(*{s}*)", f"PowerShell(*{s}*)"]
    sessions = f"./{SESSIONS_DIR}"
    deny += [
        "Edit(./.claude/**)",
        "Edit(./CLAUDE.md)",
        "Edit(./CONSIGNE.md)",
        f"Edit(./{RUNS_DIR}/**)",
        f"Edit(./{PROBES_DIR}/**)",
        f"Edit(./{DATA_DIR}/**)",
        f"Edit({sessions}/{harness.REGISTRY_FILE})",
        f"Edit({sessions}/*/{harness.STATE_FILE})",
        f"Edit({sessions}/*/{harness.JOURNAL_FILE})",
        f"Edit({sessions}/*/{harness.LOCK_FILE})",
        f"Edit({sessions}/*/freeze/**)",
        f"Edit({sessions}/*/scripts/**)",
        f"Edit(./{harness.WORKSPACE_FILE})",
    ]
    # Noms protégés à toute profondeur (notes/ et scripts/ compris), suffixes compris : Windows
    # retire points et espaces finaux (`state.json.`) et accepte les flux (`state.json::$DATA`).
    for name in PROTECTED_FILES:
        for form in (name, name.upper()):
            deny += [f"Edit(**/{form})", f"Edit(**/{form}*)"]
    # Noms courts 8.3 (`STATE~1.JSO`) : cible indéterminable.
    deny += ["Edit(**/*~*)", "Edit(**/*~*/**)"]
    return {
        "$schema": "https://json.schemastore.org/claude-code-settings.json",
        "enableAllProjectMcpServers": False,
        "env": {"UV_PROJECT": posix(repo)},
        "claudeMdExcludes": memory_excludes(workspace, repo),
        "hooks": hook_settings(workspace, repo),
        "permissions": {
            "defaultMode": "dontAsk",
            "disableBypassPermissionsMode": "disable",
            "disableAutoMode": "disable",
            "blockReadsOutsideWorkingDirectories": True,
            "blockWritesOutsideWorkingDirectories": True,
            "allow": allow,
            "deny": deny,
        },
    }


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:g} min"


def render(
    name: str,
    *,
    session_id: str,
    budget: int,
    repo: Path,
    data: list[str],
    cuda: bool = False,
    threads: int = DEFAULT_THREADS,
    timeout: float = DEFAULT_TIMEOUT,
    max_timeout: float = DEFAULT_MAX_TIMEOUT,
    probe_budget: int = DEFAULT_PROBE_BUDGET,
    extra: dict[str, str] | None = None,
) -> str:
    text = (TEMPLATES / name).read_text(encoding="utf-8")
    for key, value in (extra or {}).items():
        text = text.replace(key, value)
    listing = ", ".join(f"`{DATA_DIR}/{n}`" for n in data)
    for key, value in (
        ("<ID_SEANCE>", session_id),
        ("<BUDGET>", str(budget)),
        ("<BUDGET_PROBE>", f"{probe_budget:,}".replace(",", " ")),
        ("<EXTRA>", CUDA_EXTRA if cuda else ""),
        ("<DONNEES>", listing),
        ("<SCRIPT_THREADS>", str(threads)),
        ("<SCRIPT_TIMEOUT>", _minutes(timeout)),
        ("<SCRIPT_MAX_TIMEOUT>", _minutes(max_timeout)),
    ):
        text = text.replace(key, value)
    return text


def check_location(out: Path, repo: Path) -> None:
    """Refuse un espace dans le dépôt, à côté de lui (GitHub/) ou au-dessus de lui."""
    out, repo = out.resolve(), repo.resolve()
    if not _SAFE_PATH.match(posix(repo)):
        raise WorkspaceError(f"chemin du dépôt inutilisable dans une règle : {posix(repo)}")
    if _inside(out, repo.parent) or _inside(repo, out):
        raise WorkspaceError(f"espace refusé : {out} doit être hors de {repo.parent}")
    for r in _python_roots():
        if _inside(out, Path(r)):
            raise WorkspaceError(f"espace refusé : {out} est sous l'interpréteur {r}")
    if out.exists() and any(out.iterdir()):
        raise WorkspaceError(f"espace refusé : {out} existe et n'est pas vide")
    python = hook_python(repo)
    if not python.is_file() or not _SAFE_PATH.match(python.as_posix()):
        raise WorkspaceError(f"interpréteur du crochet introuvable : {python} (uv sync)")
    if not (repo / HOOK_SCRIPT).is_file():
        raise WorkspaceError(f"crochet de refus introuvable : {repo / HOOK_SCRIPT}")


def cuda_platform() -> bool:
    """L'extra `cuda` ne résout une roue CUDA que sous Windows (remplacé dans les tests)."""
    return sys.platform == "win32"


def create_workspace(
    out: str | Path,
    session_id: str,
    budget: int,
    base_checkpoint: str | Path,
    data: list[str | Path],
    *,
    base_revision: str | None = None,
    repo: str | Path | None = None,
    cuda: bool = False,
    script_threads: int = DEFAULT_THREADS,
    script_timeout: float = DEFAULT_TIMEOUT,
    script_max_timeout: float = DEFAULT_MAX_TIMEOUT,
    probe_budget: int = DEFAULT_PROBE_BUDGET,
    probe_disk_max: int = DEFAULT_PROBE_DISK_MAX,
) -> Path:
    """Ouvre la séance dans `<out>/sessions` et écrit l'espace de l'éditeur ; renvoie `<out>`."""
    if cuda and not cuda_platform():
        # L'extra `cuda` ne pointe vers une roue CUDA que sous Windows (cu132) : ailleurs,
        # `uv sync --extra cuda` échoue (macOS) ou installe torch CPU (Linux).
        raise harness.HarnessError("--cuda refusé : l'extra cuda ne vise que Windows")
    check_limits(script_threads, script_timeout, script_max_timeout)
    out = Path(out).resolve()
    repo = repo_root() if repo is None else Path(repo).resolve()
    check_location(out, repo)
    operator = operator_dir(repo) / f"{session_id}.md"
    if operator.exists():
        raise WorkspaceError(f"instructions déjà écrites : {operator} (séance déjà préparée)")
    if not data:
        raise WorkspaceError("au moins un fichier de données train (--data)")
    files = [harness.check_data_request("gsm8k", harness.TRAIN_SPLIT, d) for d in data]
    for f in files:
        if not f.is_file():
            raise WorkspaceError(f"données introuvables : {f}", 1)
    if len({f.name for f in files}) != len(files):
        raise WorkspaceError("deux fichiers de données portent le même nom")
    harness.open_session(
        session_id,
        budget,
        base_checkpoint,
        base_revision=base_revision,
        sessions_root=out / SESSIONS_DIR,
        probe_budget=probe_budget,
        probe_disk_max=probe_disk_max,
    )
    # Séance, racine et données imposées aux commandes lancées depuis l'espace (harnais, lanceur).
    info = {
        "workspace": str(out),
        "session_id": session_id,
        "sessions_root": str((out / SESSIONS_DIR).resolve()),
        "data_dir": str((out / DATA_DIR).resolve()),
        "repo": str(repo),
        "cuda": cuda,
        "script_threads": script_threads,
        "script_timeout": script_timeout,
        "script_max_timeout": script_max_timeout,
        "probe_budget": probe_budget,
        "probe_disk_max": probe_disk_max,
    }
    (out / harness.WORKSPACE_FILE).write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for d in (SCRIPTS_DIR, DATA_DIR, NOTES_DIR, ".claude"):
        (out / d).mkdir(exist_ok=True)
    for f in files:
        shutil.copy2(f, out / DATA_DIR / f.name)
    (out / NOTES_DIR / BLOCAGES_FILE).write_text(BLOCAGES_HEADER, encoding="utf-8", newline="\n")
    settings = build_settings(out, repo, session_id, cuda, operator.parent)
    (out / ".claude" / "settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    names = [f.name for f in files]
    for name in ("CLAUDE.md", "CONSIGNE.md"):
        text = render(
            name,
            session_id=session_id,
            budget=budget,
            repo=repo,
            data=names,
            cuda=cuda,
            threads=script_threads,
            timeout=script_timeout,
            max_timeout=script_max_timeout,
            probe_budget=probe_budget,
        )
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    write_operator(operator, out, session_id, base_checkpoint, base_revision)
    return out


def write_operator(
    path: Path, out: Path, session_id: str, base: str | Path, revision: str | None
) -> None:
    """OPERATEUR.md hors de l'espace (CAP-6), jamais écrasé : gel, audit, verrou, mesure."""
    text = render(
        OPERATOR_TEMPLATE,
        session_id=session_id,
        budget=0,
        repo=out,
        data=[],
        extra={
            "<ESPACE>": str(out),
            "<SEANCES>": str(out / SESSIONS_DIR),
            "<BASE>": str(Path(base).resolve()),
            "<REVISION>": revision or "<révision publiée>",
            "<LANCEMENT>": LAUNCH_COMMAND,
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "session-workspace",
        help="ouvre une séance dans un espace hors du dépôt (réglages Claude Code, consigne)",
        description=(
            "Pour Anaël : crée <out>/ (réglages dontAsk, CLAUDE.md, CONSIGNE.md, scripts/, "
            "data/, sessions/) et y ouvre la séance. L'éditeur se lance ensuite dans <out>/."
        ),
    )
    p.add_argument("--out", required=True, help="dossier neuf hors du dépôt et de son parent")
    p.add_argument("--session", required=True)
    p.add_argument("--budget", required=True, type=int, help="10, 25 ou 50 lots")
    p.add_argument("--base-checkpoint", required=True, help="dossier safetensors HF de base")
    p.add_argument("--base-revision", help="révision HF du checkpoint de base")
    p.add_argument("--data", required=True, action="append", help="fichier JSONL train (répétable)")
    p.add_argument(
        "--cuda",
        action="store_true",
        help="poste avec GPU : les commandes de l'éditeur portent --extra cuda",
    )
    p.add_argument(
        "--script-threads",
        type=int,
        default=DEFAULT_THREADS,
        help="plafond de threads de calcul (BLAS) des scripts (défaut 4)",
    )
    p.add_argument(
        "--script-timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="délai par défaut d'un script, en secondes (défaut 900)",
    )
    p.add_argument(
        "--script-max-timeout",
        type=float,
        default=DEFAULT_MAX_TIMEOUT,
        help="délai maximum qu'un script peut demander, en secondes (défaut 3600)",
    )
    p.add_argument(
        "--probe-budget",
        type=int,
        default=DEFAULT_PROBE_BUDGET,
        help="budget de probe, en jetons passés en forward (défaut 2000000)",
    )
    p.add_argument(
        "--probe-disk-max",
        type=int,
        default=DEFAULT_PROBE_DISK_MAX,
        help="plafond cumulé des captures de probe, en octets (défaut 20 Go)",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        out = create_workspace(
            args.out,
            args.session,
            args.budget,
            args.base_checkpoint,
            args.data,
            base_revision=args.base_revision,
            cuda=args.cuda,
            script_threads=args.script_threads,
            script_timeout=args.script_timeout,
            script_max_timeout=args.script_max_timeout,
            probe_budget=args.probe_budget,
            probe_disk_max=args.probe_disk_max,
        )
    except harness.HarnessError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return exc.exit_code
    except OSError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    print(f"Espace de séance : {out}")
    print(f"Séance {args.session} ouverte dans {out / SESSIONS_DIR} : budget {args.budget} lots")
    print(harness.CEILING_NOTE)
    repo = repo_root()
    print(f"Instructions d'Anaël (hors de l'espace) : {operator_dir(repo) / args.session}.md")
    print(f"Lancer l'éditeur depuis {out} : {LAUNCH_COMMAND}")
    print("Avant le premier message : /hooks, /memory, /mcp, /plugin et /permissions")
    return 0
