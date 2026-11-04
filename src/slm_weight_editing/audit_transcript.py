"""Audit de la transcription JSONL d'une séance d'édition Claude Code (story 5, CAP-6).

Lit les blocs `tool_use` des messages assistant (fichiers `~/.claude/projects/<projet>/<id>.jsonl`
et les transcriptions de sous-agents `<id>/subagents/*.jsonl`) et refuse tout ce que le protocole
interdit : web, lectures interdites, exécution du modèle hors du harnais, écriture de l'état de
séance, ouverture de séance par l'agent, commandes hors liste blanche. Avec le dossier de séance,
compare `base_tensors_sha256` de la ligne `open` du journal à une référence fournie par Anaël.

L'audit est textuel et volontairement large : un faux positif se relit à la main, un contournement
non vu ne se rattrape pas. Ses limites sont écrites dans la fiche de la story 5.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

RULES = {
    "transcription": "ligne de transcription illisible (JSON invalide)",
    "outil-non-autorise": "outil hors de la liste des outils de séance",
    "web": "accès web (WebFetch, WebSearch, MCP, client réseau)",
    "sous-agent": "sous-agent lancé (sa transcription échappe aux règles de séance)",
    "lecture-interdite": "lecture d'un chemin interdit (heldout, banc, journaux de référence)",
    "modele-hors-harnais": "exécution ou chargement du modèle hors des sous-commandes du harnais",
    "api-harnais": "module interne du dépôt appelé hors de la liste blanche",
    "etat-seance": "écriture, déplacement ou suppression de l'état de séance hors du harnais",
    "harnais-modifie": "modification du harnais, de l'environnement ou des réglages de séance",
    "ouverture-seance": "ouverture de séance par l'agent (réservée à Anaël)",
    "gel-seance": "gel de séance par l'agent (réservé à Anaël)",
    "commande-hors-liste": "sous-commande slm-weight-editing hors de run-batch et session-status",
    "git": "commande git ou gh (historique du dépôt)",
    "installation": "installation ou synchronisation de paquets",
    "obfuscation": "commande encodée, évaluée ou lancée par un sous-processus",
    "hors-espace-seance": "transcription dont le dossier courant n'est pas l'espace de séance",
    "journal-seance": "ligne open du journal absente, multiple ou illisible",
    "base-differente": "séance ouverte sur une base qui diffère de la révision publiée",
}

ALLOWED_SUBCOMMANDS = ("run-batch", "session-status")
ALLOWED_INTERNAL_MODULES = ("safetensors_file",)
# Outils de séance ; tout autre outil (WebFetch, WebSearch, Agent, Task, Skill, mcp__*...) échoue.
FILE_READ_TOOLS = {"Read", "Grep", "Glob", "LSP"}
FILE_WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
SHELL_TOOLS = {"Bash", "PowerShell"}
NEUTRAL_TOOLS = {"TodoWrite", "AskUserQuestion", "ToolSearch", "EnterPlanMode", "ExitPlanMode"}
WEB_TOOLS = {"WebFetch", "WebSearch"}
AGENT_TOOLS = {"Agent", "Task"}

PROTECTED_NAMES = ("state.json", "journal.jsonl", "registry.jsonl", "session.lock")

# Termes interdits cherchés aussi sous forme compacte (sans ponctuation ni espaces), pour voir
# `"held" + "out"` ou `held`out` ; les faux positifs (« threshold out ») se relisent à la main.
_FORBIDDEN_COMPACT = (
    "heldout",
    "slmqualitybench",
    "lmunderstand",
    "rikiyat",
    "bmadoutput",
    "referencemit",
    "weightspace",
    "cadrage",
    "reportgguf",
    "memlog",
)
_SEGMENTS = r"(docs|tests|\.git|_bmad|_bmad-output|\.claude)"
# Un segment de chemin : suivi de `/`, ou précédé de `/` (un mot isolé comme `tests` n'en est pas).
_FORBIDDEN_SEGMENT = re.compile(
    r"(?:^|[\s/=:(,\[{])" + _SEGMENTS + r"(?=/)|/" + _SEGMENTS + r"(?=\s|$|[),\]}*;|&])"
)
_LEAVE_WORKSPACE = re.compile(
    r"(?:^|[\s;|&(])(?:cd|pushd|chdir|set-location|sl|push-location)\s+(?:\.\.(?:/\.\.)*/?|/|~)"
    r"(?=[\s;|&)]|$)"
)
_FORBIDDEN_FILE = re.compile(
    r"(?:^|[\s/=:(,\[{])(readme\.md|agents\.md|slm_weight_editing/__init__\.py|\.memlog\.md)\b"
)
_SHORT_NAME = re.compile(r"(?:^|[/\s])[a-z0-9_-]{1,6}~\d")
_HF_CACHE = re.compile(r"huggingface|\.cache/hf\b")

_MODEL_EXEC = re.compile(
    r"\btransformers\b|\btorch\b|from_pretrained|\.generate\s*\(|\.forward\s*\(|\bpipeline\s*\("
    r"|llama|\bollama\b|\bvllm\b|onnxruntime|\bmlx\b|\boptimum\b|\baccelerate\b|\bjax\b"
    r"|tensorflow|\bkeras\b|\bsglang\b|\bpytest\b|\bgguf\b|automodel"
)
_NETWORK = re.compile(
    r"\bcurl\b|\bwget\b|invoke-webrequest|invoke-restmethod|\biwr\b|\birm\b|https?://|\bftp://"
    r"|\brequests\b|urllib|http\.client|\bhttpx\b|\baiohttp\b|\bsocket\b|net\.webclient"
    r"|start-bitstransfer|\bssh\b|\bscp\b|\bhf\s+(?:download|upload|auth|repo)|\bnc\b"
)
_OBFUSCATION = re.compile(
    r"base64|-encodedcommand|\s-e(?:nc?)?\s+[a-z0-9+/=]{16,}|\beval\b|\biex\b|invoke-expression"
    r"|\bexec\s*\(|__import__|importlib|\bcompile\s*\(|codecs\.decode|bytes\.fromhex|\[char\]"
    r"|\bchr\s*\(|rot13|\bxxd\b|start-process|subprocess|os\.system|os\.popen|os\.exec|\bpty\b"
    r"|\bctypes\b|\bmarshal\b|\bpickle\b"
)
_GIT = re.compile(r"(?:^|[\s;|&(])(?:git|gh)(?:\.exe)?(?=\s|$)")
_INSTALL = re.compile(
    r"\bpip3?\b|\buv\s+(?:add|pip|sync|tool|lock|remove)\b|\buvx\b|\bconda\b|\bwinget\b"
    r"|\bchoco\b|\bnpm\b|\bnpx\b|\bcargo\b|\bbrew\b|\bapt(?:-get)?\b|install-module"
)
_WRITE_VERB = re.compile(
    r"\brm\b|\brmdir\b|\bdel\b|\berase\b|\brd\b|remove-item|\bri\b|\bmv\b|\bmove\b|move-item"
    r"|\bmi\b|\bcp\b|\bcopy\b|copy-item|\bcpi\b|rename-item|\bren\b|\brename\b|\brni\b"
    r"|set-content|\bsc\b|add-content|\bac\b|out-file|clear-content|\bclc\b|new-item|\bni\b"
    r"|\btee\b|tee-object|\btruncate\b|\btouch\b|\bsed\b[^|;&]*\s-i|\bperl\b[^|;&]*\s-[a-z]*i"
    r"|\bdd\b|>|\bunlink\b|\bshred\b|\bchmod\b|\battrib\b|\bicacls\b|\bln\b|mklink|robocopy"
    r"|xcopy|\.write|\bopen\s*\(|os\.remove|os\.replace|os\.rename|shutil|rmtree|\.unlink"
    r"|\.rename|\.replace\s*\(|\.touch|write_text|write_bytes|json\.dump|\bsave\w*\s*\("
)
_DELETE_VERB = re.compile(
    r"\brm\b|\brmdir\b|\bdel\b|\berase\b|\brd\b|remove-item|\bri\b|rmtree|shutil|\bmv\b"
    r"|move-item|\bmove\b"
)
_HARMLESS_REDIRECT = re.compile(r"\d?>\s*(?:/dev/null|\$null|nul\b)|\d>&\d")
_PROTECTED_COMPACT = ("statejson", "journaljsonl", "registryjsonl", "sessionlock")
_SESSION_DIR = re.compile(
    r"sessions/?(?=[\s;|&)\]]|$)"  # dossier des séances
    r"|sessions/[^\s/;|&]+/?(?=[\s;|&)\]]|$)"  # dossier d'une séance
    r"|sessions/[^\s/;|&]+/(?!work(?:/|\b))\S*"  # tout fichier de la séance hors de work/
    r"|sessions/[^\s/;|&]+/work/?(?=[\s;|&)\]]|$)"  # work/ entier (copie, déplacement)
    r"|(?:^|[\s/])models/?(?=[\s;|&)\]]|$)"
)
_DOT_TARGET = re.compile(r"(?:^|\s)\.{1,2}/?(?=\s|$)")
_TAMPER_PATH = re.compile(
    r"(?:^|/)(?:src|\.venv|\.claude|tests|_bmad|_bmad-output)/|site-packages"
    r"|(?:^|/)(?:pyproject\.toml|uv\.lock|\.python-version|claude\.md|agents\.md|consigne\.md)$"
)
_TAMPER_MENTION = re.compile(
    r"(?:^|[\s/=])(?:src|\.venv|\.claude)/|site-packages|pyproject\.toml|uv\.lock"
    r"|(?:^|[\s/])(?:claude|agents|consigne)\.md\b"
)
_CLI = re.compile(
    r"(?<!from )(?<!import )(?<!-m )slm[-_]weight[-_]editing(?:\.exe)?(?=[\s;|&)]|$)\s*([^\s;|&)]*)"
)
_CLI_IMPORT = re.compile(
    r"(?:from|import)\s+slm_weight_editing(?:\.(\w+))?(?:\s+import\s+([\w \t,()]+))?"
)
_OPEN = re.compile(r"session-open|open_session|open-session")
_FREEZE = re.compile(r"session-freeze|freeze_session")


@dataclass(frozen=True)
class Violation:
    source: str
    line: int
    rule: str
    tool: str
    detail: str

    def render(self) -> str:
        return f"{self.source}:{self.line} · {self.rule} · {self.tool} · {self.detail}"


def normalize(text: str) -> str:
    """Minuscules, `/` pour `\\`, sans guillemets ni caractères d'échappement des shells."""
    text = text.replace("\\", "/").lower()
    return re.sub(r"[\"'`^]", "", text)


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def forbidden_reference(text: str) -> str | None:
    """Raison du refus si le texte désigne un chemin ou un nom interdit, None sinon."""
    norm = normalize(text)
    compact = _compact(norm)
    for term in _FORBIDDEN_COMPACT:
        if term in compact:
            return f"terme interdit « {term} »"
    for pattern in (_FORBIDDEN_SEGMENT, _FORBIDDEN_FILE):
        m = pattern.search(norm)
        if m:
            return f"chemin interdit « {next(g for g in m.groups() if g)} »"
    if _LEAVE_WORKSPACE.search(norm):
        return "sortie de l'espace de séance (cd vers un parent ou la racine)"
    if _SHORT_NAME.search(norm):
        return "nom court Windows (8.3) : cible indéterminable"
    if _HF_CACHE.search(norm):
        return "cache ou hub Hugging Face"
    return None


_OUTSIDE = re.compile(r"(?:^|/)\.\.(?:/|$)|^/|^~|^[a-z]:/|^\$home|^%userprofile%")
_ANCESTOR = re.compile(r"^(?:\.\./?)+$|^/$|^~/?$|^[a-z]:/?$|^\.\./\.\.")


def broad_reference(token: str) -> str | None:
    """Raison du refus d'un chemin qui sort de l'espace sans être lisible tel quel."""
    norm = normalize(token).strip()
    if _ANCESTOR.search(norm):
        return "répertoire parent ou racine entier"
    if _OUTSIDE.search(norm):
        if any(c in norm for c in "*?["):
            return "motif de chemin hors de l'espace de séance"
        if "$" in norm or "%" in norm:
            return "variable dans un chemin hors de l'espace de séance"
    return None


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _mentions_protected(norm: str, shell: bool) -> str | None:
    for name in PROTECTED_NAMES:
        if name in norm:
            return name
    compact = _compact(norm)
    for name in _PROTECTED_COMPACT:
        if name in compact:
            return name
    m = _SESSION_DIR.search(norm)
    if m:
        return m.group(0).strip()
    if shell:
        if _DELETE_VERB.search(norm) and _DOT_TARGET.search(norm):
            return "dossier courant ou parent"
        for token in norm.split():
            if any(c in token for c in "*?[") and "work/" not in token:
                return f"motif {token}"
    return None


def _check_code(text: str, tool: str, shell: bool) -> list[tuple[str, str]]:
    """Règles communes aux commandes shell et au code écrit par l'agent."""
    found: list[tuple[str, str]] = []
    norm = normalize(text)
    stripped = _HARMLESS_REDIRECT.sub(" ", norm)
    if _OPEN.search(norm):
        found.append(("ouverture-seance", _OPEN.search(norm).group(0)))
    if _FREEZE.search(norm):
        found.append(("gel-seance", _FREEZE.search(norm).group(0)))
    for m in _CLI.finditer(norm):
        sub = m.group(1)
        if sub not in ALLOWED_SUBCOMMANDS and not _OPEN.search(sub) and not _FREEZE.search(sub):
            found.append(("commande-hors-liste", f"sous-commande « {sub or '(aucune)'} »"))
    for m in _CLI_IMPORT.finditer(norm):
        if m.group(1):
            modules = [m.group(1)]
        else:
            modules = [n for n in re.split(r"[\s,()]+", m.group(2) or "") if n] or [""]
        for module in modules:
            if module not in ALLOWED_INTERNAL_MODULES:
                found.append(("api-harnais", f"slm_weight_editing.{module}".rstrip(".")))
    if re.search(r"-m\s+slm_weight_editing\b", norm):
        found.append(("api-harnais", "python -m slm_weight_editing"))
    m = _MODEL_EXEC.search(norm)
    if m:
        found.append(("modele-hors-harnais", f"« {m.group(0).strip()} »"))
    m = _NETWORK.search(norm)
    if m:
        found.append(("web", f"« {m.group(0).strip()} »"))
    m = _OBFUSCATION.search(norm)
    if m:
        found.append(("obfuscation", f"« {m.group(0).strip()} »"))
    if shell:
        if _GIT.search(norm):
            found.append(("git", "git ou gh"))
        m = _INSTALL.search(norm)
        if m:
            found.append(("installation", f"« {m.group(0).strip()} »"))
    reason = forbidden_reference(text)
    if not reason and shell:
        tokens = re.split(r"[\s;|&()<>=]+", text.replace("\\", "/"))
        reason = next((r for r in map(broad_reference, tokens) if r), None)
    if reason:
        found.append(("lecture-interdite", reason))
    if _WRITE_VERB.search(stripped):
        target = _mentions_protected(stripped, shell)
        if target:
            found.append(("etat-seance", f"écriture visant « {target} »"))
        m = _TAMPER_MENTION.search(stripped)
        if m and shell:
            found.append(("harnais-modifie", f"écriture visant « {m.group(0).strip()} »"))
    return found


def _check_write_path(path: str) -> list[tuple[str, str]]:
    norm = normalize(path)
    found: list[tuple[str, str]] = []
    name = norm.rsplit("/", 1)[-1]
    if name in PROTECTED_NAMES or re.search(r"sessions/[^/]+/(?!work/)", norm):
        found.append(("etat-seance", f"écriture de {path}"))
    elif norm.endswith("/sessions") or re.search(r"(?:^|/)sessions/[^/]+/?$", norm):
        found.append(("etat-seance", f"écriture de {path}"))
    if _TAMPER_PATH.search(norm):
        found.append(("harnais-modifie", f"écriture de {path}"))
    reason = forbidden_reference(path)
    if reason:
        found.append(("lecture-interdite", reason))
    return found


def check_tool_use(name: str, tool_input: dict) -> list[tuple[str, str]]:
    """Violations (règle, détail) d'un appel d'outil de l'agent."""
    if name in WEB_TOOLS or name.startswith("mcp__"):
        return [("web", f"outil {name}")]
    if name in AGENT_TOOLS:
        return [("sous-agent", f"outil {name}")]
    if name in SHELL_TOOLS:
        return _check_code(str(tool_input.get("command", "")), name, shell=True)
    if name in FILE_READ_TOOLS:
        found = []
        for value in _strings(tool_input):
            reason = forbidden_reference(value) or broad_reference(value)
            if reason:
                found.append(("lecture-interdite", reason))
        return found
    if name in FILE_WRITE_TOOLS:
        path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        found = _check_write_path(path)
        content = "\n".join(
            s
            for key, value in tool_input.items()
            if key not in ("file_path", "notebook_path", "old_string")
            for s in _strings(value)
        )
        return found + _check_code(content, name, shell=False)
    if name in NEUTRAL_TOOLS:
        return []
    return [("outil-non-autorise", f"outil {name}")]


def _cwd_ok(cwd: str, workspace: Path) -> bool:
    try:
        return Path(cwd).resolve() == workspace.resolve()
    except OSError:
        return False


def audit_transcript(path: str | Path, workspace: str | Path | None = None) -> list[Violation]:
    """Violations d'une transcription JSONL (une par règle et par appel d'outil)."""
    path = Path(path)
    source = path.name
    violations: list[Violation] = []
    seen_ids: set[str] = set()
    cwd_reported = False
    with path.open(encoding="utf-8") as f:
        for number, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                violations.append(Violation(source, number, "transcription", "-", "JSON invalide"))
                continue
            if not isinstance(entry, dict):
                continue
            cwd = entry.get("cwd")
            if workspace is not None and isinstance(cwd, str) and not cwd_reported:
                if not _cwd_ok(cwd, Path(workspace)):
                    violations.append(
                        Violation(source, number, "hors-espace-seance", "-", f"cwd {cwd}")
                    )
                    cwd_reported = True
            message = entry.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") not in (
                    "tool_use",
                    "server_tool_use",
                ):
                    continue
                block_id = block.get("id")
                if isinstance(block_id, str):
                    if block_id in seen_ids:
                        continue
                    seen_ids.add(block_id)
                name = str(block.get("name", ""))
                if block.get("type") == "server_tool_use":
                    name = name if name in WEB_TOOLS or name.startswith("mcp__") else "WebSearch"
                tool_input = block.get("input")
                if not isinstance(tool_input, dict):
                    tool_input = {"value": tool_input}
                for rule, detail in check_tool_use(name, tool_input):
                    violations.append(Violation(source, number, rule, name, detail))
    return violations


def load_reference(path: str | Path) -> tuple[dict[str, str], str | None]:
    """Référence de la base publiée : `{"revision": ..., "tensors_sha256": {...}}` ou dict plat."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("référence : objet JSON attendu")
    tensors = data.get("tensors_sha256", data)
    revision = data.get("revision") if "tensors_sha256" in data else None
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("référence : aucun sha256 de tenseur")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in tensors.items()):
        raise ValueError("référence : sha256 non textuels")
    return {k: v.lower() for k, v in tensors.items()}, revision


def audit_session_base(
    session: str | Path, reference: dict[str, str], revision: str | None = None
) -> list[Violation]:
    """Compare la ligne `open` du journal de séance à la référence de la base publiée."""
    journal = Path(session) / "journal.jsonl"
    source = journal.as_posix()
    opens: list[tuple[int, dict]] = []
    try:
        lines = journal.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [Violation(source, 0, "journal-seance", "-", f"journal illisible : {exc}")]
    for number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            return [Violation(source, number, "journal-seance", "-", "JSON invalide")]
        if isinstance(entry, dict) and entry.get("event") == "open":
            opens.append((number, entry))
    if len(opens) != 1:
        return [Violation(source, 0, "journal-seance", "-", f"{len(opens)} lignes open")]
    number, entry = opens[0]
    base = entry.get("base_tensors_sha256")
    if not isinstance(base, dict) or not base:
        return [Violation(source, number, "journal-seance", "-", "base_tensors_sha256 absent")]
    found: list[Violation] = []
    differing = sorted(k for k in base.keys() & reference.keys() if base[k] != reference[k])
    missing = sorted(reference.keys() - base.keys())
    extra = sorted(base.keys() - reference.keys())
    for label, names in (("différents", differing), ("absents", missing), ("en trop", extra)):
        if names:
            shown = ", ".join(names[:3]) + (" ..." if len(names) > 3 else "")
            detail = f"{len(names)} tenseurs {label} : {shown}"
            found.append(Violation(source, number, "base-differente", "-", detail))
    if revision is not None and entry.get("base_revision") != revision:
        detail = f"révision {entry.get('base_revision')} au lieu de {revision}"
        found.append(Violation(source, number, "base-differente", "-", detail))
    return found


def transcript_files(paths: list[str]) -> list[Path]:
    """Transcriptions données, plus celles des sous-agents (`<id>/subagents/*.jsonl`)."""
    files: list[Path] = []
    for p in map(Path, paths):
        files.append(p)
        files.extend(sorted(p.with_suffix("").joinpath("subagents").glob("*.jsonl")))
    return files


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "audit-session",
        help="audite la transcription d'une séance d'édition Claude Code",
        description=(
            "Échoue (code 1) sur tout accès web, lecture interdite, exécution du modèle hors du "
            "harnais, écriture de l'état de séance ou ouverture de séance par l'agent."
        ),
    )
    p.add_argument("transcripts", nargs="+", help="transcription(s) JSONL de la séance")
    p.add_argument("--workspace", help="espace de séance attendu comme dossier courant")
    p.add_argument("--session-dir", help="dossier de séance (models/sessions/<id>)")
    ref = p.add_mutually_exclusive_group()
    ref.add_argument("--reference", help="JSON des sha256 de la base publiée")
    ref.add_argument(
        "--reference-from", help="checkpoint de la révision publiée, dont on calcule les sha256"
    )
    p.add_argument("--base-revision", help="révision attendue de la base (ligne open)")
    p.add_argument("--write-reference", help="écrit la référence calculée par --reference-from")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        reference: dict[str, str] | None = None
        revision = args.base_revision
        if args.reference:
            reference, ref_revision = load_reference(args.reference)
            revision = revision or ref_revision
        elif args.reference_from:
            from slm_weight_editing.harness import tensor_hashes

            reference = tensor_hashes(Path(args.reference_from))
            if args.write_reference:
                Path(args.write_reference).write_text(
                    json.dumps({"revision": revision, "tensors_sha256": reference}, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
        if args.session_dir and reference is None:
            print("Erreur : --session-dir demande --reference ou --reference-from", file=sys.stderr)
            return 2
        files = transcript_files(args.transcripts)
        missing = [f for f in files if not f.is_file()]
        if missing:
            print(f"Erreur : transcription introuvable : {missing[0]}", file=sys.stderr)
            return 2
        violations = [v for f in files for v in audit_transcript(f, args.workspace)]
        if args.session_dir:
            violations += audit_session_base(args.session_dir, reference, revision)
    except (OSError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
    for v in violations:
        print(v.render())
    if violations:
        rules = sorted({v.rule for v in violations})
        print(f"Audit en échec : {len(violations)} violation(s) ({', '.join(rules)})")
        return 1
    print(f"Audit réussi : {len(files)} transcription(s), aucune violation")
    return 0
