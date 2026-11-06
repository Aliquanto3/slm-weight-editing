"""Crochet PreToolUse de l'espace de séance (stories 2.5 et 2.9, CAP-6) : chaque refus dit sa règle.

Lancé par Claude Code en forme exec : `<dépôt>/.venv/.../python -I -B _refusal_hook.py <espace>`.
Il lit l'appel d'outil en JSON sur l'entrée standard, le passe à `audit_transcript.check_tool_use`
(les règles de l'audit, pas une copie des deny) avec le `Context` de `<espace>/workspace.json`,
et répond `deny` avec la règle et la voie permise si l'audit y voit une violation.

- Il ne répond jamais `allow` ni `ask` : un `allow` passerait outre `dontAsk`. Un appel permis
  sort en code 0 sans rien écrire et suit les règles des réglages.
- Échec fermé (story 2.9, remplace la décision 2 de la fiche 2.5) : entrée illisible (vide, JSON
  invalide, non UTF-8, pas un objet), `workspace.json` absent ou illisible, exception de l'audit
  donnent `deny` avec la règle `crochet-en-echec` et le rappel BLOCAGES. Une analyse plus longue
  que `DEADLINE` (sous le délai `HOOK_TIMEOUT` de Claude Code) donne le même refus, « analyse trop
  longue ». Un délai dépassé ou un code de sortie autre que 0 ou 2 bloque aussi l'appel :
  `onFailure: "block"` du bloc rendu par `workspace.hook_settings`.
- La racine de l'espace vient de l'argument, jamais du `cwd` reçu ; un chemin relatif se résout
  sur le `cwd` reçu, puis liens et jonctions sont résolus par l'audit.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import threading
from pathlib import Path

from slm_weight_editing import audit_transcript as audit

BLOCAGES_HINT = "Si cette voie ne suffit pas, note le blocage dans notes/BLOCAGES.md."
FAILURE_RULE = "crochet-en-echec"
# Échéance interne (s), sous `workspace.HOOK_TIMEOUT` (10 s) : le refus part avant le délai.
DEADLINE = 6.0


def context(workspace: str | Path) -> audit.Context:
    """Contexte d'audit de l'espace : dépôt et séance lus dans `workspace.json`."""
    workspace = Path(workspace)
    info = json.loads((workspace / "workspace.json").read_text(encoding="utf-8"))
    return audit.Context.build(workspace, info["repo"], info["session_id"])


def violations(payload: dict, ctx: audit.Context) -> list[tuple[str, str]]:
    """Violations (règle, détail) de l'appel reçu, comme l'audit les verrait."""
    name = str(payload.get("tool_name", ""))
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = {"value": tool_input}
    cwd = payload.get("cwd")
    found: list[tuple[str, str]] = []
    if isinstance(cwd, str) and cwd:
        # Donnée de Claude Code, pas de l'éditeur : résolue (noms courts 8.3, liens) comme
        # `audit_transcript._cwd_ok`, et non refusée comme un chemin indéterminable.
        real = audit._real(cwd)
        if not os.path.isabs(cwd) or audit._norm_path(real) != ctx.workspace:
            found.append(("hors-espace-seance", f"dossier courant {cwd}"))
        else:
            ctx = dataclasses.replace(ctx, cwd=real)
    return found + audit.check_tool_use(name, tool_input, ctx)


def reason(found: list[tuple[str, str]]) -> str:
    """Motif du refus : une ligne par règle (première occurrence), puis le rappel BLOCAGES."""
    lines: dict[str, str] = {}
    for rule, detail in found:
        lines.setdefault(rule, audit.refusal_message(rule, detail))
    return "\n".join([*lines.values(), BLOCAGES_HINT])


def decide(payload: dict, ctx: audit.Context) -> dict | None:
    """Sortie du crochet : décision `deny` motivée, ou None (appel laissé aux réglages)."""
    found = violations(payload, ctx)
    return deny(found) if found else None


def deny(found: list[tuple[str, str]]) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason(found),
        }
    }


def failure(detail: str) -> dict:
    """Refus du crochet en échec : règle `crochet-en-echec`, détail, rappel BLOCAGES."""
    return deny([(FAILURE_RULE, detail)])


def _analyse(argv: list[str]) -> dict | None:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    except ValueError as exc:  # JSON invalide, entrée vide ou non UTF-8
        return failure(f"entrée illisible ({type(exc).__name__})")
    if not isinstance(payload, dict):
        return failure(f"entrée illisible (objet JSON attendu, reçu {type(payload).__name__})")
    return decide(payload, context(argv[1]))


def main(argv: list[str], deadline: float = DEADLINE) -> int:
    """Lit l'appel, écrit `deny` ou rien ; refuse sur exception ou au-delà de `deadline`."""
    done: list[tuple[bool, object]] = []

    def work() -> None:
        try:
            done.append((True, _analyse(argv)))
        except BaseException as exc:  # noqa: BLE001 - échec fermé : tout échec refuse
            done.append((False, exc))

    # Fil démon : à l'échéance, le refus part sans attendre la fin de l'analyse.
    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(deadline)
    if not done:
        out = failure(f"analyse trop longue (plus de {deadline:g} s)")
    elif done[0][0]:
        out = done[0][1]
    else:
        exc = done[0][1]
        out = failure(f"exception {type(exc).__name__} : {str(exc)[:200]}")
    if out is not None:
        try:
            sys.stdout.buffer.write(json.dumps(out, ensure_ascii=True).encode("ascii"))
            sys.stdout.buffer.flush()
        except Exception:  # noqa: BLE001 - sortie impossible : code 2, l'appel est bloqué
            return 2
    return 0


if __name__ == "__main__":
    code = main(sys.argv)
    sys.stdout.flush()
    os._exit(code)  # sans attendre un fil d'analyse encore en cours après l'échéance
