"""Crochet PreToolUse de l'espace de séance (story 2.5, CAP-6) : chaque refus dit sa règle.

Lancé par Claude Code en forme exec : `<dépôt>/.venv/.../python -I -B _refusal_hook.py <espace>`.
Il lit l'appel d'outil en JSON sur l'entrée standard, le passe à `audit_transcript.check_tool_use`
(les règles de l'audit, pas une copie des deny) avec le `Context` de `<espace>/workspace.json`,
et répond `deny` avec la règle et la voie permise si l'audit y voit une violation.

- Il ne répond jamais `allow` ni `ask` : un `allow` passerait outre `dontAsk`. Un appel permis
  sort en code 0 sans rien écrire et suit les règles des réglages.
- Un crochet en échec (entrée illisible, exception, délai) laisse passer l'appel : les deny des
  réglages restent la porte, le crochet ne fait qu'expliquer (décision 2 de la fiche 2.5).
- La racine de l'espace vient de l'argument, jamais du `cwd` reçu ; un chemin relatif se résout
  sur le `cwd` reçu, puis liens et jonctions sont résolus par l'audit.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from slm_weight_editing import audit_transcript as audit

BLOCAGES_HINT = "Si cette voie ne suffit pas, note le blocage dans notes/BLOCAGES.md."


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
        if audit._absolute(cwd, ctx) != ctx.workspace:
            found.append(("hors-espace-seance", f"dossier courant {cwd}"))
        else:
            ctx = dataclasses.replace(ctx, cwd=cwd)
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
    if not found:
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason(found),
        }
    }


def main(argv: list[str]) -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        if not isinstance(payload, dict):
            return 0
        out = decide(payload, context(argv[1]))
        if out is not None:
            sys.stdout.buffer.write(json.dumps(out, ensure_ascii=True).encode("ascii"))
            sys.stdout.buffer.flush()
    except Exception:  # noqa: BLE001 - un crochet en échec laisse les deny décider
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
