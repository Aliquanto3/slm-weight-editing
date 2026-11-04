---
status: review
---

# Fix : commandes de séance sans chemin du dépôt (`UV_PROJECT`)

Défaut constaté en séance réelle s1 (2026-11-04) : les règles `allow` `Bash(uv run --project <dépôt> slm-weight-editing ...)` citent un chemin hors de l'espace ; avec `blockReadsOutsideWorkingDirectories: true`, Claude Code le traite comme un accès hors des répertoires de travail et le mode `dontAsk` refuse les trois commandes.

## Correctif

- `settings.json` : `"env": {"UV_PROJECT": "<dépôt en posix>"}` (uv lit `UV_PROJECT` comme `--project`). Le chemin du dépôt n'est plus écrit que dans `settings.json` (`env`) et `workspace.json`.
- Règles `allow`, `CONSIGNE.md`, `CLAUDE.md` : `uv run[ --extra cuda] slm-weight-editing <run-batch|session-status|run-script> ...` ; marqueur `<DEPOT>` supprimé des gabarits.
- `deny` : `UV_` reste (l'éditeur ne redéfinit pas `UV_PROJECT`) ; ajout de `*--project*` et `*--directory*` (Bash et PowerShell).
- Audit : liste blanche `^uv run( --extra cuda)? slm-weight-editing (run-batch|session-status|run-script)` ; `--project`, `--directory` et toute autre option uv échouent (l'ancienne forme n'est pas relue : aucune séance réelle n'a tourné). Le contrôle « dépôt attendu » lit `env.UV_PROJECT` de `<espace>/.claude/settings.json` : absent, illisible ou différent de `--repo` (défaut : ce dépôt) donne `commande-hors-liste`.

## Tests

`tests/test_session_workspace.py` (forme des allow, `env.UV_PROJECT`, deny, chemin absent des instructions rendues) ; `tests/test_audit_transcript.py` (formes avec et sans `--extra cuda`, `--project` et `--directory` refusés, `settings.json` : bon dépôt, autre dépôt, sans `env`, absent).
