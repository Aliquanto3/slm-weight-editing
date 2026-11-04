---
status: review
---

# Fix : commandes de séance sans chemin du dépôt (`UV_PROJECT`)

Défaut constaté en séance réelle s1 (2026-11-04) : les règles `allow` `Bash(uv run --project <dépôt> slm-weight-editing ...)` citent un chemin hors de l'espace ; avec `blockReadsOutsideWorkingDirectories: true`, Claude Code le traite comme un accès hors des répertoires de travail et le mode `dontAsk` refuse les trois commandes.

## Correctif

- `settings.json` : `"env": {"UV_PROJECT": "<dépôt en posix>"}` (uv lit `UV_PROJECT` comme `--project`). Le chemin du dépôt n'est plus écrit que dans `settings.json` (`env`, `claudeMdExcludes`) et `workspace.json`.
- Règles `allow`, `CONSIGNE.md`, `CLAUDE.md` : `uv run[ --extra cuda] slm-weight-editing <run-batch|session-status|run-script> ...` ; marqueur `<DEPOT>` supprimé des gabarits.
- `deny` : `UV_` reste (l'éditeur ne redéfinit pas `UV_PROJECT`) ; ajout de `*--project*` et `*--directory*` (Bash et PowerShell).
- Audit : liste blanche `^uv run( --extra cuda)? slm-weight-editing (run-batch|session-status|run-script)` ; `--project`, `--directory` et toute autre option uv échouent (l'ancienne forme n'est pas relue : aucune séance réelle n'a tourné). Le contrôle « dépôt attendu » lit `env.UV_PROJECT` de `<espace>/.claude/settings.json` : absent, illisible ou différent de `--repo` (défaut : ce dépôt) donne `commande-hors-liste` ; `settings.local.json`, s'il existe, est prioritaire ; toute autre clé `UV_` de `env` (`UV_WORKING_DIR`...) donne `environnement` ; `--project` ou `--directory` n'importe où dans une commande (même après la sous-commande) donne `commande-hors-liste`.

## Tests

`tests/test_session_workspace.py` (forme des allow, `env.UV_PROJECT`, deny, chemin absent des instructions rendues) ; `tests/test_audit_transcript.py` (formes avec et sans `--extra cuda`, `--project` et `--directory` refusés, `settings.json` : bon dépôt, autre dépôt, sans `env`, absent).

## Revue

### Passe 1 (2026-11-04, SHA revu 0f4fb4d)

Base à jour (origin/main 89d1f0a). Conception jugée solide : le dépôt sort des commandes (cause du refus `dontAsk` + `blockReadsOutsideWorkingDirectories`, bissection de l'orchestrateur), `env.UV_PROJECT` est hors de portée de l'éditeur (`Edit(./.claude/**)` refusé, `UV_` refusé dans toute commande) et uv cherche `uv.toml`/`.python-version` depuis le projet, pas depuis l'espace.

- Nit, low, second avis 1 : l'audit ne lisait que `settings.json` ; `settings.local.json` (prioritaire) et un `UV_WORKING_DIR` posé dans `env` échappaient au contrôle. Corrigé : `env` fusionné des deux fichiers, autre clé `UV_` → `environnement` (tests `test_settings_local_overrides_uv_project`, `test_other_uv_variable_in_settings_is_refused`). Reste hors audit : un `UV_WORKING_DIR` hérité du terminal d'Anaël (non visible dans la transcription) ; la CLI refuserait alors l'espace (`workspace.json` introuvable), échec fermé.
- Nit, low, second avis 2 : comparaison lexicale (`C:/x/sub/..` accepté). Rejeté : `settings.json` est écrit par `session-workspace` et non modifiable par l'éditeur ; `..` normalisé désigne bien le même dossier.
- Nit, low, second avis 3 : README inexact, le chemin du dépôt figure aussi dans `claudeMdExcludes`. Corrigé (README, cette fiche, story 5).
- Nit, low, second avis 4 : fixture autouse qui remplace `_settings_uv_project` pour l'espace fictif `WS`, cas limites JSON non testés. Rejeté pour la fixture (elle ne vise que `WS`, qui n'existe pas ; les tests sur disque passent par la vraie fonction) ; cas JSON couverts en partie (absent, sans `env`, `settings.local.json`), le reste échoue fermé (`None` ⇒ violation).
- Nit, low, second avis 5 : `--project`/`--directory` après la sous-commande passaient l'audit. Vérifié : la CLI les refuse (`unrecognized arguments`) et les `deny` `*--project*`/`*--directory*` les bloquent. Corrigé quand même, par cohérence avec les `deny` : refus n'importe où (`test_project_or_directory_after_subcommand_is_refused`).
- Complexité : 0 constat (lecture des réglages une fois par audit).
- Fiche `story-5-protocole-seance.md` réalignée (allow, `--repo`, liste blanche, faux positif n° 10).
