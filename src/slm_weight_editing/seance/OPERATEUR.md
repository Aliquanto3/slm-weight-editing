# Instructions d'Anaël pour la séance `<ID_SEANCE>`

Fichier écrit par `session-workspace` hors de l'espace de séance (`<ESPACE>`), dans le dépôt sous `models/operateur/` (ignoré par git). L'éditeur ne le voit pas : réglages (`Read` refusé), lanceur et audit le refusent.

## Avant de lancer l'éditeur

1. Aucun mod de Claude Code installé (un mod peut approuver un appel qu'un deny ou un crochet refuse).
2. Lancer l'éditeur depuis l'espace, et seulement ainsi (réglages utilisateur, plugins, MCP et commandes slash coupés ; `--bare` et `--safe-mode` couperaient aussi le crochet de l'espace) :

   ```
   cd <ESPACE>
   <LANCEMENT>
   ```

3. Avant le premier message, vérifier dans la session :

   | Commande | Résultat attendu |
   |---|---|
   | `/hooks` | un seul crochet `PreToolUse`, celui de l'espace (`_refusal_hook.py`, motif `*`) ; aucun autre, et aucun qui réponde `allow` |
   | `/memory` | seulement `CLAUDE.md` de l'espace ; ni mémoire utilisateur, ni `CLAUDE.md` du dépôt ou de ses parents |
   | `/mcp` | aucun serveur MCP |
   | `/plugin` | aucun plugin actif |
   | `/permissions` | mode `dontAsk`, liste blanche des quatre commandes, deny en place |

4. Essai à blanc : demander `cat CONSIGNE.md`. Attendu : refus avec le motif du crochet (`commande-hors-liste : ... ; permis : ...`). Un refus muet signale un crochet qui ne tourne pas.

## Gel de la séance

```
uv run slm-weight-editing session-freeze --session <ID_SEANCE> --sessions-root <SEANCES> --original <BASE>
```

Le gel copie `notes/BLOCAGES.md` dans `freeze/` (sha256 dans `diff.json`), avec le journal public.

## Audit de la transcription

Transcription : `~/.claude/projects/<projet de l'espace>/<id>.jsonl`, sous-agents compris.

```
uv run slm-weight-editing audit-session <transcription>.jsonl --workspace <ESPACE> --session-dir <SEANCES>/<ID_SEANCE> --reference-from <BASE> --base-revision <REVISION>
```

## Verrou orphelin

```
uv run slm-weight-editing session-unlock --session <ID_SEANCE> --sessions-root <SEANCES>
```

## Mesure

```
uv run slm-weight-editing apply-diff --diff <SEANCES>/<ID_SEANCE>/freeze/diff.json --original <BASE> --out models/rejoue/<ID_SEANCE>
uv run slm-weight-editing report-gguf --checkpoint models/rejoue/<ID_SEANCE> --base-checkpoint <BASE> --edit-diff <SEANCES>/<ID_SEANCE>/freeze/diff.json --tag qwen3.5-<ID_SEANCE>:0.8b
```

Puis `ollama create` et la campagne du banc `slm-quality-bench`, dans un terminal à part.
