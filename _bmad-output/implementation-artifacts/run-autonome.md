# Run autonome — lot 1 de slm-weight-editing

Fichier d'état de l'orchestrateur, à relire en premier après chaque `/clear`. Lu par `/agent-cost-kit:avancement-run`. Stories : `_bmad-output/specs/spec-slm-weight-editing/stories.yaml` (ids 3 à 7).

## Périmètre

| Périmètre | Epics | Note |
|---|---|---|
| Dans le run | Epics 3 à 6 | stories 3, 4, 5, 6 du lot 1 (dépôt slm-weight-editing) |
| Hors run | Epics 7 à 7 | campagne du lot 1 dans slm-quality-bench : attend la séance d'édition (Anaël) |
| Hors run | Epics 1 à 2 | story 1 fusionnée (PR #3) ; story 2 dans slm-quality-bench, hors de ce run |

Au plus 2 stories en développement à la fois, chacune dans son worktree.
Plafond global : 5 agents actifs à la fois sous Windows.

## Règles

- Chaîne par story : dev-story → second-avis (Haiku) → reviewer (Opus, avec la liste du second avis) → PR → CI verte → fusion par merge commit. ★ = story pivot, dev-story sur Opus.
- Cadrage : pas de fichier BMAD par story ; l'orchestrateur écrit les critères Given/When/Then dans le prompt de dev-story à partir de `invoke_dev_with` (décision du 2026-11-04).
- Fichiers d'état du run : jamais dans une PR de story, PR `chore/etat-run` à part.
- Suite complète : la CI de branche (trois OS) sert de porte de fusion. Attente : `gh run watch <id> --interval 60`.
- Aucun Ollama, aucun téléchargement de modèle, aucune séance d'édition lancés par un agent.
- Heure du journal prise par `date`. `uv run python`, jamais `python` nu.

## Sessions autorisées à modifier le périmètre

- (aucune)

## Contrats partagés en cours

- CLI : chaque nouveau module expose `add_parser(sub)` qui crée sa sous-commande et pose `set_defaults(func=run)` ; `run(args) -> int`. `__init__._parser` appelle `add_parser(sub)` ; `main` fait `if getattr(args, "func", None): return args.func(args)` avant le traitement de `report-gguf`. Conflits d'`__init__.py` résolus par l'orchestrateur.
- Dépendances : torch et transformers ajoutés par la story 3 seulement (`uv add`). Story 6 : numpy seul (SVD par `numpy.linalg`).
- Séance (story 3) : `models/sessions/<id>/` avec `state.json` (`budget`, `used`, `frozen`), `work/` (safetensors de travail), `journal.jsonl`. Le harnais refuse tout lot quand `frozen` est vrai ; la story 4 pose `frozen`.
- Comparaison (story 6) : entrées = checkpoint original + deux checkpoints édités (safetensors). L'entrée par diff est ajoutée par la story 4 (rejeu du diff vers un checkpoint).

## Sprint 1 — lot 1 (stories 3 à 6)

| Vague | Stories en parallèle | Après | Fichiers ou écrans touchés |
|---|---|---|---|
| V0 | 3 ★ · 6 | — | 3 `harness.py`, deps ; 6 `compare_edits.py` ; `__init__.py` partagé (contrat CLI) |
| V1 | 4 | 4 ← 3 ; 4 ← 6 | `freeze.py`, `harness.py`, `compare_edits.py` |
| V2 | 5 | 5 ← 4 | `session/`, `audit_transcript.py` |

## Journal

- 2026-11-04 08:34 · spec · PR · #5 ouverte (stories 3 à 7) · —
- 2026-11-04 08:35 · spec · fusion · #5 fusionnée, MERGED vérifié · 5ec3233
- 2026-11-04 08:35 · run · décision · cadrage dans les prompts, contrats CLI, deps et état de séance fixés · —
- 2026-11-04 08:36 · 3 · dev · lancé (dev-story Opus, worktree wt-s3, feat/harnais-budget) · —
- 2026-11-04 08:36 · 6 · dev · lancé (dev-story Sonnet, worktree wt-s6, feat/comparaison-editions) · —
- 2026-11-04 08:40 · 6 · dev · fini, 11 tests ajoutés, suite verte · b598c2d
- 2026-11-04 08:43 · 6 · second-avis · 2 medium, 7 low · —
- 2026-11-04 08:43 · 6 · review · lancée (Opus, passe 1) · —
- 2026-11-04 08:43 · 3 · dev · fini, 21 tests (39 cas), suite verte, torch 2.15.0 +cpu sous Linux · 33b924c
- 2026-11-04 08:44 · 6 · review · passe 1 : 2 Important corrigés, 5 low corrigés · d9929d9
- 2026-11-04 08:45 · 6 · PR · #6 ouverte · d9929d9
- 2026-11-04 08:45 · 6 · review · passe 2 lancée (correctifs de la passe 1) · —
- 2026-11-04 08:45 · 3 · second-avis · 4 high (budget contournable, compteur après journal, gel sans verrou), 3 medium, 2 low · —
- 2026-11-04 08:45 · 3 · review · passe 1 lancée (Opus) · —
- 2026-11-04 08:45 · 3 · décision · falsification manuelle de state.json et du journal renvoyée à l'audit de la story 5 (exigence notée dans story-3-harnais.md) · —
- 2026-11-04 08:47 · 6 · review · passe 2 : 0 constat · 9e4a6ca
- 2026-11-04 08:47 · 6 · CI · verte sur les trois OS · 9e4a6ca
- 2026-11-04 08:47 · 6 · fusion · #6 fusionnée, MERGED vérifié · c6011ff
