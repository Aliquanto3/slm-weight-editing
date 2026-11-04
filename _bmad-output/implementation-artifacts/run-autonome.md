# Run autonome — lot 1 de slm-weight-editing

Fichier d'état de l'orchestrateur, à relire en premier après chaque `/clear`. Lu par `/agent-cost-kit:avancement-run`. Stories : `_bmad-output/specs/spec-slm-weight-editing/stories.yaml` ; l'id N y devient la story 1.N ici (1.1 fusionnée par la PR #3, 1.2 dans slm-quality-bench).

## Périmètre

| Périmètre | Epics | Note |
|---|---|---|
| Dans le run | Epics 1 à 1 | lot 1 : stories 1.3 à 1.6 (dépôt slm-weight-editing) ; 1.7 attend la séance d'Anaël |
| Hors run | Epics 2 à 2 | lots suivants |

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
- Dépendances : torch et transformers ajoutés par la story 1.3 seulement (`uv add`). Story 1.6 : numpy seul (SVD par `numpy.linalg`).
- Séance (story 1.3) : `models/sessions/<id>/` avec `state.json` (`budget`, `used`, `frozen`), `work/` (safetensors de travail), `journal.jsonl`. Le harnais refuse tout lot quand `frozen` est vrai ; la story 1.4 pose `frozen`.
- Comparaison (story 1.6) : entrées = checkpoint original + deux checkpoints édités (safetensors). L'entrée par diff est ajoutée par la story 1.4 (rejeu du diff vers un checkpoint).

## Sprint 1 — lot 1 (stories 1.3 à 1.6)

| Vague | Stories en parallèle | Après | Fichiers ou écrans touchés |
|---|---|---|---|
| V0 | 1.3 ★ · 1.6 | — | 1.3 `harness.py`, deps ; 1.6 `compare_edits.py` ; `__init__.py` partagé (contrat CLI) |
| V1 | 1.4 | 1.4 ← 1.3 ; 1.4 ← 1.6 | `freeze.py`, `harness.py`, `compare_edits.py` |
| V2 | 1.5 | 1.5 ← 1.4 | `session/`, `audit_transcript.py` |
| V3 | 1.7 (humaine) | 1.7 ← 1.5 | slm-quality-bench, après la séance d'Anaël |

## Journal

- 2026-11-04 08:34 · spec · PR · #5 ouverte (stories 3 à 7) · —
- 2026-11-04 08:35 · spec · fusion · #5 fusionnée, MERGED vérifié · 5ec3233
- 2026-11-04 08:35 · run · décision · cadrage dans les prompts, contrats CLI, deps et état de séance fixés · —
- 2026-11-04 08:36 · 1.3 · dev · lancé (dev-story Opus, worktree wt-s3, feat/harnais-budget) · —
- 2026-11-04 08:36 · 1.6 · dev · lancé (dev-story Sonnet, worktree wt-s6, feat/comparaison-editions) · —
- 2026-11-04 08:40 · 1.6 · dev · fini, 11 tests ajoutés, suite verte · b598c2d
- 2026-11-04 08:43 · 1.6 · second-avis · 2 medium, 7 low · —
- 2026-11-04 08:43 · 1.6 · review · lancée (Opus, passe 1) · —
- 2026-11-04 08:43 · 1.3 · dev · fini, 21 tests (39 cas), suite verte, torch 2.15.0 +cpu sous Linux · 33b924c
- 2026-11-04 08:44 · 1.6 · review · passe 1 : 2 Important corrigés, 5 low corrigés · d9929d9
- 2026-11-04 08:45 · 1.6 · PR · #6 ouverte · d9929d9
- 2026-11-04 08:45 · 1.6 · review · passe 2 lancée (correctifs de la passe 1) · —
- 2026-11-04 08:45 · 1.3 · second-avis · 4 high (budget contournable, compteur après journal, gel sans verrou), 3 medium, 2 low · —
- 2026-11-04 08:45 · 1.3 · review · passe 1 lancée (Opus) · —
- 2026-11-04 08:45 · 1.3 · décision · falsification manuelle de state.json et du journal renvoyée à l'audit de la story 5 (exigence notée dans story-3-harnais.md) · —
- 2026-11-04 08:47 · 1.6 · review · passe 2 : 0 constat · 9e4a6ca
- 2026-11-04 08:47 · 1.6 · CI · verte sur les trois OS · 9e4a6ca
- 2026-11-04 08:47 · 1.6 · fusion · #6 fusionnée, MERGED vérifié · c6011ff
- 2026-11-04 08:48 · 1.7 · attente humaine · attend la séance d'édition d'Anaël (après 1.5) · —
- 2026-11-04 08:48 · run · décision · stories renumérotées 1.N pour le suivi, sprint-status.yaml ajouté · —
- 2026-11-04 08:51 · 1.3 · review · passe 1 : 4 high, 2 medium, 2 low corrigés ; n° 6 reporté à 1.5 · 774e143
- 2026-11-04 08:51 · 1.3 · décision · push forcé refusé par le classifieur ; branche rebasée poussée sous feat/harnais-budget-rebase · 774e143
- 2026-11-04 08:51 · 1.3 · PR · #7 ouverte · 774e143
- 2026-11-04 08:51 · 1.3 · review · passe 2 lancée · —
- 2026-11-04 08:54 · 1.3 · review · passe 2 : 1 medium reporté à 1.5 (copie de work/ hors des séances), 2 low · ea20ab4
- 2026-11-04 08:54 · 1.3 · CI · verte sur les trois OS · ea20ab4
- 2026-11-04 08:54 · 1.3 · fusion · #7 fusionnée, MERGED vérifié · 64d6b78
- 2026-11-04 08:55 · 1.4 · décision · report-gguf reçoit une option qui lit les sha256 attendus dans le diff (CLI limitée aux sha256 du MIT) · —
- 2026-11-04 08:55 · 1.4 · dev · lancé (dev-story Sonnet, worktree wt-s4, feat/gel-diff-edition) · —
- 2026-11-04 09:03 · 1.4 · dev · fini, 26 tests, suite verte · 3328dcd
- 2026-11-04 09:03 · 1.4 · second-avis · lancé · —
- 2026-11-04 09:07 · 1.4 · second-avis · 2 medium (journal publié avec chemins absolus, plafond 10 % annoncé au gel seulement), 7 low · —
- 2026-11-04 09:07 · 1.4 · review · passe 1 lancée (Opus) · —
- 2026-11-04 09:07 · 1.5 · décision · lancée pendant la revue de 1.4 (fichiers disjoints, commandes du gel fixées) · —
- 2026-11-04 09:07 · 1.5 · dev · lancé (dev-story Opus, story à risque, worktree wt-s5, feat/protocole-seance) · —
- 2026-11-04 09:07 · 1.4 · décision · Anaël : plafond du diff gardé à 10 % par tenseur, éditions creuses imposées par la consigne de séance · —
- 2026-11-04 09:12 · 1.4 · review · passe 1 : 2 medium et 3 low Important corrigés, 1 low corrigé · c70c58f
- 2026-11-04 09:12 · 1.4 · PR · #8 ouverte · c70c58f
- 2026-11-04 09:12 · 1.4 · review · passe 2 lancée · —
- 2026-11-04 09:15 · 1.4 · review · passe 2 : 0 Important, 2 low corrigés (nom d'utilisateur depuis C:\, garde de type) · 9d352e2
- 2026-11-04 09:15 · 1.4 · CI · verte sur les trois OS · 9d352e2
- 2026-11-04 09:15 · 1.4 · fusion · #8 fusionnée, MERGED vérifié · 0fc353b
- 2026-11-04 09:19 · 1.5 · dev · fini, 240 tests verts (73 cas de violation), merge de main 0fc353b · d075517
- 2026-11-04 09:19 · 1.5 · second-avis · lancé, avec vérification de la syntaxe des permissions dans la doc officielle · —
- 2026-11-04 09:20 · 1.5 · second-avis · syntaxe des permissions vérifiée (doc officielle) : // et Read/Grep/Glob, PowerShell(...), claudeMdExcludes confirmés ; Task sans effet ; disableAutoMode à ajouter · —
- 2026-11-04 09:23 · 1.5 · second-avis · 4 high (lecture par chemin absolu, échappements, script lancé non audité, PYTHONPATH), 5 medium, 2 low · —
- 2026-11-04 09:23 · 1.5 · décision · audit textuel insuffisant seul : vérification d'un mode liste blanche et d'un lanceur de scripts à garde-fous d'exécution avant la revue · —
- 2026-11-04 09:25 · 1.5 · décision · refonte en trois couches : espace de séance hors du dépôt (dontAsk, blockReads/blockWrites, liste blanche), lanceur run-script (python -I, audit hook), audit renforcé ; pas de sandbox sous Windows natif (doc officielle) · —
- 2026-11-04 09:25 · 1.5 · dev · reprise lancée (dev-story Opus, même branche) · —
- 2026-11-04 09:53 · 1.5 · dev · reprise finie, 315 tests verts ; espace hors dépôt, lanceur run-script, audit renforcé · 5a2b88c
- 2026-11-04 09:53 · 1.5 · second-avis · relancé sur la reprise · —
- 2026-11-04 10:00 · 1.5 · second-avis · reprise : 2 high (run-batch sur racine et données non bornées, point final NTFS sur les noms protégés), 2 medium, 3 low ; lanceur tenu sur toutes les sondes directes · —
- 2026-11-04 10:00 · 1.5 · review · passe 1 lancée (Opus) · —
- 2026-11-04 10:09 · 1.5 · review · passe 1 : 4 Important corrigés (workspace.json, noms normalisés, copie exécutée, options auditées) · 18535f2
- 2026-11-04 10:09 · 1.5 · PR · #9 ouverte · 18535f2
- 2026-11-04 10:09 · 1.5 · review · passe 2 lancée · —
- 2026-11-04 10:14 · 1.5 · review · passe 2 : 0 Important, 1 low corrigé (abréviation argparse), 1 low reporté (Workspace.json en casse mixte, à vérifier par Anaël) · 697e5ba
- 2026-11-04 10:14 · 1.5 · CI · verte sur les trois OS · 697e5ba
- 2026-11-04 10:14 · 1.5 · fusion · #9 fusionnée, MERGED vérifié · c9496a5
- 2026-11-04 10:16 · run · CI · main verte sur les trois OS après la dernière fusion · c9496a5
- 2026-11-04 10:16 · run · ménage · worktrees et branches locales fusionnées retirés ; branches distantes soumises à Anaël · c9496a5
- 2026-11-04 14:40 · run · ménage · branches distantes fusionnées supprimées avec l'accord d'Anaël (10, dont feat/harnais-budget) ; topic GitHub status-spec -> status-dev · c9496a5
