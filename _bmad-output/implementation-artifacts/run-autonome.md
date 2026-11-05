# Run autonome — harnais v2 (séance s2)

Fichier d'état de l'orchestrateur, à relire en premier après chaque `/clear`. Lu par `/agent-cost-kit:avancement-run`. Stories : `_bmad-output/specs/spec-harnais-v2/stories.yaml` ; l'id N y devient la story 2.N ici. Run précédent (lot 1, epic 1) : `archive/run-autonome-lot1.md`.

## Périmètre

| Périmètre | Epics | Note |
|---|---|---|
| Dans le run | Epics 2 à 2 | harnais v2 : 2.1, 2.2, 2.3 et 2.7 sans Anaël ; 2.4, 2.5, 2.6 et 2.8 après relecture de leur spec par Anaël |
| Hors run | Epics 1 à 1 | 1.7 (campagne du lot 1 au banc) attend la séance s2 |

Au plus 2 stories en développement à la fois, chacune dans son worktree.
Plafond global : 5 agents actifs à la fois sous Windows.

## Règles

- Chaîne par story : dev-story (Sonnet) → second-avis (Haiku) → reviewer (Opus, avec la liste du second avis) → PR → CI verte → fusion par merge commit.
- Cadrage : pas de fichier BMAD par story ; l'orchestrateur écrit les critères Given/When/Then dans le prompt de dev-story d'après `SPEC.md` et `invoke_dev_with`.
- Points d'arrêt (consigne d'Anaël du 2026-11-05) : relecture de la spec par Anaël avant 2.4, 2.5, 2.6 et 2.8 (`spec_checkpoint`) ; arrêt après 2.7 pour l'essai Ollama d'Anaël (`done_checkpoint`). Si l'essai échoue, CAP-10 et 2.8 sortent de s2.
- Fichiers d'état du run : jamais dans une PR de story, PR `chore/etat-run-*` à part.
- Suite complète : la CI de branche (trois OS) sert de porte de fusion. Attente : `gh run watch <id> --interval 60`.
- Aucun Ollama, aucun téléchargement de modèle, aucune séance d'édition lancés par un agent.
- Heure du journal prise par `date`. `uv run python`, jamais `python` nu.

## Sessions autorisées à modifier le périmètre

- (aucune)

## Contrats partagés en cours

- Listes des commandes de l'espace : 2.1, 2.2 et 2.3 touchent toutes les règles allow/deny de `workspace.py` et la liste blanche d'`audit_transcript.py` ; elles passent donc l'une après l'autre (vagues V0 à V2), chacune partant de `main` à jour.
- Commandes d'Anaël interdites à l'éditeur (réglages et audit) : `session-open`, `session-freeze`, `session-workspace`, `session-unlock`, `audit-session`.

## Sprint 2 — harnais v2

| Vague | Stories en parallèle | Après | Fichiers ou écrans touchés |
|---|---|---|---|
| V0 | 2.1 · 2.7 | — | 2.1 `harness.py` (verrou), `workspace.py`, `audit_transcript.py`, `__init__.py` ; 2.7 outil GGUF (`report_gguf.py` ou module neuf), doc de l'essai |
| V1 | 2.2 | 2.2 ← 2.1 | `script_runner.py`, `_sandbox.py`, `workspace.py` |
| V2 | 2.3 | 2.3 ← 2.2 | `harness.py` (run-batch), `workspace.py`, `audit_transcript.py` |
| V3 | 2.4 (humaine) · 2.8 (humaine) | 2.4 ← 2.3 ; 2.8 ← 2.7 et essai Ollama | relecture de spec par Anaël d'abord |
| V4 | 2.5 (humaine) | 2.5 ← 2.4 | relecture de spec par Anaël d'abord ; story à risque |
| V5 | 2.6 (humaine) | 2.6 ← 2.5 | relecture de spec par Anaël d'abord |

## Journal

- 2026-11-05 21:32 · run · décision · run lancé sur consigne d'Anaël (2.1, 2.2, 2.3, 2.7 ; arrêts avant 2.4, 2.5, 2.6, 2.8 et après 2.7) ; aucune PR ouverte, aucune branche en cours · 7152600
