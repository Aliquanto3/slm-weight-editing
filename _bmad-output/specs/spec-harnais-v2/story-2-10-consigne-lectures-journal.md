---
status: done
story: "2.10"
title: Consigne sur les lectures de l'état de séance et journal illisible hors reprise
---

# Story 2.10 : consigne et journal illisible

Reports de la story 2.9.

## But

1. `CONSIGNE.md` et `CLAUDE.md` de la séance : le dossier se lit librement, mais un script qui mentionne l'état de séance (`state.json`, `journal.jsonl`, `registry.jsonl`, `session.lock`, `workspace.json`) autrement que par une lecture prouvée est refusé (famille `etat-seance`) ; le journal se lit par Read.
2. `harness.read_journal` (donc `used_batches`, `used_probe_tokens`) et `freeze.public_journal` (`session-freeze`) lèvent `harness.RecoveryError` (un `HarnessError`) au lieu d'une `JSONDecodeError` brute, via `harness.journal_unreadable`. Rien n'est modifié sur disque. `recover_detach` garde son message propre.

## Tests

`tests/test_freeze.py::test_unreadable_journal_gives_clear_error_and_touches_nothing` : ligne tronquée et octets non UTF-8, `used_batches` et `session-freeze`, octets du dossier de séance inchangés.

## Revue

### Passe 1 (2026-11-07, SHA revu 99c9fdc)

- Important · medium · second avis 1 · corrigé : `recover_detach` laissait passer le `RecoveryError` générique de `read_journal`, le message de reprise (« work/ et detach-bak/ laissés tels quels ») était perdu, contre le critère « recover_detach garde son message propre ». Sous-classe `JournalUnreadableError`, reprise ré-enveloppée ; le test de reprise 2.9 vérifie désormais `reprise … detach-bak/` (rouge sans le correctif).
- Nit · low · second avis 2 · corrigé : ligne de journal JSON non objet (`5`, `[1]`, `null`) → `AttributeError` brute. `harness.parse_journal` partagé par `read_journal` et `public_journal`, refus `illisible (TypeError)`.
- Nit · low · second avis 3 · corrigé : `public_journal` calculé avant `_freeze_state`, un journal illisible ne fige plus rien (coût nul).
- Nit · low · second avis 4 · rejeté : la consigne nomme bien l'exception juste après la liste ; le libellé de `RULES["etat-seance"]` date d'avant la story (Pre-existing, non bloquant, la consigne dit la règle exacte) ; heredoc Bash vers `scripts/` non vérifié et hors périmètre.
- Nit · low · second avis 5 · corrigé : test étendu à `used_probe_tokens` et aux lignes non objets, instantané de toute la racine des séances ; reprise couverte par le test 2.9 durci.
- Complexité : 0 constat (lecture du journal inchangée, O(n) par appel comme avant).
