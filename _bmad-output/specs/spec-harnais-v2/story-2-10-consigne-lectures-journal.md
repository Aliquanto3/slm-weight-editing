---
status: review
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
