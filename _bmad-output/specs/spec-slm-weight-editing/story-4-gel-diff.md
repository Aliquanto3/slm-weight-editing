---
status: review
story: "4"
title: Gel de la séance et diff vérifiable de l'édition
---

# Story 4 : gel de la séance et diff vérifiable

SPEC CAP-2, CAP-1. Module `src/slm_weight_editing/freeze.py`, sous-commandes `session-freeze` et `apply-diff`.

## Critères et choix

- Gel : l'original passé est contrôlé d'abord (sha256 des tenseurs et des autres fichiers égaux à ceux de la ligne `open`, sinon refus sans rien écrire). Le diff est calculé avant de poser `frozen` (un plafond dépassé ne fige donc rien), puis `frozen` est posé via `harness.update_state` (réessai borné si un lot tourne), puis `work/` est re-hashé : s'il a changé pendant le gel, refus. Un gel posé sans diff (interruption) se reprend ; une séance avec événement `freeze` est refusée.
- Diff : tenseurs dont les octets diffèrent (comparaison par entier non signé de la largeur du dtype, donc bit à bit, NaN compris). Noms, formes et dtypes de `work/` doivent égaler l'original. Fichiers non safetensors qui diffèrent, ajoutés ou supprimés : `files_changed` du JSON et avertissement en CLI ; ils ne sont pas rejoués.
- Format `slm-weight-editing/edit-diff-v1` : `diff.json` (session, révision de base, budget, lots utilisés, `journal.sha256`, `values_file.sha256`, `max_changed_share`, `tensors[]` avec `name, shape, dtype, numel, changed, sha256_before, sha256_after, index_dtype, indices_offset, values_offset`, `files_changed`, `frozen_at`) et `diff.bin` (indices u4/u8 puis valeurs brutes, par tenseur).
- Plafond : `MAX_CHANGED_SHARE = 0.10` d'éléments changés par tenseur (l'édition du MIT en change 1,8 % des colonnes de deux tenseurs).
- Journal : copie `freeze/journal.jsonl` (sha256 dans le diff) puis événement `freeze` ajouté au journal de la séance.
- Rejeu : sha256 avant et après vérifiés avant toute écriture ; fichiers safetensors de l'original recopiés puis patchés aux mêmes offsets (même découpage en fichiers), autres fichiers recopiés ; sortie non vide refusée ; sortie supprimée en cas d'échec.
- `report-gguf --edit-diff` : sha256 attendus pris du diff, `--tag` obligatoire. `report_gguf` reçoit `provenance` : sans elle, MIT seulement si `expected_sha256` est omis ; des sha256 fournis sans provenance donnent `base` et `edit` à `null` (avant, tout diff se disait du MIT).
- `compare-edits --diff-a/--diff-b` : diff rejoué en mémoire (`freeze.diff_deltas`) puis `compare_deltas`. `--edit-a/--edit-b` ne sont plus exigés par argparse : `CompareError` (code 1) si un côté n'a pas exactement une source.

## Écarts et reports

- Pas de script d'application distinct : `apply-diff` en tient lieu.
- La correspondance GGUF reste limitée à `down_proj` (story 1) : un diff touchant un autre tenseur est refusé par `report-gguf`.
- La lignée de `work/` par copie (report de la story 3) reste à l'audit de la story 5.
