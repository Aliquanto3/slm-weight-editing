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
- Plafond : `MAX_CHANGED_SHARE = 0.10` d'éléments changés par tenseur (l'édition du MIT en change 1,8 % des colonnes de deux tenseurs). Décision d'Anaël du 2026-11-04 : valeur maintenue, éditions creuses imposées par la consigne de séance ; rappelé par `session-open` et `session-status` (`harness.CEILING_NOTE`).
- Journal : version publique `freeze/journal.jsonl` (chemins `base_checkpoint` et `data_path` réécrits par `freeze.public_path`, dossier personnel en `~` ; son sha256 dans le diff), puis événement `freeze` ajouté au journal brut de la séance, qui n'est pas publié.
- Rejeu : sha256 avant et après vérifiés avant toute écriture ; fichiers safetensors de l'original recopiés puis patchés aux mêmes offsets (même découpage en fichiers), autres fichiers recopiés ; sortie non vide refusée ; sortie supprimée en cas d'échec.
- `report-gguf --edit-diff` : sha256 attendus pris du diff, `--tag` obligatoire. `report_gguf` reçoit `provenance` : sans elle, MIT seulement si `expected_sha256` est omis ; des sha256 fournis sans provenance donnent `base` et `edit` à `null` (avant, tout diff se disait du MIT).
- `compare-edits --diff-a/--diff-b` : diff rejoué en mémoire (`freeze.diff_deltas`) puis `compare_deltas`. `--edit-a/--edit-b` ne sont plus exigés par argparse : `CompareError` (code 1) si un côté n'a pas exactement une source.

## Écarts et reports

- Pas de script d'application distinct : `apply-diff` en tient lieu.
- La correspondance GGUF reste limitée à `down_proj` (story 1) : un diff touchant un autre tenseur est refusé par `report-gguf`.
- La lignée de `work/` par copie (report de la story 3) reste à l'audit de la story 5.

## Revue

### Passe 1 (2026-11-04, SHA revu 3328dcd)

Base à jour (main 64d6b78, aucun rebase). Second avis Haiku repris point par point ; aucun constat propre en plus. Complexité : 0 constat (le journal n'est relu qu'une fois par gel).

- Important · medium · second avis n° 1 · corrigé : `freeze/journal.jsonl` copiait le journal brut, avec `base_checkpoint` et `data_path` absolus (chemin personnel publié, sha lié au diff). Le gel écrit une version publique (`public_journal`, `public_path`), c'est son sha256 qui va dans `diff.json` et dans l'événement `freeze` ; test `test_published_journal_has_no_local_path` (ni `tmp_path` ni le home dans `freeze/`).
- Important · medium · second avis n° 4 · corrigé : plafond découvert seulement au gel. Décision d'Anaël du 2026-11-04 : 10 % maintenu (éditions creuses imposées par la consigne de séance) ; constante déplacée dans `harness`, affichée par `session-open` et `session-status`, documentée dans le README (section du harnais).
- Important · low · second avis n° 2 et 3 · corrigé : diff altéré non validé (numel hors forme, `index_dtype` `f8` ou `i4`, offsets négatifs ou hors `diff.bin`, indices non triés) donnait `IndexError` ou une écriture à l'index -1. `load_diff` valide chaque entrée (`_check_entry`), `_entry_arrays` exige des indices triés, uniques et dans le tenseur : `DiffError`, captée par `apply-diff` et `compare-edits` ; test paramétré `test_malformed_diff_is_a_clean_diff_error`.
- Important · low · second avis n° 6 · corrigé : `report-gguf --edit-diff` ignorait `files_changed` ; il refuse désormais un tel diff, avec le motif ; test `test_report_gguf_refuses_a_diff_with_changed_files`.
- Important · low · second avis n° 7 · corrigé : contrôle de l'événement `freeze` hors verrou ; tout `freeze_session` tient maintenant le verrou de séance (réessai borné conservé) ; test `test_whole_freeze_runs_under_the_session_lock`.
- Nit · low · second avis n° 8 · corrigé : un échec après la pose du gel (work/ changé, écriture du diff) dit « séance déjà gelée, relancer session-freeze pour écrire le diff » ; test `test_failed_write_after_freeze_says_how_to_resume`.
- Nit · low · second avis n° 5 · documenté : un tenseur de 9 éléments ou moins ne peut changer d'aucun élément ; conséquence assumée de la décision du plafond (docstring de `freeze`, README).
- Nit · low · second avis n° 9 · rejeté : changement voulu par la spec (des sha256 fournis sans provenance ne se disent plus l'édition du MIT) ; l'assertion modifiée et `test_default_report_still_describes_the_mit_edit` couvrent les deux cas.

### Passe 2 (2026-11-04, SHA revu c70c58f)

Relecture des seuls correctifs de la passe 1 (`git show c70c58f`), base à jour (main 64d6b78). Les correctifs tiennent : le journal ne porte de chemin que dans `base_checkpoint` et `data_path` (`settings`, `problems`, `outputs` sans chemin ; l'événement `freeze` est écrit après la publication), le verrou tenu par `__enter__`/`__exit__` est bien relâché sur exception, `load_diff` refuse les entrées mal formées, les tests du harnais passent avec le plafond déplacé. Complexité : 0 constat.

- Nit · low · reviewer · corrigé : `public_path` gardait un chemin relatif quand le dossier courant est au-dessus du dossier personnel (racine du disque, `/home`) : `Users/<nom>/models/base` publié, nom d'utilisateur compris, sans préfixe que `public_text` sache masquer. Ce cas donne maintenant `.../<nom>` ; test `test_public_path_hides_user_name_when_cwd_is_above_home`.
- Nit · low · reviewer · corrigé : `_check_entry` appelait `.get` sur un `object` non restreint (signalé par Pyright) ; garde `isinstance(entry, dict)` explicite, comportement inchangé.
- Nit · low · reviewer · noté : le journal publié dépend du dossier courant au gel (relatif dessous, `.../<nom>` ailleurs), donc son sha256 aussi ; sans effet sur la vérification, puisque `diff.json` porte le sha256 de la version publiée, et le gel est unique par séance.
