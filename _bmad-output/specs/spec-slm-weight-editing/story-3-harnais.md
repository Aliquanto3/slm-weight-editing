---
id: STORY-3-harnais
status: review
parent: SPEC.md
capabilities: [CAP-5]
---

# Story 3 : harnais d'édition à budget de lots fixé avant la séance

## Livré

- Module `src/slm_weight_editing/harness.py`, sous-commandes `session-open`, `run-batch`, `session-status` (branchées par `add_parser(sub)` et `func`, changement minimal de `__init__.py`).
- Séance `models/sessions/<id>/` : `state.json` (`session_id`, `budget`, `used`, `frozen`, `base_checkpoint`, `base_revision`, `created_at`, `updated_at`), `work/` (copie des fichiers du checkpoint de base), `journal.jsonl` (ligne `open` puis une ligne `batch` par lot).
- API pour la story 4 : `read_state(dir)` (état validé), `update_state(dir, frozen=True)` (seul changement admis, pris sous `session.lock` : refusé avec « un lot tourne déjà » pendant un lot ; pas de dégel), `used_batches(dir)` (lots consommés recoupés avec le journal), `read_journal(dir)` (filtrer `event == "batch"` pour les sorties, `reserve` pour le décompte), `tensor_hashes(work)` (tenseurs seuls), `work_file_hashes(work)` (autres fichiers du checkpoint).
- Budget 10, 25 ou 50 (entier strict) ; réouverture refusée ; `update_state` et `_write_state` refusent tout changement de budget ; un `state.json` modifié à la main est refusé au lot suivant (budget comparé à la ligne `open` du journal).
- Refus avant lecture : jeu hors `gsm8k`/`math`, split hors `train`, chemin sous `heldout/` (chemin brut, absolu normalisé et résolu, donc `..` et liens), nom de fichier sans « train ». Puis refus des lots de plus de 512 problèmes et des lignes dont `split` n'est pas `train`.
- Lot : verrou `session.lock`, sha256 des tenseurs de `work/`, rechargement par `from_pretrained(local_files_only=True)`, `requires_grad_(False)`, `torch.inference_mode()`, glouton (`do_sample` False, `num_beams` 1, température nulle), `max_new_tokens` 4096 constant ; gabarit de chat s'il existe, texte brut sinon. `done_reason` : `stop` (jeton de fin) ou `length` (plafond). Le lot est réservé avant la génération (`state.json` puis ligne `reserve`, sous verrou) ; un plantage le consomme. Lots consommés = max(`state.used`, réservations du journal) ; `state.used` inférieur au journal refuse la séance. Écriture de `state.json` réessayée (20 × 0,1 s) sur `PermissionError`, puis échec sans génération.
- torch 2.15.0 : PyPI sous Windows et macOS (roues CPU), index `download.pytorch.org/whl/cpu` sous Linux seulement (`2.15.0+cpu`, sans `nvidia-*`) ; transformers 5.22.1.

## Tests

`tests/test_harness.py` : modèle Llama minuscule de config aléatoire et tokenizer WordLevel construits dans `tmp_path`, norme finale à zéro pour une sortie déterministe (jeton 0). Budgets, refus au-delà (journal inchangé), refus de budget et de modification, CLI et codes de sortie, gel posé à la main, 512/513 problèmes, jeux, splits, `heldout/` (relatif, `..`, casse, inexistant, lien symbolique sauté sans droit), contenu exact du journal, rechargement de `work/`, plafond 4096 atteint (`length`), absence de gradient (hook global de forward, `backward` et `Optimizer` patchés pour lever), verrou, identifiants de séance.

## Reports

- Les sha256 sont calculés juste avant le chargement des fichiers par transformers, pas sur les mêmes octets en mémoire (fenêtre de modification négligeable, l'agent n'écrit pas pendant un lot).
- Un verrou orphelin après un plantage se supprime à la main par Anaël (message explicite), jamais par l'agent.
- Génération problème par problème, sans lots rembourrés : à revoir si la durée d'un lot de 512 problèmes de 4096 jetons gêne la séance (GPU de séance à décider plus tard).
- Pas de vérification que le fichier de données vient bien du split train officiel au-delà du nom et du champ `split`.

## Exigence transmise à la story 5

Le harnais ne peut pas se défendre contre une réécriture cohérente, à la main, de `state.json` et de `journal.jsonl` ensemble (ni contre la suppression de `registry.jsonl`). L'audit de transcription de la story 5 doit échouer sur toute écriture, déplacement ou suppression, par l'agent éditeur et hors des commandes du harnais, de `state.json`, `journal.jsonl`, `session.lock`, `models/sessions/registry.jsonl` ou d'un dossier de séance (seul `work/` est modifiable par l'agent).

L'audit doit aussi échouer sur toute ouverture de séance (`open-session`) par l'agent éditeur : seul Anaël ouvre une séance, avant son début. Le harnais refuse une base prise dans un dossier de séance, mais une copie de `work/` hors de `models/sessions/` (sans `state.json` au-dessus) passe, et ouvrir une seconde séance depuis la base d'origine donnerait un budget neuf. L'audit compare donc aussi `base_tensors_sha256` de la ligne `open` aux sha256 du checkpoint de référence (révision publiée) et refuse toute base qui en diffère sans décision écrite. Un `session.lock` orphelin après un plantage se supprime par Anaël, pas par l'agent.

## Revue

### Passe 1 (2026-11-04, SHA revu 33b924c)

Base à jour au moment de la revue (merge-base = main 5ec3233), rebasée ensuite sur main c6011ff (story 6).

- Important | high | second avis n° 1 | corrigé | `open_session` refuse une base située sous le dossier des séances ou dans un dossier de séance (ancêtre portant `state.json` ou `journal.jsonl`, donc aussi une copie) ; sha256 des tenseurs et fichiers de la base dans la ligne `open`.
- Important | high | second avis n° 2 | corrigé | ligne `reserve` par lot ; `used_batches` refuse un `state.json` qui compte moins de lots que le journal (lot et `session-status`).
- Important | high | second avis n° 3 | corrigé | réservation avant la génération, écriture de `state.json` réessayée sur `PermissionError` puis `HarnessError` sans génération ni ligne de journal.
- Important | high | second avis n° 4 | corrigé | `update_state` prend `session.lock` et relit l'état sous verrou ; `run_batch` tient le verrou du contrôle de `frozen` à la fin du lot et n'écrit plus `state.json` après la génération : un gel ne peut ni être posé pendant un lot (refus explicite), ni être annulé.
- Important | medium | second avis n° 5 | corrigé | `work_file_hashes` (config, generation_config, tokenizer, gabarit) journalisé par lot (`files_sha256`) et à l'ouverture ; fonction distincte de `tensor_hashes` pour garder un dictionnaire de tenseurs seuls à la story 4. Test : un `generation_config.json` d'échantillonnage à 5 jetons dans `work/` ne change ni le glouton ni le plafond de 4096 (le harnais remplace `generation_config`).
- Important | medium | second avis n° 6 | reporté | falsification cohérente de `state.json` et du journal : hors de portée du harnais, exigence écrite pour la story 5 (section ci-dessus).
- Important | medium | second avis n° 7 | corrigé | un enregistrement sans `split` est refusé (README mis à jour : `split` obligatoire).
- Nit | low | second avis n° 8 | corrigé | registre `registry.jsonl` en ajout seul : un identifiant supprimé ne se rouvre pas.
- Nit | low | second avis n° 9 | corrigé | par la réservation : un plantage après génération consomme le lot (sorties perdues, décompte conservateur).
- Complexité : 0 constat (`used_batches` relit le journal une fois par lot, au plus 50 lots par séance).

### Passe 2 (2026-11-04, SHA revu 774e143)

Base à jour (merge-base = main c6011ff). Relu : `git show 774e143`. Suite complète 110 tests verts (1 sauté), ruff vert.

- Important | medium | relecteur | reporté (story 5) | copie de `work/` hors de `models/sessions/` puis `open-session` depuis cette copie, ou seconde séance depuis la base d'origine : budget neuf, que le contrôle de lignée ne voit pas (aucun `state.json` au-dessus). Le harnais ne peut pas reconnaître une base éditée sans référence ; exigence ajoutée à la section « Exigence transmise à la story 5 » (ouverture réservée à Anaël, sha256 de la base comparé à la révision publiée).
- Nit | low | relecteur | corrigé (fiche) | verrou orphelin : la fiche disait « se supprime à la main », alors que l'audit de la story 5 refuse toute suppression de `session.lock` par l'agent ; précisé : suppression par Anaël.
- Nit | low | relecteur | non corrigé | `base_files_sha256` parcourt la base récursivement (`rglob`) alors que la copie ne prend que les fichiers du premier niveau : empreinte plus large que `work/`, sans effet sur le décompte.
- Vérifié sans constat : `state.json` écrit puis échec de l'ajout `reserve` : `used` = `state.used` (lot compté, aucune génération, verrou rendu) ; l'inverse ne se produit pas (ordre fixe) et, fait à la main, il est refusé (`state.used` < réservations). Échec de `_write_state` : `HarnessError`, ni génération ni journal. Gel pendant un lot : refusé ; le lot n'écrit plus `state.json` après la génération, donc ne peut pas annuler un gel. `split` absent refusé.
- Rejeté | — | consigne | `safetensors` et `tokenizers` importés par les tests : dépendances obligatoires (non optionnelles) de `transformers` dans `uv.lock`, que le harnais utilise lui-même pour charger les poids ; garanties tant que `transformers` est une dépendance directe.
- Rejeté | — | Pyright | « Expected 0 positional arguments » (tests 158, 177, 216, 231) : faux positif, `main(argv: list[str] \| None = None)` accepte la liste.
- Complexité : 0 constat.
