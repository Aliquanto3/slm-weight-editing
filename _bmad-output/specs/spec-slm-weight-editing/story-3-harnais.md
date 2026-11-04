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
- API pour la story 4 : `read_state(dir)` (état validé), `update_state(dir, frozen=True)` (seul changement admis ; pas de dégel), `read_journal(dir)`, `tensor_hashes(work)`.
- Budget 10, 25 ou 50 (entier strict) ; réouverture refusée ; `update_state` et `_write_state` refusent tout changement de budget ; un `state.json` modifié à la main est refusé au lot suivant (budget comparé à la ligne `open` du journal).
- Refus avant lecture : jeu hors `gsm8k`/`math`, split hors `train`, chemin sous `heldout/` (chemin brut, absolu normalisé et résolu, donc `..` et liens), nom de fichier sans « train ». Puis refus des lots de plus de 512 problèmes et des lignes dont `split` n'est pas `train`.
- Lot : verrou `session.lock`, sha256 des tenseurs de `work/`, rechargement par `from_pretrained(local_files_only=True)`, `requires_grad_(False)`, `torch.inference_mode()`, glouton (`do_sample` False, `num_beams` 1, température nulle), `max_new_tokens` 4096 constant ; gabarit de chat s'il existe, texte brut sinon. `done_reason` : `stop` (jeton de fin) ou `length` (plafond). Le compteur n'avance qu'après écriture du journal.
- torch 2.15.0 : PyPI sous Windows et macOS (roues CPU), index `download.pytorch.org/whl/cpu` sous Linux seulement (`2.15.0+cpu`, sans `nvidia-*`) ; transformers 5.22.1.

## Tests

`tests/test_harness.py` : modèle Llama minuscule de config aléatoire et tokenizer WordLevel construits dans `tmp_path`, norme finale à zéro pour une sortie déterministe (jeton 0). Budgets, refus au-delà (journal inchangé), refus de budget et de modification, CLI et codes de sortie, gel posé à la main, 512/513 problèmes, jeux, splits, `heldout/` (relatif, `..`, casse, inexistant, lien symbolique sauté sans droit), contenu exact du journal, rechargement de `work/`, plafond 4096 atteint (`length`), absence de gradient (hook global de forward, `backward` et `Optimizer` patchés pour lever), verrou, identifiants de séance.

## Reports

- Les sha256 sont calculés juste avant le chargement des fichiers par transformers, pas sur les mêmes octets en mémoire (fenêtre de modification négligeable, l'agent n'écrit pas pendant un lot).
- Un verrou orphelin après un plantage se supprime à la main (message explicite).
- Génération problème par problème, sans lots rembourrés : à revoir si la durée d'un lot de 512 problèmes de 4096 jetons gêne la séance (GPU de séance à décider plus tard).
- Pas de vérification que le fichier de données vient bien du split train officiel au-delà du nom et du champ `split`.
