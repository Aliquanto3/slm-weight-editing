---
status: review
story: "2.2"
title: run-script maîtrisé, threads, délai, sortie en fichier (CAP-8)
---

# Story 2.2 : run-script maîtrisé

- `session-workspace --script-threads (4) --script-timeout (900 s) --script-max-timeout (3600 s)` : validés (threads >= 1, délais > 0, défaut <= maximum), stockés dans `workspace.json` (non éditable par l'éditeur), relus par `run-script`. Pas d'option `--threads` sur run-script ; un espace sans ces clés prend les défauts.
- Threads : `OPENBLAS/OMP/MKL/BLIS/NUMEXPR_NUM_THREADS` et `VECLIB_MAXIMUM_THREADS` posées dans `child_env()` avant le démarrage de l'enfant, donc avant le préchargement de numpy ; modifier `os.environ` dans le script est sans effet, `ctypes` et `threadpoolctl` sont refusés par le crochet. `threadpoolctl` n'étant pas une dépendance, le test vérifie l'environnement vu par l'enfant.
- `--timeout` absent : délai par défaut ; > maximum ou <= 0 : refus sans exécution ni journal. Dépassement : arrêt de l'arbre (`taskkill /T /F` sous Windows, `killpg` ailleurs, session dédiée), `timed_out: true`.
- Sortie complète dans `<espace>/runs/<horodatage>-<script>.log` (création exclusive, jamais écrasé, `Edit(./runs/**)` refusé), plafond 50 Mio (`log_truncated`). Le journal garde l'extrait (20 000 octets), `output_bytes`, `log`, `log_bytes`, `log_sha256`, `timeout`.
- Consigne : délai, maximum, threads et fichier de sortie (`CONSIGNE.md`, `CLAUDE.md`).

Reports : taille 50 Mio non configurable ; fichiers `runs/` jamais purgés.

## Revue

### Passe 1 (2026-11-05, SHA revu ea78b64)

- Important, medium, second avis 1 : `check_limits` acceptait un délai infini ou démesuré (`inf`, `1e300`) ; `threading.Timer` lève alors OverflowError dans son thread (reproduit) et le script tourne sans délai. Corrigé : délais finis, au plus 24 h (`TIMEOUT_CEILING`), threads entiers ≥ 1 ; mêmes contrôles à la relecture de `workspace.json` (valeur abîmée : refus net, plus de traceback).
- Nit, low, second avis 6 : fichier `runs/` ouvert après le lancement (erreur d'ouverture : enfant orphelin, pas de journal). Corrigé : fichier créé avant `Popen` ; sha256 calculé en flux sur les octets écrits (plus de relecture du fichier).
- Nit, low, second avis 3 : threads seulement vus par l'environnement. Corrigé : test qui mesure `openblas_get_num_threads` de l'OpenBLAS de numpy chargé avec `child_env()`, égal au plafond (4 et 2) après modification de `os.environ` ; saut si numpy sans OpenBLAS embarqué.
- Nit, low, second avis 5 : arrêt de l'arbre non testé sous Windows. Corrigé : test multiplateforme (le petit-enfant tient le tube ; échoue si l'arbre n'est pas arrêté, vérifié par mutation).
- Nit, low, second avis 4 : `snapshot` excluait tout `runs/`. Corrigé : seulement le `runs/` voisin d'un `workspace.json`.
- Nit, low, second avis 2 : `log_sha256` non revérifié par `audit-session`. Rejeté : CAP-8 demande que le journal le garde, pas que l'audit le vérifie ; `runs/` est refusé à l'édition et aux scripts. Casse `Runs/` sous NTFS : rattaché au constat 7.
- Pre-existing, low, second avis 7 : noms protégés en casse mixte sous NTFS (`Workspace.json`, `Runs/`) passent les deny ; déjà reporté au lot 1, non traité ici.
