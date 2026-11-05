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

Reports : test de l'arbre de processus seulement sous POSIX ; taille 50 Mio non configurable ; fichiers `runs/` jamais purgés.
