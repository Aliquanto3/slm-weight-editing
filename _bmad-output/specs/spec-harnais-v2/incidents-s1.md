# Incidents de la séance s1 et réponse du harnais v2

Sources : `docs/CADRAGE.md`, ainsi que `EDITION.md` et `notes/journal_de_bord.md` de l'espace s1 (hors dépôt).

| Incident s1 | Fait relevé | Réponse v2 |
|---|---|---|
| Générations hors budget | `run-batch` applique le gabarit sans `enable_thinking`. L'éditeur a écrit un forward numpy (`qwen_np.py`, `qwen_batch.py`) et généré des traces avec raisonnement (problèmes 0 à 87, jusqu'à 3 000 jetons) hors lots. s1 n'est donc pas une réplication à budget égal. | CAP-1, CAP-2, CAP-3 |
| Rejeux en forward | Les 48 traces ont été rejouées en numpy pour lire le résidu et les activations MLP de la couche 23. C'est un forward forcé légitime, mais coûteux à écrire. | CAP-1 |
| Exactitude à la main | Le journal ne donne pas l'exactitude : l'éditeur a comparé à la main les réponses en gras. | hors périmètre (non demandé) |
| Machine saturée | Une autre session Claude Code ralentissait OpenBLAS d'un facteur 5. Quatre rejeux ont été tués à 30 min ; l'éditeur a sondé les threads (`threads_probe.py`). | CAP-8 |
| Verrou orphelin | `session.lock` est resté après les rejeux tués, et Anaël l'a supprimé à la main. | CAP-7 |
| Sortie tronquée | Le journal coupe la sortie de `run-script` à 20 000 octets. | CAP-8 |
| Faux positifs de l'audit | `re.compile` (pris pour `compile(`), le décodage des octets du tokenizer et Grep sur le journal de l'espace ont été signalés. | CAP-5, CAP-9 |
| Embedding d'entrée perturbé | Modifier la ligne de `</think>` dans les embeddings liés change aussi son entrée, présente dans le prompt : environ 20 % des réponses sans raisonnement changent au lot 2. | CAP-10 |
