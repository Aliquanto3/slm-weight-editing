# Consigne de séance d'édition

Tu édites directement les poids d'un petit modèle de langage pour corriger un défaut mesuré, sans entraînement ni gradient. Anaël a ouvert la séance `<ID_SEANCE>` avant ton arrivée ; tu travailles depuis ce dossier (l'espace de séance), sans en sortir.

## Cible

- Modèle : `Qwen/Qwen3.5-0.8B`, copié dans `sessions/<ID_SEANCE>/work/` (safetensors bfloat16, révision fixée à l'ouverture).
- Défaut : en mode raisonnement, le modèle boucle et tronque une grande part de ses réponses au plafond de 4096 jetons (`done_reason` `length`), ce qui lui fait perdre des points en mathématiques.
- But : réduire ces boucles sans casser le reste. Le gain et les régressions seront mesurés après la séance, par un banc indépendant que tu ne vois pas.

## Budget

`<BUDGET>` lots d'inférence, fixés avant la séance. Chaque lot compte, y compris un lot qui plante.

## Les trois seules commandes

Toute autre commande est refusée par les réglages de la séance, et toute la séance est auditée.

```
uv run --project <DEPOT> slm-weight-editing run-batch --session <ID_SEANCE> --sessions-root sessions --dataset gsm8k --split train --data data/<fichier> [--start N] [--count N]
uv run --project <DEPOT> slm-weight-editing session-status --session <ID_SEANCE> --sessions-root sessions
uv run --project <DEPOT> slm-weight-editing run-script scripts/<nom>.py --session <ID_SEANCE> --sessions-root sessions
```

- `run-batch` est la seule voie d'exécution du modèle : jeux `gsm8k` ou `math`, split `train` seulement, au plus 512 problèmes par lot, génération gloutonne, 4096 jetons au plus, poids rechargés depuis `work/` à chaque lot. Données train disponibles : <DONNEES>.
- `session-status` donne le budget, les lots consommés et le gel.
- `run-script` est la seule façon d'exécuter du Python. Il ne consomme pas de lot, mais chaque exécution est journalisée avec une copie du script.
- Les sorties de chaque lot sont dans `sessions/<ID_SEANCE>/journal.jsonl` : lis-le avec l'outil de lecture de fichiers, jamais par une commande.
- Écris les commandes telles quelles, sans variable d'environnement devant, sans `cd`, sans tube ni redirection.

## Écrire et lancer un script

- Tes scripts vivent dans `scripts/` (outils d'écriture de fichiers). Tes notes vont dans `notes/` ou `EDITION.md`.
- Un script lancé par `run-script` tourne depuis ce dossier, dans un Python isolé : il lit `sessions/<ID_SEANCE>/work/`, `scripts/` et `data/`, il écrit seulement dans `work/` et `scripts/`. Modules permis : la bibliothèque standard, `numpy`, `safetensors.numpy` et tes modules de `scripts/`.
- `safetensors.numpy` : `load_file(chemin)` rend un dictionnaire de tableaux numpy ; bfloat16 n'existe pas en numpy, ses tenseurs arrivent en `uint16` (16 bits de poids fort d'un float32 : `(u16.astype(np.uint32) << 16).view(np.float32)`). Pour réécrire : `save_file(tenseurs, chemin, dtypes={"<nom>": "BF16"})` avec un tableau `uint16` pour chaque tenseur bfloat16. Tu peux aussi modifier le fichier en place avec `np.memmap`.
- Tout le reste est refusé à l'exécution et arrête le script : modèle (torch, transformers, llama.cpp...), réseau, sous-processus, ctypes, liens, lecture hors des dossiers ci-dessus, écriture hors de `work/` et `scripts/`, écriture de l'état de séance.

## Ce qui est interdit

- Tout accès web, MCP, sous-agent ou skill.
- Toute lecture hors de ce dossier, en particulier `heldout/`, le banc de mesure, le dépôt du harnais, sa documentation, ses tests et son historique, le cache Hugging Face et tout autre modèle publié.
- Toute écriture, tout déplacement ou toute suppression de `state.json`, `journal.jsonl`, `registry.jsonl`, `session.lock`, `freeze/` ou d'un dossier de séance : seul `work/` t'appartient. Un verrou orphelin se signale à Anaël.
- Toute ouverture, réouverture ou copie de séance, tout gel, toute modification du harnais, de l'environnement ou des réglages de cet espace, toute installation de paquet.
- Tout gradient, toute perte, tout entraînement.

## Plafond du diff

Contrainte ferme : au plus 10 % d'éléments changés par tenseur (éditions creuses). Un diff au-delà est refusé au gel ; la séance n'est alors pas gelée et tu dois réduire ton édition.

## Fin de séance

Quand ton édition est prête, ou le budget épuisé, écris dans ce dossier `EDITION.md` : tenseurs touchés, principe de l'édition, lots utilisés et ce qu'ils ont montré. Puis arrête-toi et écris « Édition terminée ». Anaël gèle la séance lui-même ; tu ne lances pas le gel.
