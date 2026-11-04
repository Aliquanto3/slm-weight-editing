# Consigne de séance d'édition

Tu édites directement les poids d'un petit modèle de langage pour corriger un défaut mesuré, sans entraînement ni gradient. Anaël a ouvert la séance `<ID_SEANCE>` avant ton arrivée ; tu travailles depuis ce dossier `session/`.

## Cible

- Modèle : `Qwen/Qwen3.5-0.8B`, copié dans `../models/sessions/<ID_SEANCE>/work/` (safetensors bfloat16, révision fixée à l'ouverture).
- Défaut : en mode raisonnement, le modèle boucle et tronque une grande part de ses réponses au plafond de 4096 jetons (`done_reason` `length`), ce qui lui fait perdre des points en mathématiques.
- But : réduire ces boucles sans casser le reste. Le gain et les régressions seront mesurés après la séance, par un banc indépendant que tu ne vois pas.

## Budget

`<BUDGET>` lots d'inférence, fixés avant la séance. Chaque lot compte, y compris un lot qui plante. `uv run slm-weight-editing session-status --session <ID_SEANCE> --sessions-root ../models/sessions` donne le budget, les lots consommés et le gel.

## Seule voie d'exécution du modèle : le harnais

```
uv run slm-weight-editing run-batch --session <ID_SEANCE> --sessions-root ../models/sessions --dataset gsm8k --split train --data ../data/<fichier train>.jsonl [--start N] [--count N]
```

- Jeux `gsm8k` ou `math`, split `train` seulement, au plus 512 problèmes par lot. Génération gloutonne, 4096 jetons au plus, poids rechargés depuis `work/` à chaque lot.
- Les sorties de chaque lot sont dans `../models/sessions/<ID_SEANCE>/journal.jsonl` : tu peux le lire, jamais l'écrire.
- Ne charge jamais le modèle autrement : ni transformers, ni torch, ni llama.cpp, ni ollama, ni aucune autre bibliothèque d'inférence, ni les tests du dépôt.

## Ce que tu peux faire

- Lire et écrire les fichiers `*.safetensors` de `../models/sessions/<ID_SEANCE>/work/` par un script Python écrit dans ce dossier, avec numpy et `slm_weight_editing.safetensors_file` (lecture des octets bruts). bfloat16 : 16 bits de poids fort d'un float32 (vue `uint16` puis décalage de 16 bits). Écris le chemin complet de `work/` en un seul littéral dans tes scripts.
- Lire, analyser et réécrire les poids ; tenir tes notes dans ce dossier.
- Commandes autorisées : `uv run slm-weight-editing run-batch` et `session-status` comme ci-dessus, `uv run python <script>.py`, lecture de fichiers de ce dossier, de `work/` et du journal.

## Ce qui est interdit

- Tout accès web, MCP, sous-agent ou skill.
- Toute lecture hors de ce dossier, de `work/`, du journal de séance et des données train : en particulier `heldout/`, le banc de mesure, la documentation, les specs, les tests et l'historique git du dépôt, le cache Hugging Face et tout autre modèle publié.
- Toute écriture, tout déplacement ou toute suppression de `state.json`, `journal.jsonl`, `registry.jsonl`, `session.lock` ou d'un dossier de séance : seul `work/` t'appartient. Un verrou orphelin se signale à Anaël.
- Toute ouverture, réouverture ou copie de séance, toute modification du harnais, de l'environnement ou de ces réglages, toute installation de paquet.
- Tout gradient, toute perte, tout entraînement.

## Plafond du diff

Contrainte ferme : au plus 10 % d'éléments changés par tenseur (éditions creuses). Un diff au-delà est refusé au gel ; la séance n'est alors pas gelée et tu dois réduire ton édition.

## Fin de séance

Quand ton édition est prête, ou le budget épuisé, écris dans ce dossier `EDITION.md` : tenseurs touchés, principe de l'édition, lots utilisés et ce qu'ils ont montré. Puis arrête-toi et écris « Édition terminée ». Anaël gèle la séance lui-même ; tu ne lances pas le gel.
