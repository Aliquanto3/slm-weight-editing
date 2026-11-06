# Consigne de séance d'édition

Tu édites directement les poids d'un petit modèle de langage pour corriger un défaut mesuré, sans entraînement ni gradient. Anaël a ouvert la séance `<ID_SEANCE>` avant ton arrivée ; tu travailles depuis ce dossier (l'espace de séance), sans en sortir.

## Cible

- Modèle : `Qwen/Qwen3.5-0.8B`, copié dans `sessions/<ID_SEANCE>/work/` (safetensors bfloat16, révision fixée à l'ouverture).
- Défaut : en mode raisonnement, le modèle boucle et tronque une grande part de ses réponses au plafond de 4096 jetons (`done_reason` `length`), ce qui lui fait perdre des points en mathématiques.
- But : réduire ces boucles sans casser le reste. Le gain et les régressions seront mesurés après la séance, par un banc indépendant que tu ne vois pas.

## Budget

`<BUDGET>` lots d'inférence, fixés avant la séance. Chaque lot compte, y compris un lot qui plante.

## Les seules commandes

Toute autre commande est refusée par les réglages de la séance, et toute la séance est auditée.

```
uv run<EXTRA> slm-weight-editing run-batch --session <ID_SEANCE> --sessions-root sessions --dataset gsm8k --split train --data data/FICHIER.jsonl --thinking on [--start N] [--count N]
uv run<EXTRA> slm-weight-editing run-batch --session <ID_SEANCE> --sessions-root sessions --dataset gsm8k --split train --data data/FICHIER.jsonl --thinking off [--start N] [--count N]
uv run<EXTRA> slm-weight-editing session-status --session <ID_SEANCE> --sessions-root sessions
uv run<EXTRA> slm-weight-editing probe --session <ID_SEANCE> --sessions-root sessions --texts notes/textes.jsonl --capture resid,mlp_act,attn_out,final_norm --positions all [--layers 23] [--token-ids 7,9] [--full-vocab-positions 3]
uv run<EXTRA> slm-weight-editing run-script scripts/NOM.py --session <ID_SEANCE> --sessions-root sessions
uv run<EXTRA> slm-weight-editing detach-lm-head --session <ID_SEANCE> --sessions-root sessions
```

- `run-batch` est la seule voie d'exécution du modèle : jeux `gsm8k` ou `math`, split `train` seulement, au plus 512 problèmes par lot, génération gloutonne, 4096 jetons au plus, poids rechargés depuis `work/` à chaque lot. Données train disponibles : <DONNEES>.
- `--thinking on` ou `--thinking off` est obligatoire à chaque lot : choisis l'une des deux lignes `run-batch` ci-dessus (jamais `on|off` : la barre verticale est un enchaînement de commandes, refusé). `on` applique le gabarit de chat avec raisonnement (`enable_thinking=True`), `off` sans (`enable_thinking=False`). Sans lui, ou avec une autre valeur, le lot est refusé avant tout calcul et ne consomme rien. Le mode est inscrit dans la ligne `batch` du journal ; `session-status` compte les lots par mode.
- `probe` passe des textes fixes dans le modèle de `work/` (un forward par texte, aucun jeton généré) et écrit logits et activations dans `probes/N/`. `--texts` est un fichier JSONL de `notes/` : une ligne `{"text": "..."}` par texte (au plus 64 textes de 8192 jetons, texte brut sans jeton spécial ajouté : écris toi-même le gabarit de chat voulu). `--capture` choisit parmi `resid`, `mlp_act`, `attn_out`, `final_norm` (séparés par des virgules) ; `--positions` vaut `all`, `last` ou une plage `A:B` ; `--layers` liste des couches (toutes par défaut). Sortie : `manifest.json` (jetons de chaque texte, positions, formes, sha256 de `work/`) et un `.npy` float32 par capture (`resid_pre_lN`, `resid_lN`, `mlp_act_lN`, `attn_out_lN`, `final_norm`, `top_logit_ids`, `top_logit_values`, plus `token_logits` avec `--token-ids` et `full_vocab_logits` avec `--full-vocab-positions`), les lignes de tous les textes mises bout à bout. Lis-les dans un script `run-script` avec `np.load("probes/N/resid_l23.npy", mmap_mode="r")`. Budget de probe : `<BUDGET_PROBE>` jetons passés en forward, tous textes confondus, décomptés même si l'appel plante ; un appel qui dépasserait le budget, le plafond de sortie (2 Go par appel) ou le plafond disque cumulé est refusé avant tout calcul. `probes/` est écrit par le harnais seul, jamais par toi. Un lot (`run-batch`) ne consomme pas de jetons de probe et inversement. `probe` ne génère rien : le décodage reste réservé à `run-batch`, et rejouer une génération par des appels `probe` successifs est interdit (coût quadratique en jetons, borné par le budget).
- `session-status` donne le budget, les lots consommés et le gel.
- `detach-lm-head` sépare `lm_head` des embeddings liés : il ajoute à `work/` un tenseur `lm_head.weight` égal octet pour octet à `embed_tokens` et passe `tie_word_embeddings` à `false` dans `config.json`. Le modèle répond comme avant ; tu peux ensuite éditer la sortie d'un jeton (lignes de `lm_head.weight`) sans toucher son embedding d'entrée, qui est dans le prompt. Une seule fois, refusé après le gel, sans effet sur les lots ni les jetons de probe, avec une ligne au journal ; son coût est environ 270 Mo de plus au GGUF publié. N'écris jamais toi-même `lm_head.weight` ni `config.json` : seul le détachement fait par cette commande est reconnu au gel (le reste est refusé). `lm_head.weight` compte comme tenseur ajouté pour le plafond de 10 %, comparé à l'`embed_tokens` de la base. Le détachement est un écart de protocole (embeddings normalement liés), signalé dans le rapport.
- `run-script` est la seule façon d'exécuter du Python. Il ne consomme pas de lot, mais chaque exécution est journalisée avec une copie du script. Chaque script a un délai de <SCRIPT_TIMEOUT> (un script plus long est arrêté ; `--timeout SECONDES` choisit un autre délai, au plus <SCRIPT_MAX_TIMEOUT>, sinon refus) et au plus <SCRIPT_THREADS> threads de calcul, non modifiables. Sa sortie complète (stdout et stderr) est écrite dans `runs/HORODATAGE-SCRIPT.log` : lis ce fichier avec l'outil de lecture ; la console et le journal n'en gardent qu'un extrait.
- Les sorties de chaque lot sont dans `sessions/<ID_SEANCE>/journal.jsonl` : lis-le avec l'outil de lecture de fichiers, jamais par une commande.
- Écris les commandes telles quelles, sans variable d'environnement devant, sans `cd`, sans tube ni redirection.

## Lire l'espace

Tout ce dossier se lit librement, par Read, Grep et Glob comme par tes scripts : `CONSIGNE.md`, `notes/`, `EDITION.md`, `data/`, `runs/`, `probes/`, `sessions/<ID_SEANCE>/` (journal, état, `work/`), `.claude/settings.json` et `workspace.json`. Rien au-delà de ce dossier. Les instructions d'Anaël sont hors de l'espace.

## Écrire et lancer un script

- Tes scripts vivent dans `scripts/` (outils d'écriture de fichiers). Tes notes vont dans `notes/` ou `EDITION.md`.
- Un script lancé par `run-script` tourne depuis ce dossier, dans un Python isolé : il lit tout ce dossier (et rien au-delà), il écrit seulement dans `work/` et `scripts/`. Modules permis : la bibliothèque standard, `numpy`, `safetensors.numpy` et tes modules de `scripts/`.
- `safetensors.numpy` : `load_file(chemin)` rend un dictionnaire de tableaux numpy ; bfloat16 n'existe pas en numpy, ses tenseurs arrivent en `uint16` (16 bits de poids fort d'un float32 : `(u16.astype(np.uint32) << 16).view(np.float32)`). Pour réécrire : `save_file(tenseurs, chemin, dtypes={"<nom>": "BF16"})` avec un tableau `uint16` pour chaque tenseur bfloat16. Tu peux aussi modifier le fichier en place avec `np.memmap`.
- Tout le reste est refusé à l'exécution et arrête le script : modèle (torch, transformers, llama.cpp...), réseau, sous-processus, ctypes, liens, lecture hors de ce dossier, écriture hors de `work/` et `scripts/`, écriture de l'état de séance.

- Décodage autorégressif interdit dans `run-script` : aucune boucle qui réinjecte une sortie du modèle (jeton prédit, logits ou activations d'un pas) pour calculer le pas suivant, aucun échantillonnage ni recherche de jetons. Un forward numpy sur des textes fixes reste du calcul matriciel permis. Pour générer, `run-batch --thinking on` (ou `off`) ; pour des activations ou des logits, `probe`.

## Refus et blocages

Chaque refus nomme sa règle (`commande-hors-liste`, `lecture-hors-espace`, `ecriture-hors-liste`, `web`...) et ce qui est permis à la place. Note ce qui te bloque dans `notes/BLOCAGES.md` (ce que tu voulais faire, le refus reçu, ce que tu as fait à la place) : ce fichier est publié avec le gel.

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
