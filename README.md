# slm-weight-editing

Corriger les défauts de petits modèles de langage en éditant directement leurs poids, sans entraînement, avec un LLM éditeur : Claude Opus 5.5 dans Claude Code.

> **État : initialisation.** Aucun modèle édité publié pour l'instant. Cadrage dans [`docs/CADRAGE.md`](docs/CADRAGE.md).

## Idée

En novembre 2026, une équipe du MIT CSAIL a confié les poids de Qwen3.5-0.8B à un LLM de pointe, avec une session Python, un budget de 50 lots d'inférence et des exercices d'entraînement, sans gradient. Le LLM a trouvé la direction qui fait boucler le raisonnement du modèle (couches 15 et 19) et l'a retirée de 0,016 % des poids. Les réponses coupées au plafond de tokens sont passées de 93 % à 33 % ([billet](https://rikiyat.github.io/blog/weight-space-programming/), [modèle](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)).

Ce dépôt reproduit la démarche avec Claude, puis l'applique à d'autres SLM (Granite 4.0 350M et des modèles jusqu'à 2 Md de paramètres) et à d'autres défauts que les boucles.

## Différences avec l'étude d'origine

- **Mesure indépendante** : chaque modèle édité passe dans [slm-quality-bench](https://github.com/Aliquanto3/slm-quality-bench), un banc en français, contre son original, pour mesurer aussi les régressions.
- **Format embarqué** : l'édition est reportée dans le GGUF Ollama officiel du modèle, avec la même quantification, pour que seule l'édition diffère.
- **Protocole vérifiable** : budget de passes compté par le harnais, jeux d'évaluation hors de portée de l'agent éditeur, diff des poids publié avec ses empreintes sha256.

## Usage : reporter l'édition du MIT dans le GGUF Ollama (lot 0)

L'outil remplace dans le GGUF officiel de `qwen3.5:0.8b` (Q8_0) les deux tenseurs édités par le MIT (`down_proj` des couches 15 et 19, devenus `blk.15.ffn_down.weight` et `blk.19.ffn_down.weight`), requantifiés en Q8_0. Il ne lance ni Ollama ni aucun téléchargement : il lit le magasin local d'Ollama (`~/.ollama/models`, ou `OLLAMA_MODELS`, ou `--ollama-models`) en lecture seule.

1. Anaël télécharge dans `models/`, ignoré par git, le checkpoint édité et sa base (safetensors) ; la base sert à contrôler les originaux des deux tenseurs visés et donne leur sha256 avant édition :

   ```
   uvx --from huggingface_hub hf download RikiyaT/Qwen3.5-0.8B-weight-space-programming --local-dir models/hf/qwen3.5-wsp
   uvx --from huggingface_hub hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir models/hf/qwen3.5-base
   ```

   Depuis le dépôt slm-weight-editing ; `uvx` lance la CLI `hf` sans l'installer (sinon `uv tool install huggingface_hub`).

2. Report, avec `qwen3.5:0.8b` déjà présent dans Ollama (`ollama pull qwen3.5:0.8b` au besoin) :

   ```
   uv run slm-weight-editing report-gguf --checkpoint models/hf/qwen3.5-wsp --base-checkpoint models/hf/qwen3.5-base
   ```

   Options : `--ollama-models` (magasin), `--out` (défaut `models/qwen3.5-wsp-0.8b`), `--intermediate f16` (export F16 puis quantifié, si le contrôle de fidélité échoue en f32).

   Dans l'ordre, l'outil :
   - vérifie les sha256 des octets bf16 bruts des deux tenseurs publiés, et refuse le report s'ils diffèrent (code 3) ;
   - fait le contrôle de fidélité : requantifier les tenseurs `down_proj` non édités du checkpoint et les originaux des deux tenseurs visés (base) doit redonner exactement leurs octets dans le GGUF officiel. En cas d'échec, il n'écrit aucun GGUF, rend le code 2 et décrit l'écart (tenseur, blocs, échelles, octets) dans `models/qwen3.5-wsp-0.8b/fidelite.json` ;
   - écrit dans `models/qwen3.5-wsp-0.8b/` le GGUF édité (identique à l'officiel hors des deux tenseurs), le `Modelfile` (renderer, parser, paramètres et licence de l'officiel) et `diff.json` (tenseurs touchés, sha256 avant et après côté HF et côté GGUF, révision de base, commande d'application).

3. Anaël crée le modèle dans un terminal à part :

   ```
   ollama create qwen3.5-wsp:0.8b -f models/qwen3.5-wsp-0.8b/Modelfile
   ```

   Puis `ollama show qwen3.5-wsp:0.8b` doit donner le même renderer, le même parser et les mêmes paramètres que `qwen3.5:0.8b`.

## Licence

Code sous Apache 2.0. Les modèles édités gardent la licence de leur modèle d'origine.

## Comparer deux éditions

```
uv run slm-weight-editing compare-edits --original <orig> --edit-a <edition-a> --edit-b <edition-b> \
  [--label-a Claude --label-b MIT] [--summary a/summary.json [--summary b/summary.json]] \
  [--tag-a <tag> --tag-b <tag>] [--out models/comparaison-editions/rapport.md]
```

Les checkpoints sont des fichiers ou dossiers safetensors du même original (mêmes noms, formes et dtypes, sinon erreur et code 1). Le rapport Markdown et un JSON des chiffres (même nom, extension `.json`) donnent les tenseurs touchés (communs, propres à chacun, nombre et part des poids modifiés), le cosinus des deltas et celui de leurs directions principales (SVD, valeur absolue ; n/a pour un tenseur 1-D comme un biais ou une norme). Avec un ou deux `summary.json` de `slm-quality-bench`, il ajoute l'écart de score et de taux tronqué par piste et par catégorie, avec un intervalle à 95 % (Wilson, puis Newcombe pour l'écart, modèles traités comme indépendants). Un seul résumé contenant les deux modèles se sélectionne par `--tag-a` et `--tag-b`. Une piste ou catégorie présente dans un seul résumé est listée « absente de A » (ou de B). Le rapport ne reprend ni question, ni réponse, ni chemin.

## Séance d'édition : harnais à budget de lots

Le harnais est la seule voie d'exécution du modèle pour l'agent éditeur (protocole dans `AGENTS.md`).

1. Ouverture, budget de 10, 25 ou 50 lots fixé une fois pour toutes :

   ```
   uv run slm-weight-editing session-open --session s1 --budget 25 --base-checkpoint models/hf/qwen3.5-base --base-revision 2fc06364715b967f1860aea9cf38778875588b17
   ```

   La séance vit dans `models/sessions/s1/` : `state.json` (budget, lots consommés, gel, base), `work/` (copie du checkpoint que l'agent modifie) et `journal.jsonl`. Une réouverture ou un budget hors de 10, 25 et 50 est refusé.

2. Lot d'inférence, compté contre le budget :

   ```
   uv run slm-weight-editing run-batch --session s1 --dataset gsm8k --split train --data data/gsm8k/train.jsonl --start 0 --count 64
   uv run slm-weight-editing session-status --session s1
   ```

   Au plus 512 problèmes par lot. Génération gloutonne, `max_new_tokens` 4096 (non réglable), sous `torch.inference_mode()`, poids rechargés depuis `work/` à chaque lot. Refus (code 2) : budget épuisé, séance gelée, jeu autre que `gsm8k` ou `math`, split autre que `train`, chemin sous `heldout/` (même par `..` ou lien), nom de fichier sans « train ».

3. Données : un fichier JSONL local par split train, une ligne par problème, `{"id": "...", "question": "..."}` (GSM8K) ou `{"id": "...", "problem": "..."}` (MATH) ; les autres champs sont ignorés, sauf `split`, qui doit valoir `train`. Aucun téléchargement par le harnais.

4. Journal : `models/sessions/s1/journal.jsonl`, une ligne `open`, puis une ligne `batch` par lot (numéro, budget restant, réglages, problèmes, sorties avec `done_reason` `stop` ou `length`, sha256 des tenseurs chargés et du fichier de données).
