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

3. Données : un fichier JSONL local par split train, une ligne par problème, `{"id": "...", "question": "..."}` (GSM8K) ou `{"id": "...", "problem": "..."}` (MATH) ; avec `"split": "train"` obligatoire sur chaque ligne (une ligne sans `split` est refusée) ; les autres champs sont ignorés. Aucun téléchargement par le harnais.

4. Journal : `models/sessions/s1/journal.jsonl`, une ligne `open` (sha256 des tenseurs et fichiers de la base), puis par lot une ligne `reserve` écrite avant la génération (le lot est décompté même si la génération plante) et une ligne `batch` (numéro, budget restant, réglages, problèmes, sorties avec `done_reason` `stop` ou `length`, sha256 des tenseurs, des autres fichiers de `work/` et du fichier de données). Le checkpoint de base ne peut pas venir d'une séance, et un identifiant déjà ouvert (registre `models/sessions/registry.jsonl`) ne se rouvre pas, même après suppression de son dossier.

5. Plafond d'édition : au gel, un tenseur dont plus de 10 % des éléments changent est refusé (décision du 2026-11-04 : la consigne de séance impose des éditions creuses ; une projection de rang 1 sur une matrice entière ne passe pas, un tenseur de 9 éléments ou moins ne peut changer d'aucun élément). `session-open` et `session-status` le rappellent.

## Geler une séance et publier le diff

1. Gel : pose `frozen` (définitif, le harnais refuse ensuite tout lot), compare `work/` à l'original de la séance (ses sha256 doivent égaler ceux de la ligne `open` du journal, sinon refus sans rien écrire) et écrit dans `models/sessions/s1/freeze/` :

   ```
   uv run slm-weight-editing session-freeze --session s1 --original models/hf/qwen3.5-base
   ```

   - `diff.json` (format `slm-weight-editing/edit-diff-v1`) : identifiant de séance, révision de base, budget et lots utilisés, sha256 du journal, sha256 du fichier de valeurs et, par tenseur modifié, nom, forme, dtype, nombre d'éléments et d'éléments changés, sha256 des octets bruts avant et après ; plus la liste des fichiers non safetensors de `work/` qui diffèrent de la base (signalés, non rejoués) ;
   - `diff.bin` : par tenseur, les indices plats des éléments changés (u4 ou u8) puis leurs valeurs après édition en octets bruts du dtype d'origine (rejeu bit à bit, aucun arrondi) ;
   - `journal.jsonl` : version publique du journal au gel, dont le sha256 figure dans `diff.json` (chemins locaux réécrits : relatifs au dossier courant s'ils sont dessous, sinon `.../<nom>` ; dossier personnel remplacé par `~`) ; le journal brut reste dans la séance, non publié, et reçoit un événement `freeze`.

   Plafond : un tenseur dont plus de 10 % des éléments changent est refusé (le diff reviendrait à publier le tenseur complet) ; la séance n'est alors pas gelée. Tout le gel tient le verrou de séance : si un lot tourne encore, il réessaie une dizaine de secondes puis échoue clairement. Si l'écriture du diff échoue après la pose du gel, relancer `session-freeze`.

2. Rejeu sur l'original (sha256 avant et après vérifiés, refus sinon ; les autres fichiers de l'original sont recopiés) :

   ```
   uv run slm-weight-editing apply-diff --diff models/sessions/s1/freeze/diff.json --original models/hf/qwen3.5-base --out models/rejoue/s1
   ```

3. Report dans le GGUF Ollama, sans modifier le checkpoint rejoué (les sha256 attendus viennent du diff, `--tag` est obligatoire) ; `ollama create` se lance ensuite par Anaël :

   ```
   uv run slm-weight-editing report-gguf --checkpoint models/rejoue/s1 --base-checkpoint models/hf/qwen3.5-base --edit-diff models/sessions/s1/freeze/diff.json --tag qwen3.5-s1:0.8b
   ```

   Un diff dont `files_changed` n'est pas vide (config ou tokenizer modifiés dans `work/`) est refusé : ces fichiers ne sont rejoués ni par `apply-diff` ni par `report-gguf`. Sans `--edit-diff`, `report-gguf` garde son comportement (édition du MIT). Le `diff.json` de report décrit la séance (`edit.session_id`, sha256 du diff et du journal), jamais l'édition du MIT.

4. Comparer à une autre édition sans rejouer sur disque : `compare-edits --original <base> --diff-a models/sessions/s1/freeze/diff.json --edit-b <checkpoint>` (`--diff-a` remplace `--edit-a`, `--diff-b` remplace `--edit-b`).

## Préparer et auditer une séance

La séance se déroule dans un espace hors du dépôt, protégé par trois couches : réglages Claude Code (l'agent ne voit que l'espace et n'a que trois commandes), lanceur `run-script` (seule voie pour exécuter du Python, sous crochet d'audit), audit de la transcription après la séance. Conception, contournements couverts et limites : `_bmad-output/specs/spec-slm-weight-editing/story-5-protocole-seance.md`.

1. Anaël crée l'espace et ouvre la séance, depuis le dépôt (dossier neuf, hors du dépôt et de son dossier parent) :

   ```
   uv run slm-weight-editing session-workspace --out %USERPROFILE%\seances\s1 --session s1 --budget 25 --base-checkpoint <checkpoint> --base-revision <révision> --data <gsm8k-train.jsonl> [--data <math-train.jsonl>]
   ```

   L'espace reçoit `.claude/settings.json` (mode `dontAsk`, lectures et écritures bloquées hors de l'espace, liste blanche `uv run slm-weight-editing run-batch|session-status|run-script` (sans chemin du dépôt), `env.UV_PROJECT` (le dépôt, lu par uv comme `--project`), règles `deny` des commandes de lecture et d'exécution génériques, du web, des sous-agents et de MCP, `claudeMdExcludes` des CLAUDE.md et AGENTS.md parents), `CONSIGNE.md` et `CLAUDE.md` rendus pour la séance, `scripts/`, `notes/`, `data/` (copie des fichiers train) et `sessions/s1/` (séance du harnais, `work/` compris). Le chemin du dépôt n'est écrit que dans `env.UV_PROJECT` de `settings.json` et dans `workspace.json`. Pourquoi pas `--project <dépôt>` dans les commandes : avec `blockReadsOutsideWorkingDirectories`, ce chemin hors de l'espace est traité comme un accès hors des répertoires de travail, et le mode `dontAsk` refuse alors les trois commandes autorisées ; `deny` interdit `UV_`, `--project` et `--directory` pour que l'éditeur ne rebranche pas uv ailleurs.
2. Lancement depuis l'espace, sans serveur MCP : `claude --strict-mcp-config`. Avant le premier message, vérifier `/permissions` (mode `dontAsk`, les six règles `allow` de commande, les `deny`) et `/memory` (seul le `CLAUDE.md` de l'espace, plus le `CLAUDE.md` personnel de `~/.claude/`, que `claudeMdExcludes` ne couvre pas). Premier message : « Lis CONSIGNE.md et suis-la. »
   L'espace reçoit aussi `workspace.json` (séance, dossier des séances, `data/`) : lancés depuis l'espace, `run-batch`, `session-status` et `run-script` refusent une autre séance, un autre `--sessions-root` et des données hors de `data/`.
3. Pendant la séance, l'éditeur exécute ses scripts par `run-script scripts/<nom>.py --session s1 --sessions-root sessions` : `python -I` isolé, environnement purgé, lecture de `work/`, `scripts/` et `data/` seulement, écriture de `work/` et `scripts/`, bibliothèque standard, numpy et `safetensors.numpy` (version Python pure, bfloat16 en `uint16`) ; tout autre import, sous-processus, socket, ctypes, lien ou fichier hors de ces dossiers arrête le script (code 3). Chaque exécution tient le verrou de séance, ne consomme pas de lot et ajoute au journal un événement `script` (sha256, copie dans `sessions/s1/scripts/`, code de retour, sortie tronquée à 20 000 octets).
4. Gel par Anaël (`session-freeze --session s1 --sessions-root %USERPROFILE%\seances\s1\sessions ...`), puis audit de la transcription (`~/.claude/projects/<projet de l'espace>/<id>.jsonl`, sous-agents compris) :

   ```
   uv run slm-weight-editing audit-session <transcription>.jsonl --workspace %USERPROFILE%\seances\s1 --session-dir %USERPROFILE%\seances\s1\sessions\s1 --reference-from <checkpoint de la révision publiée> --base-revision <révision> --write-reference <référence>.json
   ```

   `--workspace`, `--session-dir` et la référence (`--reference` ou `--reference-from`) sont obligatoires ; la base de la ligne `open` est comparée à la révision publiée (tenseurs et autres fichiers). Code 0 : aucune violation ; code 1 : violations, une par ligne (`fichier:ligne · règle · outil · détail`) ; code 2 : entrée illisible. Toute commande hors liste blanche, toute lecture hors de l'espace, toute transcription vide et tout script refusé par `run-script` font échouer l'audit ; les copies des scripts lancés sont auditées comme le reste.

## Poste de séance avec GPU

Par défaut (CI, autres postes), torch vient de PyPI (Windows, macOS) ou de l'index CPU (Linux) : aucune roue CUDA. Sur le poste de séance, l'extra optionnel `cuda` installe torch CUDA 13.2 (roue Windows d'environ 2 Go, index `pytorch-cu132`) :

```
uv sync --extra cuda
uv run --extra cuda python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

L'extra `cuda` ne vise que Windows (roue cu132 Windows seulement) : sous macOS, `uv sync --extra cuda` échoue, sous Linux il installe torch CPU, et `session-workspace --cuda` y est refusé.

Toute commande `uv run` du poste porte `--extra cuda` : sans lui, `uv run` resynchronise l'environnement et retire la roue CUDA. L'espace de séance se crée avec `--cuda` : les règles `allow`, la consigne et le `CLAUDE.md` de séance écrivent alors `uv run --extra cuda slm-weight-editing ...`, et l'audit accepte cette forme (et aucune autre option d'`uv`) ; il lit aussi `env.UV_PROJECT` de `.claude/settings.json` et exige qu'il désigne le dépôt attendu (`--repo`, défaut ce dépôt).

```
uv run --extra cuda slm-weight-editing session-workspace --cuda --out %USERPROFILE%\seances\s1 --session s1 ...
```

`run-batch --device auto|cpu|cuda` (défaut `auto` : CUDA si disponible, sinon CPU ; `cuda` sans GPU est refusé). L'appareil est fixé par le premier lot de la séance (même un lot qui plante) : un lot sur un autre appareil est refusé sans consommer de budget, car le décodage glouton peut différer entre CPU et GPU. Chaque lot journalise l'appareil, le nom du GPU et les versions de torch et de CUDA ; le dtype est celui du checkpoint (bf16 si le modèle l'est), la mémoire GPU est libérée après chaque lot. Sur CUDA, avant de réserver le lot, le harnais compare la taille des poids au dtype chargé (plus 10 % et 1 Gio de marge) à la mémoire libre du GPU : s'ils ne tiennent pas, le lot est refusé sans être consommé, sans conversion implicite d'un checkpoint fp32 en bf16 (elle changerait le décodage).
