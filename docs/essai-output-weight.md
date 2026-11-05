# Essai : Ollama charge-t-il un `output.weight` séparé pour qwen3.5 ?

Préalable de CAP-10 (`_bmad-output/specs/spec-harnais-v2/SPEC.md`, section Assumptions) : avant
d'éditer seulement la sortie (`lm_head` détaché), vérifier qu'Ollama 0.40.14 charge pour
qwen3.5:0.8b un GGUF qui a un `output.weight` copié de `token_embd.weight`. Si l'essai échoue,
**CAP-10 sort de s2**.

## 1. Produire le GGUF

Depuis le blob officiel du magasin d'Ollama (lu en lecture seule) :

```
uv run slm-weight-editing gguf-add-output --model qwen3.5:0.8b --out models/qwen3.5-0.8b-output.gguf
```

(ou `--gguf <chemin>` pour un GGUF donné, sans Modelfile ; `--ollama-models` si le magasin
n'est pas celui par défaut.) La commande affiche le type de `token_embd` trouvé (« copie
d'octets » s'il est Q8_0, « requantifié » s'il est F32, F16 ou BF16 ; tout autre type, Q4_K ou
Q6_K par exemple, est refusé) et les éventuelles clés de métadonnées contenant « tie » (aucune
n'est modifiée : llama.cpp déduit la liaison de l'absence de `output.weight`). La commande
refuse un GGUF qui a déjà un `output.weight`, une sortie ou un Modelfile existants, l'entrée
elle-même ou un chemin dans le magasin d'Ollama.

Avec `--model`, elle écrit aussi deux Modelfiles qui reprennent à l'identique le gabarit, le
système, le renderer, le parser, les paramètres (`presence_penalty`, `top_k`, `top_p`…) et la
licence de l'officiel, lus dans son manifeste :

- `models/qwen3.5-0.8b-output.Modelfile` : `FROM` le GGUF d'essai ;
- `models/qwen3.5-0.8b-output-temoin.Modelfile` : `FROM` le blob GGUF officiel (témoin).

Les Modelfiles générés sous Windows avant le correctif « LF » étaient écrits en CRLF (y compris
dans les blocs `TEMPLATE`) : à régénérer. La commande refusant une sortie ou un Modelfile
existants, supprimer d'abord le GGUF d'essai et les deux Modelfiles, puis la relancer. Les fins de
ligne sont désormais en LF sous tout OS ; un `\r` présent dans le gabarit de l'officiel est
conservé tel quel.

Les deux modèles ne diffèrent donc que par `output.weight`. Ne pas écrire le Modelfile à la main :
un Modelfile réduit à `FROM` perd les paramètres de l'officiel (dont `presence_penalty`, actif
même à température 0) et l'écart de réponses ne prouverait plus rien.

## 2. Créer les modèles (Anaël)

Dans un terminal à part :

```
ollama create qwen35-temoin -f models/qwen3.5-0.8b-output-temoin.Modelfile
ollama create qwen35-output-test -f models/qwen3.5-0.8b-output.Modelfile
```

Le témoin réutilise le blob officiel (même empreinte) : pas de place disque en plus. Si
`ollama create` ou le chargement (`ollama run`) du modèle d'essai échoue : noter le message et
conclure.

## 3. Comparer aux réponses du modèle officiel

Prompts publics simples (température 0, graine 42, `num_predict` 64) :

1. Quelle est la capitale de la France ?
2. Écris un haïku sur la mer.
3. Combien font 17 fois 23 ?
4. Traduis « bonjour tout le monde » en anglais.
5. Cite trois couleurs primaires.
6. Explique en une phrase ce qu'est un nombre premier.
7. Donne le prochain mot de la suite : lundi, mardi, mercredi,
8. Complète : « Le chat dort sur le »

Pour chaque prompt et chacun des trois modèles (`qwen3.5:0.8b`, `qwen35-temoin`,
`qwen35-output-test`) :

```
curl http://localhost:11434/api/generate -d '{"model":"MODELE","prompt":"PROMPT","stream":false,"options":{"temperature":0,"seed":42,"num_predict":64}}'
```

Garder `response`, `total_duration` et `eval_count`. Contrôles préalables : l'officiel relancé
deux fois répond pareil (déterminisme), et le témoin répond comme l'officiel (Modelfile fidèle) ;
si le témoin diffère de l'officiel, l'essai n'est pas concluant : corriger le Modelfile avant
de juger. **Critère : `qwen35-output-test` répond mot pour mot comme le témoin** pour les 8
prompts.

## 4. Taille et temps

- Taille : `ollama list` ; environ 270 Mo de plus attendus pour le modèle d'essai.
- Temps : médiane de `total_duration` sur les 8 prompts, pour chaque modèle ; noter l'écart.

## 5. Consigner le résultat

Dans `docs/resultats/essai-output-weight.md` (à créer) : date, version d'Ollama, type de
`token_embd` (sortie de la commande), témoin identique à l'officiel oui/non, essai identique au témoin oui/non par prompt, tailles, temps, messages
d'erreur éventuels, conclusion **« CAP-10 maintenu » ou « CAP-10 sort de s2 »**. Ne pas y
mettre de question du banc.
