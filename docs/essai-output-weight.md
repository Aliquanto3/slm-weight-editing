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

(ou `--gguf <chemin>` pour un GGUF donné ; `--ollama-models` si le magasin n'est pas celui par
défaut.) La commande affiche le type de `token_embd` trouvé (« copie d'octets » s'il est Q8_0,
« requantifié » sinon : à noter dans le résultat) et les éventuelles clés de métadonnées
contenant « tie » (aucune n'est modifiée : llama.cpp déduit la liaison de l'absence de
`output.weight`). La commande refuse un GGUF qui a déjà un `output.weight`, une sortie
existante, l'entrée elle-même ou un chemin dans le magasin d'Ollama.

## 2. Créer le modèle (Anaël)

`models/Modelfile-output` (reprendre `FROM` avec le chemin du GGUF produit ; le gabarit,
le renderer et le parser de l'officiel, voir `ollama show qwen3.5:0.8b --modelfile`, doivent
être repris à l'identique) :

```
FROM ./qwen3.5-0.8b-output.gguf
```

puis, dans un terminal à part :

```
cd models
ollama create qwen35-output-test -f Modelfile-output
```

Si `ollama create` ou le chargement (`ollama run`) échoue : noter le message et conclure.

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

Pour chaque prompt et chacun des deux modèles (`qwen3.5:0.8b`, `qwen35-output-test`) :

```
curl http://localhost:11434/api/generate -d '{"model":"MODELE","prompt":"PROMPT","stream":false,"options":{"temperature":0,"seed":42,"num_predict":64}}'
```

Garder `response`, `total_duration` et `eval_count`. **Critère : réponses identiques mot pour
mot** pour les 8 prompts (relancer deux fois l'officiel pour vérifier qu'il est lui-même
déterministe).

## 4. Taille et temps

- Taille : `ollama list` ; environ 270 Mo de plus attendus pour le modèle d'essai.
- Temps : médiane de `total_duration` sur les 8 prompts, pour chaque modèle ; noter l'écart.

## 5. Consigner le résultat

Dans `docs/resultats/essai-output-weight.md` (à créer) : date, version d'Ollama, type de
`token_embd` (sortie de la commande), identiques oui/non par prompt, tailles, temps, messages
d'erreur éventuels, conclusion **« CAP-10 maintenu » ou « CAP-10 sort de s2 »**. Ne pas y
mettre de question du banc.
