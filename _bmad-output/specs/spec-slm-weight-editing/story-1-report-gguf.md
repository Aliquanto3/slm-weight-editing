---
id: STORY-1-report-gguf
status: review
parent: SPEC.md
capabilities: [CAP-1, CAP-2, CAP-3]
---

# Story 1 : report de l'édition du MIT dans le GGUF officiel de qwen3.5:0.8b

## Livré

- Commande `slm-weight-editing report-gguf --checkpoint <dossier safetensors> --base-checkpoint <base>` (base obligatoire depuis la passe 2 de revue ; options `--ollama-models`, `--model`, `--tag`, `--out`, `--intermediate f32|f16`). Sorties sous `models/qwen3.5-wsp-0.8b/` : GGUF édité, `Modelfile`, `diff.json`, `fidelite.json`.
- Ordre : sha256 des octets bf16 bruts des deux tenseurs de RikiyaT (code 3 si faux), puis contrôle de fidélité (code 2 en échec, aucun GGUF écrit, écart par tenseur dans `fidelite.json` : blocs, échelles, quanta, octets, premiers blocs avec ligne et colonne), puis copie du GGUF officiel avec réécriture des seules plages d'octets visées (fichier `.partial` renommé à la fin).
- Témoins du contrôle de fidélité : tous les `down_proj` non visés du checkpoint édité ; et les originaux des tenseurs visés, tirés de la base (obligatoire en CLI ; `diff.json` indique `target_originals_checked`). Aucun témoin : échec.
- Q8_0 en numpy, comme `quantize_row_q8_0_ref` de ggml ; identique à `gguf.quants` (gguf-py, dépendance de dev) sur les tests.
- Modelfile : `FROM` chemin absolu du GGUF édité, `TEMPLATE`/`SYSTEM` s'ils existent, `RENDERER` et `PARSER` du blob de configuration, `PARAMETER` de la couche params, `LICENSE`. Syntaxe reprise d'un `ollama show --modelfile` local (`RENDERER qwen3.5`, `PARSER qwen3.5`).

## Relevé du GGUF officiel (lecture seule, 2026-11-04)

- Manifeste `qwen3.5:0.8b` : couches model (`sha256:afb707b6…`, 1 036 034 688 octets), license, params ; pas de couche template ni system. Renderer et parser `qwen3.5` dans le blob de configuration (`requires: 0.17.1`). Params : `presence_penalty 1.5`, `temperature 1`, `top_k 20`, `top_p 0.95`.
- GGUF v3, alignement 32, 536 tenseurs (265 F32, 195 Q8_0, 76 F16). `blk.15.ffn_down.weight` et `blk.19.ffn_down.weight` : Q8_0, dims ggml (3584, 1024), soit la forme HF (1024, 3584) inversée, sans transposition ; 3 899 392 octets chacun. 25 `ffn_down` en Q8_0 (24 couches et `mtp.layers.0`).
- Requantifier la déquantification des `ffn_down` officiels redonne leurs octets (cohérence de l'algorithme, pas un contrôle de fidélité).

## Reports

- Contrôle de fidélité réel : à lancer par Anaël après téléchargement du checkpoint ; s'il échoue, essayer `--intermediate f16` (export F16 puis quantifié) ou décider de la suite.
- Seule correspondance HF → GGUF prise en charge : `down_proj` → `ffn_down` (vérifiée sur le GGUF). Les autres tenseurs et `mtp.*` sont refusés.
- Champ `requires` de la configuration officielle non repris dans le Modelfile.

## Revue

### Passe 1 (2026-11-04, SHA revu 3acf3ae)

Base à jour (origin/main b718329), aucun rebase. Q8_0 vérifié contre `quantize_row_q8_0_ref` (d = amax/127, id = 1/d, roundf) et contre gguf-py, un oracle externe. Aucun constat high ni medium.

- Nit, low, second avis n° 3 : corrigé. Un échec relancé dans le même `--out` laissait le GGUF, le Modelfile et le diff de la séance précédente, et `ollama create` pouvait charger une édition périmée. Tout échec les supprime désormais ; seul `fidelite.json` de la séance reste.
- Nit, low, second avis n° 7 : corrigé. `apply` et `ollama_create` de `diff.json` contenaient des chemins `C:/Users/...` ; le dossier personnel y est remplacé par `~`, et `ollama_create` reprend le `--out` donné. Le `FROM` du Modelfile reste absolu : le Modelfile est local (`models/` ignoré) et n'est pas publié.
- Nit, low, second avis n° 1 : corrigé, le `.partial` est supprimé sur échec ou interruption (même mécanisme que le n° 3).
- Nit, low, second avis n° 2 : corrigé, le Modelfile est rendu avant l'écriture du GGUF.
- Nit, low, second avis n° 5 : corrigé, le blob officiel est vérifié avant le contrôle de fidélité.
- Nit, low, second avis n° 6 : corrigé. Un manifeste, une configuration, des params ou un en-tête safetensors illisibles donnent maintenant `OllamaStoreError` ou `SafetensorsError` (code 1), plus de traceback.
- Nit, low, second avis n° 4 : rejeté. `base.revision` décrit la référence publiée (`reference-mit.md`) ; l'édition est identifiée par les sha256 des tenseurs, et `hf_bf16_sha256_before` est mesuré quand la base est fournie.
- Nit, low, second avis n° 8 : corrigé en partie. Rejouer `apply` revérifie les sha256 publiés (étape 1) ; un test vérifie maintenant que deux séances donnent le même `output_sha256` et les mêmes entrées. Pas d'outil de rejeu à part, qui reste à décider au lot 1 (CAP-2 avec un diff gelé).
- Nit, low, revue (diagnostics Pyright) : corrigé. Un `general.alignment` non entier ou nul donne `GGUFError` et non plus un `ZeroDivisionError` ; le type des params est lu sans `str | None`.
- Nit, low, revue : corrigé. La taille en octets d'un tenseur safetensors est comparée à sa forme ; avant, un écart produisait une erreur numpy non capturée.
- Complexité : 0 constat (aucune structure qui grossit à l'exécution).
- Report (inchangé) : contrôle de fidélité sur le vrai GGUF et le vrai checkpoint, à lancer par Anaël.

### Passe 2 (2026-11-04, SHA revu 48264ad)

Base à jour (origin/main b718329), aucun rebase. Relu 3acf3ae..48264ad et la nouvelle liste du second avis.

- Important, medium, second avis n° 1 : corrigé. Sans `--base-checkpoint`, l'invocation par défaut ne requantifiait pas les originaux des deux tenseurs visés (CAP-1 : « chaque tenseur original ») et laissait `hf_bf16_sha256_before` à null (CAP-2). L'option est obligatoire en CLI, le README donne le téléchargement de la base à la révision publiée, et `diff.json` indique `fidelity.target_originals_checked`. L'API Python garde la base facultative (tests).
- Nit, low, second avis n° 2 : corrigé (doc). Le commentaire de `INTERMEDIATES` disait que le contrôle désigne le bon passage ; il dit seulement si le passage choisi redonne l'officiel, f32 et f16 coïncidant presque toujours. Pas d'essai automatique des deux chaînes : `--intermediate f16` reste le recours documenté.
- Nit, low, second avis n° 3 : corrigé. Une couche texte (licence, gabarit, système) non UTF-8 donne `OllamaStoreError` ; la CLI traduit aussi `OSError` en code 1 au lieu d'une trace. Test ajouté.
- Nit, low, second avis n° 4 : rejeté, déjà traité en passe 1 (Modelfile local sous `models/`, ignoré par git, jamais publié ; le diff publié masque le dossier personnel).
- Complexité : 0 constat.
- Report (inchangé) : contrôle de fidélité sur le vrai GGUF et le vrai checkpoint, puis `ollama create` et `ollama show`, à lancer par Anaël.
