---
id: STORY-1-report-gguf
status: review
parent: SPEC.md
capabilities: [CAP-1, CAP-2, CAP-3]
---

# Story 1 : report de l'édition du MIT dans le GGUF officiel de qwen3.5:0.8b

## Livré

- Commande `slm-weight-editing report-gguf --checkpoint <dossier safetensors>` (options `--base-checkpoint`, `--ollama-models`, `--model`, `--tag`, `--out`, `--intermediate f32|f16`). Sorties sous `models/qwen3.5-wsp-0.8b/` : GGUF édité, `Modelfile`, `diff.json`, `fidelite.json`.
- Ordre : sha256 des octets bf16 bruts des deux tenseurs de RikiyaT (code 3 si faux), puis contrôle de fidélité (code 2 en échec, aucun GGUF écrit, écart par tenseur dans `fidelite.json` : blocs, échelles, quanta, octets, premiers blocs avec ligne et colonne), puis copie du GGUF officiel avec réécriture des seules plages d'octets visées (fichier `.partial` renommé à la fin).
- Témoins du contrôle de fidélité : tous les `down_proj` non visés du checkpoint édité ; en plus, les originaux des tenseurs visés si la base est fournie. Aucun témoin : échec.
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
