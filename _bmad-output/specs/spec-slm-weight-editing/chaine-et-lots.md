# Chaîne de traitement et lots

## Chaîne

| Étape | Contenu | Capacité | Qui lance |
|---|---|---|---|
| 1. Original | safetensors HF, révision figée, et GGUF Ollama officiel (manifeste local : quantification, renderer, parser, paramètres) | – | – |
| 2. Séance d'édition | Claude Code édite les safetensors par le harnais | CAP-5, CAP-6 | Anaël ouvre la séance |
| 3. Diff | tenseurs touchés, sha256 avant et après, script d'application, journal | CAP-2 | agent |
| 4. Report | tenseurs édités seuls, requantifiés dans le type d'origine (Q8_0 pour `qwen3.5:0.8b`) ; contrôle de fidélité d'abord | CAP-1 | agent |
| 5. Modèle Ollama | Modelfile repris de l'officiel ; `ollama create` | CAP-3 | Anaël |
| 6. Mesure | campagne du banc, original et édité ensemble | CAP-4 | Anaël (`precheck`, `generate`) |

## Lots

- **Lot 0, étalon** : étapes 4 à 6 sur l'édition publiée du MIT (`reference-mit.md`), sans séance. Valide la chaîne. Campagne `wsp-20261104`.
- **Lot 1, réplication** : séance Claude sur Qwen3.5-0.8B depuis l'original, sans voir l'édition du MIT, puis comparaison des deux éditions (CAP-7).
- **Lot 2, autres cibles** : `cibles-banc.md` (CAP-8).

## Campagne du lot 0 : `wsp-20261104`

Mêmes réglages que `lfm3-20261029` (relevé sur `origin/main` 1198ec2 de `slm-quality-bench`) ; seuls l'identifiant et les modèles changent :

- `set: set-v3` ; `ollama_version: "0.40.13"` ;
- `request: {temperature: 0, seed_base: 0, k: 1}` ;
- plafonds think_off 4 096 (RAG 2 048), think_on 4 096 (reasoning 4 096, RAG 2 048) ; paliers `num_ctx` 4 096, 8 192, 16 384, 32 768 ;
- juge `opus` ; `gpu_check` qui tolère `Overwolf.exe`.

Modèles : `qwen3.5:0.8b` (famille `qwen`, Q8_0, digest `f3817196d142` relevé sous Ollama 0.40.5, pistes think_off et think_on, `think_off: false`, `think_on: true`) et le modèle édité `qwen3.5-wsp:0.8b`, mêmes pistes et mêmes valeurs think. L'original est remesuré sous 0.40.13 : la comparaison ne s'appuie pas sur `full-20261018` (0.40.5).
