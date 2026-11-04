---
id: SPEC-slm-weight-editing
companions:
  - chaine-et-lots.md
  - reference-mit.md
  - cibles-banc.md
sources:
  - ../../../docs/CADRAGE.md
---

> **Contrat de référence.** Cette spec et les fichiers de `companions:` forment le contrat complet de ce qu'il faut construire, tester et valider. Les sources du frontmatter servent à la traçabilité.

# Édition directe des poids de SLM

## Why

Vision et opportunité : vérifier qu'un LLM éditeur, Claude Opus 5.5 dans Claude Code, corrige un défaut mesuré d'un petit modèle de langage en éditant ses poids, sans entraînement ni gradient. La publication du MIT du 2026-11-02 (`reference-mit.md`) montre la faisabilité avec un autre éditeur sur Qwen3.5-0.8B, sans mesure de régression hors de ses benchmarks. Le banc `slm-quality-bench` relève le même défaut sur nos cibles (`cibles-banc.md`) : `qwen3.5:0.8b` tronque 72,7 % de ses réponses think_on au plafond. Le gain doit être mesuré par ce banc indépendant, régressions comprises.

## Capabilities

- **CAP-1** — report d'une édition dans le GGUF officiel
  - **intent:** Remplacer dans le GGUF Ollama officiel les seuls tenseurs modifiés d'un checkpoint Hugging Face édité, requantifiés dans leur type d'origine.
  - **success:** Le contrôle de fidélité passe (requantifier chaque tenseur original redonne exactement ses octets dans le GGUF officiel), et le GGUF produit ne diffère de l'officiel que par les tenseurs visés.
- **CAP-2** — diff vérifiable d'une édition
  - **intent:** Publier chaque édition sous une forme vérifiable : tenseurs touchés, sha256 avant et après, script d'application, journal de la séance.
  - **success:** Rejouer le diff sur l'original redonne les sha256 publiés ; au lot 0, les sha256 des deux tenseurs de RikiyaT (`reference-mit.md`).
- **CAP-3** — modèle Ollama du modèle édité
  - **intent:** Fournir un Modelfile qui reprend le renderer, le parser et les paramètres du manifeste officiel et pointe vers le GGUF édité.
  - **success:** Après `ollama create` par Anaël, `ollama show` du modèle édité donne le même renderer, le même parser et les mêmes paramètres que l'officiel.
- **CAP-4** — mesure appariée dans le banc
  - **intent:** Mesurer l'original et l'édité dans une même campagne `slm-quality-bench`, avec la même version d'Ollama et les mêmes réglages.
  - **success:** Le `summary.json` de la campagne donne pour les deux modèles les scores et le taux tronqué par catégorie et par piste, avec intervalles de confiance.
- **CAP-5** — harnais d'édition (lot 1)
  - **intent:** Le harnais est la seule voie d'exécution du modèle pour l'agent éditeur ; il compte et journalise chaque lot d'inférence contre un budget fixé avant la séance.
  - **success:** Un lot au-delà du budget est refusé, et le journal liste chaque lot (problèmes, réglages, sorties).
- **CAP-6** — séance d'édition Claude (lot 1)
  - **intent:** Opus 5.5 édite Qwen3.5-0.8B depuis l'original, sans voir l'édition du MIT.
  - **success:** La séance produit un diff gelé (CAP-2) dans son budget, et son journal ne montre ni accès web ni lecture des journaux du MIT.
- **CAP-7** — comparaison de deux éditions (lot 1)
  - **intent:** Comparer deux éditions d'un même modèle par les tenseurs touchés, le cosinus de leurs directions et leurs scores au banc.
  - **success:** Un rapport chiffré compare l'édition Claude à l'édition du MIT sur ces trois axes.
- **CAP-8** — autres cibles (lot 2)
  - **intent:** Appliquer la chaîne à d'autres modèles et défauts relevés par le banc.
  - **success:** Au moins une cible autre que `qwen3.5:0.8b` est éditée et mesurée par la même chaîne.

## Constraints

- Aucun gradient, aucune perte, aucun optimiseur, aucun entraînement : `torch.inference_mode()` partout ; seulement lecture, analyse et écriture directe des poids.
- L'agent éditeur n'exécute le modèle que par le harnais ; budget de 10, 25 ou 50 lots (jusqu'à 512 problèmes par lot) fixé avant la séance.
- Données d'entraînement seulement (splits train). Jeux d'évaluation dans `heldout/` ou hors dépôt, jamais lus par l'agent éditeur, pas plus que le banc ; l'évaluation tourne après le gel de l'édition, par une commande à part.
- Pas de contamination : pendant une séance, ni web ni journaux du MIT. La coupure d'Opus 5.5 (juin 2026) précède l'étude.
- Plafond de génération du harnais aligné sur le banc : 4 096 tokens, température 0.
- Report : seuls les tenseurs édités sont remplacés, dans le type d'origine. Si le contrôle de fidélité échoue, le report s'arrête sans écrire de GGUF et rapporte l'écart (tenseur, blocs, échelles, octets) ; Anaël décide de la suite.
- Aucun Ollama lancé par un agent : `ollama create` et les campagnes du banc se lancent par Anaël dans un terminal à part. Lire le magasin local d'Ollama n'est pas un lancement.
- Dépôt public : ni poids, ni GGUF, ni données, ni sorties volumineuses, ni question privée du banc, ni extrait de réponse dans un commit.
- `slm-quality-bench` est en lecture seule depuis ce dépôt ; ses changements passent par une PR dans son propre dépôt.
- Tests sans modèle réel ni téléchargement : tenseurs synthétiques.

## Non-goals

- Fine-tuning ou entraînement, LoRA compris : hors de ce dépôt.
- Séance d'édition au lot 0 : il reporte l'édition publiée du MIT.
- Reproduire les chiffres du MIT (32 768 tokens, température 0,6) : on mesure au budget de déploiement du banc.
- Corriger des manques de connaissance : l'édition retire un obstacle, elle n'ajoute pas de capacité.
- Défauts que le banc ne mesure pas encore par réponse (langue, format, fuite du raisonnement) : hors périmètre tant qu'ils n'y sont pas instrumentés.

## Success signal

- Une édition réussit si le défaut visé recule nettement sur le banc, si aucune catégorie ne régresse au-delà de son intervalle de confiance et si l'édition se reproduit depuis le diff publié, vérifiée par sha256.
- Lot 0 : dans `wsp-20261104`, le modèle édité, construit par CAP-1 depuis les tenseurs vérifiés de RikiyaT, tronque nettement moins que les 72,7 % think_on de `qwen3.5:0.8b`, sans régression hors intervalle.

## Assumptions

- L'outil de report vit dans ce dépôt (CLI Python) et lit le GGUF officiel par le manifeste local d'Ollama, en lecture seule.
- Le modèle édité garde la famille `qwen` au registre du banc ; son digest et ses capacités sont à confirmer par `precheck` après `ollama create`, comme pour `lfm3-20261029`.
- Le harnais exécute le modèle par transformers sur les safetensors HF, sous `torch.inference_mode()`, en décodage glouton plafonné à 4 096 tokens ; pas d'Ollama dans le harnais.
- « Seule voie d'exécution » : on ne peut pas l'imposer techniquement à un agent qui écrit du Python ; elle repose sur les réglages de la séance et sur un audit de la transcription après coup.

## Open Questions

- Quel seuil chiffré pour « recule nettement » ?
- Périmètre de taille : ≤ 2 Md (AGENTS.md) ou ≤ 2,5 Md (cadrage, `qwen3.5:2b` à 2,3 Md en cible n° 2) ?
- Cible de la séance du lot 1 : Qwen3.5-0.8B à confirmer après le résultat de `wsp-20261104` ; si l'édition du MIT n'y fait pas reculer le tronqué, revoir la cible.
- Matériel des séances : tour RTX 3050, portable `portable-rtx3060` (déclaré au banc pour les petits modèles) ou CPU ?
- Coût d'une séance Opus 5.5 en effort high : à mesurer au lot 1 avant d'élargir ; Fable 5.1 seulement après deux échecs.
- Quels défauts autres que les boucles retenir (signature mesurable sur données d'entraînement) ?
- Fine-tuning Service Public : appliquer l'édition après la fusion du LoRA puis réévaluer (l'ordre inverse risque de réintroduire le défaut) ; à confirmer.
