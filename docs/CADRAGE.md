# Cadrage

Version du 2026-11-04. Document de départ, à transformer en spec (`bmad-spec`) avant le premier développement.

## Objectif

Vérifier qu'un LLM éditeur, Claude Opus 5.5 dans Claude Code, peut corriger un défaut mesuré d'un SLM en éditant ses poids, sans entraînement. Le gain doit être mesuré par un banc indépendant, régressions comprises.

Critères de succès d'une édition :

- le défaut visé recule nettement sur le banc (par exemple le taux de réponses au plafond de tokens) ;
- aucune catégorie du banc ne régresse au-delà de son intervalle de confiance ;
- l'édition est reproductible depuis le diff publié, vérifié par sha256.

## Référence : « Sparks of Weight-Space Programming »

Takehi, Teoh, Zhang et Khattab, MIT CSAIL, 2026-11-02 ([billet](https://rikiyat.github.io/blog/weight-space-programming/), [journaux et diff](https://github.com/RikiyaT/lm_understand-logs), [modèle](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)).

- **Harnais** : session Python sur les poids, sans GPU pour l'édition ; 10, 25 ou 50 lots d'inférence (jusqu'à 512 problèmes par lot) sur les splits d'entraînement de GSM8K et MATH ; ni gradient, ni perte, ni entraînement.
- **Édition** : une direction écrite par les MLP des couches 15 et 19, retirée de 64 colonnes de `down_proj` par couche (131 072 poids, 0,016 %).
- **Résultats déclarés** : MATH-500 de 7,9 à 48,1 % (51,0 % à la reproduction d'akhaliq), réponses au plafond de 93 à 33,1 %, IFEval de 21,0 à 62,3 %. Aucun gain avec 10 ou 25 lots.
- **Limites** : aucune mesure de régression publiée hors de ces benchmarks. Échec sur Gemma 3 270M, qui boucle peu : l'édition retire un obstacle, elle n'ajoute pas de capacité. Un seul éditeur testé (GPT-6 Astra).
- **Travail voisin** : [arXiv 2606.13705](https://arxiv.org/abs/2606.13705) (Gemma 4, boucles corrigées par l'édition de quelques neurones, scores généraux conservés).

## Chaîne de traitement

1. **Original** : safetensors Hugging Face du modèle, révision figée, plus son GGUF Ollama officiel (manifeste local : quantification, renderer, parser, paramètres).
2. **Séance d'édition** : Claude Code travaille sur les safetensors par le harnais (voir ci-dessous) et produit les tenseurs modifiés.
3. **Diff** : tenseurs touchés, sha256 avant et après, script d'application, journal de la séance.
4. **Report dans le GGUF officiel** : seuls les tenseurs édités sont remplacés, requantifiés dans le type d'origine (Q8_0 pour `qwen3.5:0.8b`). Contrôle préalable : requantifier le tenseur original doit redonner les octets du GGUF officiel ; sinon la chaîne n'est pas fidèle.
5. **Modèle Ollama** : Modelfile qui reprend le renderer, le parser et les paramètres de l'officiel. `ollama create` est lancé par Anaël.
6. **Mesure** : campagne `slm-quality-bench` qui fait tourner l'original et l'édité ensemble, avec la même version d'Ollama et les mêmes réglages.

## Contrat du harnais

- Le modèle tourne uniquement par le harnais. Chaque lot d'inférence est compté et journalisé, et le budget (10, 25 ou 50 lots, comme le MIT) est fixé avant la séance.
- `torch.inference_mode()` partout ; pas d'optimiseur ni de perte.
- Données : splits d'entraînement seulement. Les jeux d'évaluation et le banc restent hors de portée de l'agent éditeur.
- **Pas de contamination** : pendant une séance, l'agent éditeur n'accède ni au web ni aux journaux du MIT. Opus 5.5 a une date de coupure en juin 2026, avant l'étude ; la séance doit garder cet avantage.
- Plafond de génération du harnais aligné sur le banc (4 096 tokens, température 0) : on corrige le défaut au budget où le modèle sera déployé.

## Lots envisagés

- **Lot 0, étalon** : reporter l'édition publiée du MIT dans le GGUF Ollama de `qwen3.5:0.8b` (étape 4), puis la mesurer contre l'original dans le banc. Ce lot valide la chaîne sans séance d'édition. La campagne est préparée dans `slm-quality-bench`.
- **Lot 1, réplication** : séance Claude sur Qwen3.5-0.8B depuis l'original, sans voir l'édition du MIT. On compare ensuite les deux éditions (tenseurs touchés, cosinus des directions) et leurs scores sur le banc.
- **Lot 2, autres cibles** : modèles et défauts relevés par le banc (section suivante).

## Cibles relevées par le banc

À compléter avec le relevé des campagnes de `slm-quality-bench` (troncatures au plafond, autres défauts mesurés).

## Questions ouvertes

- **Matériel** : la tour RTX 3050 pour les séances, ou le portable CPU 16 Go ? Un lot de 512 problèmes à 4 096 tokens est lourd sur CPU. vLLM ne tourne pas nativement sous Windows : passer par transformers, llama.cpp ou WSL.
- **Coût** : une séance Opus 5.5 en effort high se mesure en dizaines de tours ; à mesurer sur le lot 1 avant d'élargir. Fable 5.1 seulement après deux échecs.
- **Défauts autres que les boucles** : retenir ceux qui ont une signature mesurable sur des données d'entraînement (langue, format, fuite du raisonnement), pas les manques de connaissance, que l'édition ne corrige pas.
- **Lien avec le fine-tuning Service Public** : appliquer l'édition après la fusion du LoRA, puis réévaluer ; l'ordre inverse risque de réintroduire le défaut.
