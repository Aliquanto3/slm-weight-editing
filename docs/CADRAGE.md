# Cadrage

Version du 2026-11-04, mise à jour le soir même après la mesure du lot 0 (critère de succès, lot 2, matériel). Document de départ, transformé en spec dans `_bmad-output/specs/spec-slm-weight-editing/`.

## Objectif

Vérifier qu'un LLM éditeur, Claude Opus 5.5 dans Claude Code, peut corriger un défaut mesuré d'un SLM en éditant ses poids, sans entraînement. Le gain doit être mesuré par un banc indépendant, régressions comprises.

Critères de succès d'une édition :

- le défaut visé recule nettement sur le banc (par exemple le taux de réponses au plafond de tokens) ;
- aucune catégorie du banc ne régresse au-delà de son intervalle de confiance ;
- l'édition est reproductible depuis le diff publié, vérifié par sha256 ;
- pour une édition des boucles de raisonnement, le modèle édité **avec** raisonnement dépasse le score global de l'original **sans** raisonnement, et la piste sans raisonnement ne régresse pas. Sinon, le meilleur réglage du modèle reste l'original sans raisonnement, et l'édition n'a pas d'usage pratique.

Lot 0 (`wsp-20261104`, 2026-11-04) contre ce dernier critère. L'édition du MIT fait passer Qwen3.5-0.8B de 16,6 à 32,0 % avec raisonnement (+15,4 pts, p < 0,001 ; réponses tronquées de 72,9 à 43,2 %). Elle reste sous les 37,2 % de l'original sans raisonnement, et elle coûte 3,1 pts sans raisonnement (p = 0,011, dont −10 pts en abstention). C'est la barre à franchir pour la séance de Claude.

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
  - **Séance s1, exploratoire** (gelée le 2026-11-05, décision d'Anaël le même jour) : 2 lots du harnais sur 25. L'édition ne touche que la ligne de `</think>` dans `embed_tokens` (1 021 éléments), pour remonter sa logit quand le modèle boucle. L'audit de la transcription échoue. Le point qui compte : l'éditeur a écrit un forward numpy complet et l'a fait générer par `run-script`, sans consommer de lot. Des générations ont donc échappé au budget, et s1 ne compte pas comme réplication à budget égal du MIT. Elle est quand même mesurée au banc, présentée comme exploratoire (« 2 lots du harnais, plus des générations numpy non comptées »). Elle dit si la piste de `</think>` tient. Les autres violations de l'audit sont des faux positifs (`re.compile`, décodage des octets du tokenizer, Grep sur le journal de l'espace) ou mineures.
  - **Avant la séance s2**, qui comptera pour le lot 1 : décider comment compter ou interdire la génération hors du harnais (le harnais mesure le calcul de `run-script`, ou la consigne interdit tout décodage autorégressif hors du harnais et l'audit le vérifie).
- **Lot 2, autres cibles** (décision d'Anaël, 2026-11-04) : `qwen3.5:2b` puis `minicpm5:2b`, après la mesure du lot 1. Les deux tiennent sur la RTX 3050 en bf16. Pour MiniCPM5 2B, c'est juste : environ 6,5 Gio demandés par le harnais (poids, plus 10 %, plus 1 Gio) pour environ 7 Gio libres, donc la carte doit être libre de tout autre usage.
  - `qwen3.5:2b` : même famille et même défaut que le 0.8B (53 % de réponses tronquées avec raisonnement, 15 pts perdus en activant le raisonnement). C'est le test du passage à l'échelle.
  - `minicpm5:2b` : autre famille, déjà sur la frontière de Pareto du banc sans raisonnement (61 % en 2,7 s par réponse). Avec raisonnement, 23 % de réponses tronquées et 27,6 % en raisonnement. C'est le test du transfert de la méthode.
  - Pour l'un comme l'autre, la piste sans raisonnement est à protéger : c'est elle qui les place sur la frontière ou près d'elle.
- **Lot 2 bis, `qwen3.5:4b`, reporté** : c'est le gain potentiel le plus grand (41 % de réponses tronquées, 19 pts perdus en activant le raisonnement, de 69,6 à 50,6 %). Une édition réussie pourrait le faire entrer sur la frontière avec raisonnement. Mais ses poids bf16 (environ 9 Go) ne tiennent pas dans les 8 Go de la RTX 3050, et des lots sur CPU seraient trop lents. Ce lot est à décider après la mesure des lots 1 et 2, sur un GPU loué d'au moins 24 Go (sous Linux, avec le bac à sable de Claude Code ; l'extra `cuda` est à étendre à Linux). La mesure au banc reste sur la tour, en GGUF quantisé. Écartés :
  - **l'inférence quantisée sur la tour** : le décodage glouton change et le harnais refuse toute conversion implicite ; le lot ne serait plus comparable au lot 1 ni au MIT ;
  - **l'édition directe d'un GGUF quantisé** : une édition fine (0,016 % des poids) s'écrit mal dans des blocs de 4 bits et risque de disparaître à la requantification. La chaîne reste : édition en bf16, puis requantification des seuls tenseurs touchés, validée par le lot 0.
- **Écartés du front de Pareto** : Granite 4.0 350M et LFM3 Omni 2B, qui n'ont pas de piste avec raisonnement et bouclent peu (cas de l'échec du MIT sur Gemma 3 270M) ; Gemma 4 E4B et Nemotron 3.5 Bolt E4B, qui bouclent peu (moins de 6 % de réponses tronquées) ; Qwen 3.6 35B-A3B, trop gros.

## Cibles relevées par le banc

Relevé du 2026-11-04, source : `benchmarks/quality/<campagne>/summary.json` de `slm-quality-bench` (champ `truncated` : réponse arrêtée au plafond, `done_reason == "length"`). Réglages des campagnes : température 0, plafond de 4 096 tokens (2 048 en RAG). Modèles ≤ 2,5 Md de paramètres :

| Modèle | Md | Tronqué think_off | Tronqué think_on | Score off / on | Lecture |
|---|---|---|---|---|---|
| `qwen3.5:0.8b` | 0,87 | 7,1 % | **72,7 %** | 36,9 / 15,6 | cible n° 1 : le raisonnement fait perdre 21 points |
| `qwen3.5:2b` | 2,3 | 5,7 % | **53,0 %** | 50,6 / 35,4 | cible n° 2, même famille, même défaut |
| `lfm3:1.7b-q4_K_M` | 1,73 | 12,6 % | 33,4 % | 51,1 / 52,6 | candidat, autre famille |
| `lfm2.5-thinking:1.2b` | 1,2 | – | 9,0 % | – / 47,3 | boucle peu |
| `granite4:350m` | 0,35 | 4,0 % | – | 25,1 / – | pas de mode raisonnement, boucle peu : score faible par manque de capacité, cas proche de l'échec du MIT sur Gemma 3 270M |

Au-delà de 2,5 Md, `qwen3.5:4b` (40,4 % en think_on) et `granite4.2:3b` (36,0 %) gardent le même défaut. Le banc ne mesure pas encore par réponse la langue, le format ni la fuite du raisonnement dans la réponse (critère de langue différé) : les autres défauts devront d'abord y être instrumentés.

## Questions ouvertes

- **Matériel** : tranché le 2026-11-04 : la tour RTX 3050 pour les séances, avec torch CUDA (extra `cuda`) et le harnais transformers. Un GPU loué seulement pour le lot 2 bis.
- **Coût** : une séance Opus 5.5 en effort high se mesure en dizaines de tours ; à mesurer sur le lot 1 avant d'élargir. Fable 5.1 seulement après deux échecs.
- **Défauts autres que les boucles** : retenir ceux qui ont une signature mesurable sur des données d'entraînement (langue, format, fuite du raisonnement), pas les manques de connaissance, que l'édition ne corrige pas.
- **Lien avec le fine-tuning Service Public** : appliquer l'édition après la fusion du LoRA, puis réévaluer ; l'ordre inverse risque de réintroduire le défaut.
