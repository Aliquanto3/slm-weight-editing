# Référence : « Sparks of Weight-Space Programming »

Takehi, Teoh, Zhang et Khattab, MIT CSAIL, 2026-11-02 : [billet](https://rikiyat.github.io/blog/weight-space-programming/), [journaux et diff](https://github.com/RikiyaT/lm_understand-logs) (`exp8_math/runs/p50c/edit/`, reconstruction par `exp8_math/code/apply_edit.py`), [modèle](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).

Interdit à l'agent éditeur pendant une séance (contrainte de contamination). Libre au lot 0, qui reporte cette édition.

## Protocole et édition

- Harnais : session Python sur les poids, sans GPU pour l'édition ; 10, 25 ou 50 lots d'inférence (jusqu'à 512 problèmes par lot) sur les splits d'entraînement de GSM8K et MATH ; ni gradient, ni perte, ni entraînement. Un seul éditeur testé (GPT-6 Astra).
- Édition : une direction écrite par les MLP des couches 15 et 19, retirée de 64 colonnes de `down_proj` par couche (131 072 poids, 0,016 %).

## Tenseurs publiés (lot 0)

Base : `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`. Seuls ces deux tenseurs diffèrent de la base (README du modèle, relevé du 2026-11-04) ; sha256 des octets bfloat16 bruts :

| Tenseur | sha256 après édition |
|---|---|
| `model.language_model.layers.15.mlp.down_proj.weight` | `82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53` |
| `model.language_model.layers.19.mlp.down_proj.weight` | `8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2` |

Le checkpoint HF est la version multimodale complète ; les tenseurs du modèle de langage sont identiques à l'export texte évalué par le MIT.

## Résultats déclarés

Protocole MIT : thinking on, jusqu'à 32 768 tokens, température 0,6, plusieurs tirages.

- MATH-500 de 7,9 à 48,1 % (51,0 % à la reproduction d'akhaliq) ; réponses au plafond de 93 à 33,1 % ; IFEval de 21,0 à 62,3 %.
- Aucun gain avec 10 ou 25 lots.

## Limites

- Aucune mesure de régression publiée hors de ces benchmarks.
- Échec sur Gemma 3 270M, qui boucle peu : l'édition retire un obstacle, elle n'ajoute pas de capacité.

## Travail voisin

[arXiv 2606.13705](https://arxiv.org/abs/2606.13705) : Gemma 4, boucles corrigées par l'édition de quelques neurones, scores généraux conservés.
