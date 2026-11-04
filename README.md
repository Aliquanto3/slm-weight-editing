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

## Licence

Code sous Apache 2.0. Les modèles édités gardent la licence de leur modèle d'origine.
