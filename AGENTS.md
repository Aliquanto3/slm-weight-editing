## slm-weight-editing

Édition directe des poids de petits modèles de langage (SLM ≤ 2 Md de paramètres pour démarrer) par un LLM éditeur (Claude Opus 5.5 dans Claude Code), sans entraînement ni gradient, d'après « Sparks of Weight-Space Programming » (MIT CSAIL, 2026-11-02). Chaque modèle édité est mesuré contre son original par le banc `slm-quality-bench` (dépôt voisin, lecture seule depuis ici). Cadrage : `docs/CADRAGE.md`.

## Policy

- Dépôt public : jamais de poids, GGUF, jeux de données ni sorties de modèle volumineuses dans un commit (`data/`, `models/`, `heldout/` ignorés). Jamais de question privée ni d'extrait de réponse du banc.
- Protocole d'édition (contrat du harnais, à ne pas contourner) :
  - l'agent éditeur n'exécute le modèle que par le harnais, qui compte et journalise chaque passe ; budget fixé avant la séance ;
  - aucun gradient, aucune perte, aucun entraînement : seulement lecture, analyse et écriture directe des poids ;
  - données d'entraînement uniquement (splits train) ; les jeux d'évaluation restent dans `heldout/` ou hors dépôt, jamais lus par l'agent éditeur ; l'évaluation tourne après le gel de l'édition, par une commande à part ;
  - chaque édition publiée sous forme de diff vérifiable (tenseurs touchés, sha256 avant et après) avec le journal de la séance.
- Aucun Ollama lancé par un agent : `ollama create` et les campagnes du banc se lancent par Anaël dans un terminal à part.
- Jamais de push sur `main` : branche préfixée (`feat/`, `fix/`, `docs/`, `chore/`, `exp/`), PR, fusion par merge commit.
- Ne jamais figer une version de dépendance de mémoire : `uv add` résout la version réelle.
- Docs, commits et PR en français ; identifiants de code en anglais.

## Running and verifying

- `uv sync`, puis `uv run pytest`, `uv run ruff check .` et `uv run ruff format --check .` (ligne de 100).
- Toujours `uv run python`, jamais `python` nu. Les tests n'utilisent ni modèle réel ni téléchargement : tenseurs synthétiques.

## Conventions

- Commits : `Préfixe: description en français` (`Feat:`, `Fix:`, `Docs:`, `Chore:`, `Exp:`).
- Versionnage SemVer dans `pyproject.toml` et `__init__.py`, tag annoté et release GitHub à chaque lot livré. Release seulement après une CI verte de `main` sur le commit tagué : `gh run list --branch main --commit <sha>`, puis `gh run watch <id> --interval 60` jusqu'au vert, avant `gh release create`.
