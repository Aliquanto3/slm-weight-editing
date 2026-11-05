# Espace de séance d'édition

Consigne complète : `CONSIGNE.md` (à relire après chaque compaction).

- Trois commandes seulement, écrites telles quelles (liste complète et options dans `CONSIGNE.md`) :
  - `uv run<EXTRA> slm-weight-editing run-batch --session <ID_SEANCE> --sessions-root sessions --dataset gsm8k --split train --data data/FICHIER.jsonl --thinking on`
  - `uv run<EXTRA> slm-weight-editing run-batch --session <ID_SEANCE> --sessions-root sessions --dataset gsm8k --split train --data data/FICHIER.jsonl --thinking off`
  - `uv run<EXTRA> slm-weight-editing session-status --session <ID_SEANCE> --sessions-root sessions`
  - `uv run<EXTRA> slm-weight-editing run-script scripts/NOM.py --session <ID_SEANCE> --sessions-root sessions`
- `run-batch` exige `--thinking on` ou `--thinking off` (mode raisonnement explicite, sinon refus sans décompte).
- Python uniquement par `run-script scripts/NOM.py` (numpy, `safetensors.numpy`, bibliothèque standard).
- Chaque script : délai <SCRIPT_TIMEOUT> par défaut (au plus <SCRIPT_MAX_TIMEOUT>), <SCRIPT_THREADS> threads de calcul, sortie complète dans `runs/`.
- Seul `sessions/<ID_SEANCE>/work/` est modifiable ; état, journal, verrou et registre en lecture seule.
- Ni web, ni MCP, ni sous-agent, ni lecture hors de ce dossier.
- Plafond ferme : 10 % d'éléments changés par tenseur. Le gel revient à Anaël.
- Toute la séance est auditée : une violation invalide la séance.
