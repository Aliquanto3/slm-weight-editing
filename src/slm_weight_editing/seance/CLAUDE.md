# Espace de séance d'édition

Consigne complète : `CONSIGNE.md` (à relire après chaque compaction).

- Seules ces commandes, écrites telles quelles (liste complète et options dans `CONSIGNE.md`) :
  - `uv run<EXTRA> slm-weight-editing run-batch --session <ID_SEANCE> --sessions-root sessions --dataset gsm8k --split train --data data/FICHIER.jsonl --thinking on`
  - `uv run<EXTRA> slm-weight-editing run-batch --session <ID_SEANCE> --sessions-root sessions --dataset gsm8k --split train --data data/FICHIER.jsonl --thinking off`
  - `uv run<EXTRA> slm-weight-editing session-status --session <ID_SEANCE> --sessions-root sessions`
  - `uv run<EXTRA> slm-weight-editing probe --session <ID_SEANCE> --sessions-root sessions --texts notes/textes.jsonl --capture final_norm --positions last`
  - `uv run<EXTRA> slm-weight-editing run-script scripts/NOM.py --session <ID_SEANCE> --sessions-root sessions`
  - `uv run<EXTRA> slm-weight-editing detach-lm-head --session <ID_SEANCE> --sessions-root sessions`
- `detach-lm-head` : sépare `lm_head` des embeddings liés (une seule fois, ni lot ni jeton de probe) ; ne crée jamais `lm_head.weight` ni `config.json` toi-même.
- `run-batch` exige `--thinking on` ou `--thinking off` (mode raisonnement explicite, sinon refus sans décompte).
- `probe` : forward sans génération sur `notes/textes.jsonl` (lignes `{"text": ...}`), captures dans `probes/N/` (lecture seule, `np.load(..., mmap_mode="r")` dans un script), budget de probe de <BUDGET_PROBE> jetons séparé du budget de lots.
- Python uniquement par `run-script scripts/NOM.py` (numpy, `safetensors.numpy`, bibliothèque standard).
- Chaque script : délai <SCRIPT_TIMEOUT> par défaut (au plus <SCRIPT_MAX_TIMEOUT>), <SCRIPT_THREADS> threads de calcul, sortie complète dans `runs/`.
- Seul `sessions/<ID_SEANCE>/work/` est modifiable ; état, journal, verrou et registre en lecture seule.
- Ni web, ni MCP, ni sous-agent, ni lecture hors de ce dossier.
- Plafond ferme : 10 % d'éléments changés par tenseur. Le gel revient à Anaël.
- Toute la séance est auditée : une violation invalide la séance.
