# Espace de séance d'édition

Consigne complète : `CONSIGNE.md` (à relire après chaque compaction).

- Trois commandes seulement, écrites telles quelles : `uv run --project <DEPOT> slm-weight-editing run-batch | session-status | run-script ... --session <ID_SEANCE> --sessions-root sessions`.
- Python uniquement par `run-script scripts/<nom>.py` (numpy, `safetensors.numpy`, bibliothèque standard).
- Seul `sessions/<ID_SEANCE>/work/` est modifiable ; état, journal, verrou et registre en lecture seule.
- Ni web, ni MCP, ni sous-agent, ni lecture hors de ce dossier.
- Plafond ferme : 10 % d'éléments changés par tenseur. Le gel revient à Anaël.
- Toute la séance est auditée : une violation invalide la séance.
