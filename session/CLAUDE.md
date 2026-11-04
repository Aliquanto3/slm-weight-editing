# Espace de séance d'édition

Consigne complète : `CONSIGNE.md` (à relire après chaque compaction).

- Le modèle ne s'exécute que par `uv run slm-weight-editing run-batch` ; `session-status` pour le budget.
- Seul `../models/sessions/<ID_SEANCE>/work/` est modifiable ; état, journal, verrou et registre en lecture seule.
- Ni web, ni MCP, ni sous-agent, ni lecture hors de ce dossier, de `work/`, du journal et des données train.
- Plafond ferme : 10 % d'éléments changés par tenseur. Le gel revient à Anaël.
- Toute la séance est auditée (`audit-session`) : une violation invalide la séance.
