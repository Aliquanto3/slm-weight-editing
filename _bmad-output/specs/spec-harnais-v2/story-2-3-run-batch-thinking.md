---
status: review
story: "2.3"
title: run-batch --thinking on|off (CAP-2)
---

# Story 2.3 : mode raisonnement explicite à chaque lot

- `run-batch ... --thinking on|off` : le gabarit de chat est appliqué avec `enable_thinking=True` (`on`) ou `enable_thinking=False` (`off`), toujours passé explicitement.
- Sans `--thinking`, ou avec une autre valeur (casse comprise : `ON` refusé), `harness.check_thinking` refuse en tout premier, avant contrôle des données, réservation et génération : code 2, budget et journal inchangés. Le message nomme la règle (« --thinking on|off obligatoire : le mode raisonnement est explicite à chaque lot ») et la forme permise.
- Le mode est inscrit dans la ligne `batch` (`thinking: "on"|"off"`) et dans `settings.thinking` (`_generation_settings` prend le mode en argument). `session-status` affiche « Lots par mode raisonnement : on N, off M » (`harness.batches_by_thinking`, lignes `batch` sans mode ignorées).
- Prompt `raw` (tokenizer sans gabarit de chat) : `--thinking` est accepté et inscrit, sans effet sur le prompt (`settings.prompt_format: "raw"` le signale dans le journal). Pas de refus, pour ne pas casser les modèles sans gabarit.
- Audit et réglages de l'espace : les règles allow sont par sous-commande (`run-batch ...`) et l'audit ne liste pas les options libres : `--thinking` est déjà acceptée, rien à changer. Un `run-batch` sans `--thinking` n'est pas une violation de sécurité mais un refus du harnais (aucun calcul, aucune donnée) ; l'audit ne le signale pas (test). 10 contournements, 28 scripts interdits et 73 transcriptions de violation inchangés.
- `seance/CONSIGNE.md` et `seance/CLAUDE.md` expliquent l'option et son caractère obligatoire.
- API : `run_batch(..., thinking=None)` refuse `None` ; les tests existants passent par un assistant qui fixe `off`.

## Reports

- Aucun contrôle que le gabarit du modèle lit réellement `enable_thinking` : un gabarit qui l'ignore rend le même texte pour `on` et `off`.
