---
status: done
story: "2.3"
title: run-batch --thinking on|off (CAP-2)
---

# Story 2.3 : mode raisonnement explicite à chaque lot

- `run-batch ... --thinking on|off` : le gabarit de chat est appliqué avec `enable_thinking=True` (`on`) ou `enable_thinking=False` (`off`), toujours passé explicitement.
- Sans `--thinking`, ou avec une autre valeur (casse comprise : `ON` refusé), `harness.check_thinking` refuse en tout premier, avant contrôle des données, réservation et génération : code 2, budget et journal inchangés. Le message nomme la règle (« --thinking on|off obligatoire : le mode raisonnement est explicite à chaque lot ») et la forme permise.
- Le mode est inscrit dans la ligne `batch` (`thinking: "on"|"off"`) et dans `settings.thinking` (`_generation_settings` prend le mode en argument). `session-status` affiche « Lots par mode raisonnement : on N, off M » (`harness.batches_by_thinking`, lignes `batch` sans mode ignorées).
- Prompt `raw` (tokenizer sans gabarit de chat) : `--thinking` est accepté et le mode demandé inscrit dans la ligne `batch`, sans effet sur le prompt : `settings.prompt_format: "raw"` et `settings.thinking: null` (mode appliqué) le signalent, et `batches_by_thinking` ne compte pas ces lots. Pas de refus, pour ne pas casser les modèles sans gabarit (les cibles du cadrage, Qwen3.5 et MiniCPM5, en ont un).
- Audit et réglages de l'espace : les règles allow sont par sous-commande (`run-batch ...`) et l'audit ne liste pas les options libres : `--thinking` est déjà acceptée, rien à changer. Un `run-batch` sans `--thinking` n'est pas une violation de sécurité mais un refus du harnais (aucun calcul, aucune donnée) ; l'audit ne le signale pas (test). 10 contournements, 28 scripts interdits et 73 transcriptions de violation inchangés.
- `seance/CONSIGNE.md` et `seance/CLAUDE.md` expliquent l'option et son caractère obligatoire.
- API : `run_batch(..., thinking=None)` refuse `None` ; les tests existants passent par un assistant qui fixe `off`.

## Reports

- Aucun contrôle que le gabarit du modèle lit réellement `enable_thinking` : un gabarit qui l'ignore rend le même texte pour `on` et `off`.

## Revue

### Passe 1 (2026-11-05, SHA revu 1f266c4)

Base à jour (origin/main c26a27c). Suite complète, ruff check et format verts après correctifs (CPU seulement).

- Important, medium, second avis 1 : corrigé. `CLAUDE.md` impose des commandes « écrites telles quelles », or la consigne écrivait `--thinking on|off` ; copiée, la barre est un enchaînement refusé (`check_shell` rend `commande-hors-liste`, vérifié). Consigne : `--thinking <MODE>` avec `on` ou `off` et l'avertissement ; `CLAUDE.md` : `--thinking on` ou `--thinking off`. Test : chaque commande du bloc de la consigne rendue, gabarits remplis, passe `check_shell`.
- Important, medium, second avis 2 : corrigé. En `raw`, `settings.thinking` disait `on` et `session-status` comptait un lot « on » sans raisonnement appliqué. Refuser `on` en raw demanderait de charger le tokenizer avant la réservation (complexité pour un cas hors cadrage) : le mode demandé reste dans la ligne `batch`, `settings.thinking` vaut `null` et le lot n'est pas compté par mode. Test raw renforcé ; le test CLI de décompte passe sur un checkpoint avec gabarit.
- Nit, low, second avis 3 : rejeté. CAP-2 veut alterner `on` et `off` dans une séance.
- Nit, low, second avis 4 : corrigé. `thinking` obligatoire dans `_generate` et `_generate_on` (a révélé un test qui appelait `_generate` sans mode, réaligné).
- Nit, low, second avis 5 : corrigé. `run()` contrôle le mode avant `bind_workspace`.
- Nit, low, second avis 6 : corrigé. Test : sans mode, sur séance gelée, le refus du mode passe en premier, journal et budget inchangés.
- Complexité : 0 constat (`batches_by_thinking` lit le journal une fois, seulement à `session-status`).

### Passe 2 (2026-11-05, SHA revu e6f1ad3)

Relu : le correctif e6f1ad3 seul. Base à jour (origin/main c26a27c). Suite complète (506 réussis, 1 sauté, CPU seulement), ruff check et format verts après correctif.

- Important, medium, orchestrateur : corrigé. Le gabarit `--thinking <MODE>`, recopié tel quel, est une redirection : `check_shell` rend `commande-hors-liste` (« substitution, accent grave ou redirection », vérifié), donc une faute d'audit et non un refus clair du harnais. Consigne : deux lignes `run-batch` complètes, `--thinking on` et `--thinking off`, sans gabarit de mode. Test réaligné (4 commandes, mode de chaque ligne vérifié) et nouveau test : `CONSIGNE.md` et `CLAUDE.md` rendus donnent `--thinking on` et `--thinking off`, sans `<MODE>` ni `--thinking on|off`.
- Nit, low, orchestrateur : corrigé. Fiche passée à `done` avant la fin de la revue, remise à `review`.
- Pre-existing, low, reviewer : reporté. `data/<fichier>` et `scripts/<nom>.py` ont le même défaut (chevrons recopiés = redirection, faute d'audit) ; hors périmètre de la story.
- Vérifié sans constat : la ligne `batch` (`thinking` demandé, `settings.thinking` null en `raw`) reste lisible par le gel (`public_journal` ne masque que les chemins), l'audit (ne lit que `data_path`) et compare-edits (ne lit pas le journal) ; `check_thinking` avant `bind_workspace` dans `run()`.
- Complexité : 0 constat.
