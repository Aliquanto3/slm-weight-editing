# Story 6 : comparaison chiffrée de deux éditions d'un même modèle

Spec d'origine : entrée `id: "6"` de `stories.yaml` (SPEC CAP-7). Outil `compare-edits` (`src/slm_weight_editing/compare_edits.py`) ; `compare_deltas(deltas_a, deltas_b, total_weights)` sert d'entrée par diff pour la story 4.

## Revue

### Passe 1 (2026-11-04, SHA revu b598c2d)

Base à jour (origin/main 5ec3233), aucun rebase. Branchement vérifié : `add_parser(sub)` + `set_defaults(func=run)`, dispatch `args.func` dans `main`. Aucune dépendance ajoutée. Wilson vérifié contre une valeur publiée (5/10) ; l'oracle Newcombe recalculait la même formule : ajout de l'exemple publié de Newcombe (1998), 56/70 contre 48/80, IC [0,0524 ; 0,3339]. Complexité : aucun constat (pas de structure qui grossit par événement).

- Important, medium, second avis n° 1 : corrigé. Pour un tenseur 1-D (biais, norme), le cosinus droit valait toujours 1.0 (vt 1x1) et le gauche répétait le cosinus aplati. Directions principales n/a, avec une note, pour tout tenseur de moins de 2 dimensions.
- Important, medium, second avis n° 2 : corrigé. Une piste ou une catégorie présente dans un seul résumé était écartée sans le dire ; elle est listée « absente de A/B » (champ `missing` dans le JSON).
- Nit, low, second avis n° 3 : corrigé avec le n° 1 (un tenseur 0-d ne passe plus par la SVD).
- Nit, low, second avis n° 4 : corrigé. Un delta NaN ou inf donne une note « non fini » au lieu d'un `LinAlgError` ; le cosinus aplati peut alors valoir NaN (écrit `NaN` dans le JSON).
- Nit, low, second avis n° 5 : reporté. Les deltas sont gardés en float64 (environ 6,4 Go par édition complète d'un modèle de 0,8 Md). Les éditions visées touchent peu de tenseurs ; à revoir (float32, ou calcul tenseur par tenseur) si une édition complète doit être comparée.
- Nit, low, second avis n° 6 : corrigé. `--out x.json` est refusé (code 1), le JSON s'écrit à côté du Markdown.
- Nit, low, second avis n° 7 : corrigé. La docstring précise que l'IC du score vient du banc, tel quel.
- Nit, low, second avis n° 8 : rejeté. Faux positif de Pyright : `main(argv: list[str] | None)` accepte la liste.
- Nit, low, second avis n° 9 : corrigé. `--tag-a`/`--tag-b` sans `--summary` sont refusés ; avec un seul résumé, deux tags distincts sont exigés.

### Passe 2 (2026-11-04, SHA revu d9929d9)

Base à jour (origin/main 5ec3233), aucun rebase. Relecture du seul diff b598c2d..d9929d9. Vérifié à l'exécution : pour un biais 1-D et un scalaire 0-d, le cosinus aplati reste calculé (0,9129 et -1,0) et les directions sont n/a avec la note ; une piste ou catégorie absente d'un côté donne une ligne « absente de <libellé> » dans le Markdown et `missing` = "A"/"B" dans le JSON, sans autre texte que les noms de piste et de catégorie déjà affichés auparavant. Test de Newcombe : oracle indépendant (valeurs publiées), valable. ruff propre, 63 tests passent. Complexité : aucun constat.

- Aucun constat nouveau : les deux Important et les nits de la passe 1 tiennent.
