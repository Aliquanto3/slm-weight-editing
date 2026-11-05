# Captures de probe (CAP-1)

## Sortie

- Un dossier `probes/<n>/` dans l'espace par appel, avec :
  - `manifest.json` : identifiants des jetons de chaque texte, sha256 de `work/`, arguments, forme et type de chaque tableau ;
  - un fichier `.npy` par capture, en float32 (conversion exacte depuis bf16), lisible en `mmap_mode`.

## Choix de l'éditeur à chaque appel

| Option | Valeurs | Usage relevé en s1 |
|---|---|---|
| `--layers` | liste de couches | couche 23 (E2) |
| `--capture` | sous-ensemble de `resid` (avant et après chaque bloc), `mlp_act` (entrée de `down_proj`), `attn_out`, `final_norm` (état final normalisé) | `final_norm` (E1), `mlp_act` (E2) |
| `--positions` | toutes, la dernière, ou une plage | toutes les positions des traces |
| `--token-ids` | jetons dont on veut la logit à chaque position | `</think>` |

## Logits

- À chaque position : les 20 meilleures (identifiants et valeurs), plus les jetons de `--token-ids`.
- Vocabulaire entier (248 320) seulement aux positions listées : sur une trace entière, il pèserait environ 3 Go.

## Plafonds par appel (refus avant calcul)

- Au plus 64 textes, de 8 192 jetons chacun au plus.
- Sortie estimée à 2 Go au plus.
- Coût décompté du budget de probe : les jetons passés en forward, tous textes confondus.
