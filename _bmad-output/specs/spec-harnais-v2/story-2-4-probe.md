---
status: review
story: "2.4"
title: Commande probe et budget de probe (CAP-1)
---

# Story 2.4 : probe, forward forcé sur budget séparé

Contrat : CAP-1 de `SPEC.md`, format de `probe-captures.md`, entrée `id: "4"` de `stories.yaml`. Fiche à relire avant le code (`spec_checkpoint`).

## But

L'éditeur obtient, par une commande du harnais, les logits et les activations du modèle de `work/` sur des textes fixes, sans aucun jeton généré. Chaque appel est décompté d'un budget de probe en jetons passés en forward, distinct du budget de lots. Les captures sont écrites dans l'espace, lisibles par les scripts en `mmap_mode`.

## Critères (Given / When / Then)

1. **Given** un modèle minuscule et un texte, **when** `probe --capture resid,mlp_act,attn_out,final_norm --positions all`, **then** chaque `.npy` est égal (`np.array_equal`, float32) au même tenseur pris par un forward de référence écrit dans le test sur les mêmes poids, et les 20 meilleures logits de chaque position sont celles de la référence.
2. **Given** un budget de probe B et un usage U, **when** un appel de N jetons avec U + N > B, **then** refus avant tout chargement du modèle, journal inchangé, aucun dossier `probes/<n>/` créé.
3. **Given** 65 textes, ou un texte de 8 193 jetons, ou une sortie estimée à plus de 2 Go, **when** probe, **then** refus avant calcul, message qui nomme le plafond et la valeur demandée.
4. **Given** un appel accepté, **when** il se termine, **then** une ligne `probe_reserve` (jetons, avant calcul) puis une ligne `probe` (numéro, jetons, `tensors_sha256`, `files_sha256` de `work/`, arguments, sha256 de `manifest.json`) sont au journal ; `used_batches` est inchangé.
5. **Given** un appel, **when** il tourne, **then** les poids sont rechargés depuis `work/`, `requires_grad_(False)`, `torch.inference_mode()`, aucun appel à `generate` (test : `generate` patché pour lever), verrou de séance tenu, refus si la séance est gelée.
6. **Given** des captures écrites, **when** un script `run-script` fait `np.load("probes/1/resid_l23.npy", mmap_mode="r")`, **then** code 0 (hypothèse de SPEC.md sur le crochet à confirmer par ce test).
7. **Given** `--token-ids 7,9` et `--full-vocab-positions 3`, **then** le manifest décrit la logit des jetons 7 et 9 à chaque position et le vocabulaire entier à la seule position 3.

## Fichiers et fonctions touchés (vérifiés dans le code au 6de87e6)

- Nouveau `src/slm_weight_editing/probe.py` (harness.py fait déjà 886 lignes) : `run_probe(...)`, `estimate_output_bytes(...)`, `add_parser`, `run`. Réutilise de `harness.py` : `_session_lock`, `read_state`, `read_journal`, `_append_journal`, `tensor_hashes`, `work_file_hashes`, `bind_workspace`, `session_dir`, `_now`.
- `harness.py` : `SessionState` et `open_session` gagnent le budget de probe (ligne `open` : `probe_budget`) ; nouvelle `used_probe_tokens(directory)` (somme des `probe_reserve`, recoupée avec `state.json` comme `used_batches`).
- `workspace.py` : `ALLOWED_SUBCOMMANDS` + `probe`, `create_workspace(..., probe_budget=2_000_000)` et option `--probe-budget` de `session-workspace`, deny `Edit(./probes/**)`.
- `script_runner.py` : `cfg["read_roots"]` + `<espace>/probes` (inutile si la story 2.5 ouvre la lecture à tout l'espace).
- `audit_transcript.py` : `ALLOWED_SUBCOMMANDS` + `probe` ; `_BOUND_OPTIONS` + `--texts` (sous l'espace).
- `__init__.py` : branchement de `probe.add_parser`.
- `seance/CONSIGNE.md`, `seance/CLAUDE.md` : quatrième commande, budget de probe (`<BUDGET_PROBE>`).

## Choix de conception

| Point | Options | Recommandation |
|---|---|---|
| Entrée | (a) fichier JSONL `--texts notes/x.jsonl`, une ligne `{"text"}` ou `{"token_ids"}` ; (b) texte en argument | (a) : 64 textes de 8 192 jetons ne tiennent pas dans une commande ; `token_ids` permet de rejouer exactement une sortie de lot. |
| Gabarit | texte brut seul, ou `{"chat": {"prompt", "completion"}, "thinking": on/off}` rendu par la fonction de la story 2.3 | gabarit disponible, même fonction que `run-batch` (sinon rejeu non fidèle) ; dépend de 2.3. |
| Lot de textes | un forward par texte, sans rembourrage ; ou textes groupés | un par texte, comme `run-batch` : seule façon d'avoir des captures identiques octet pour octet à la référence. |
| Captures | crochets `register_forward_hook` / `register_forward_pre_hook` (entrée de `down_proj`, sortie d'attention, résidu avant et après bloc, norme finale) ; ou `output_hidden_states` | crochets : `output_hidden_states` ne donne ni `mlp_act` ni `attn_out`. bf16 → float32 par `.float()` (exact). |
| Décompte | réservation avant calcul sous verrou, consommée même si l'appel plante | comme les lots (story 3) ; question 3. |
| Emplacement | `probes/<n>/` à la racine de l'espace | oui (SPEC) ; écrit par le harnais seul, deny `Edit` pour l'éditeur. |

Estimation de sortie (avant calcul) : Σ textes × positions retenues × (couches × captures × hidden × 4 + 20 × 8 + len(token_ids) × 4) + positions plein vocabulaire × 248 320 × 4, plus 1 % de marge.

## Tests

- À ajouter (`tests/test_probe.py`, modèle Llama minuscule de `test_harness.py`) : critères 1 à 7 ; budget exact atteint puis refus ; `state.json` réduit à la main refusé ; séance gelée ; verrou présent ; CLI et codes de sortie ; `--texts` hors espace refusé depuis l'espace ; aucune écriture hors `probes/<n>/` et du journal.
- À réaligner : `test_session_workspace.py::test_settings_json` (allow `probe`, deny `probes/**`), `test_cuda_allow_rules_match_audit_whitelist`, `test_instructions_are_rendered` ; `test_audit_transcript.py::test_consigne_states_protocol`, `test_extra_cuda_is_accepted` (paramètre `probe`), `test_options_outside_workspace_are_flagged` (`--texts`) ; `test_harness.py::test_open_writes_state_and_journal` (champ `probe_budget`).

## Garanties à préserver

- 10 contournements, 28 scripts interdits, 73 transcriptions de violation : toujours en échec (suites existantes inchangées hors réalignements ci-dessus).
- probe ne génère rien et ne touche pas au budget de lots ; aucune lecture hors de l'espace (`--texts` lié à l'espace comme `--data`).
- Le manifest garde les identifiants de jetons de chaque texte : la story 2.6 en a besoin pour signaler les suites d'appels qui prolongent un texte d'un jeton.

## Dépendances

- Story 2.3 (gabarit `enable_thinking`) pour l'option de gabarit ; story 2.2 (`run-script`, `runs/`) touche aussi `script_runner.py` et `CONSIGNE.md` ; story 2.5 touche `workspace.py`, `script_runner.py` et `CONSIGNE.md` : ne pas les mettre dans la même vague, ou réaligner après fusion.
- Threads : probe tourne dans le processus du harnais (torch), pas sous le plafond BLAS de CAP-8 ; `torch.set_num_threads` au même plafond est proposé (non demandé par SPEC).

## Questions pour Anaël

1. Entrée de probe : texte brut seulement, ou aussi `token_ids` et gabarit de chat avec `--thinking` (rejeu exact des sorties de lots) ?
2. Budget de probe : option `--probe-budget` de `session-workspace` (défaut 2 M) inscrite dans la ligne `open` ; une séance ouverte sans ce champ (v1, s1) refuse-t-elle probe ?
3. Un appel qui plante après sa réservation consomme-t-il ses jetons, comme un lot ?
4. `probes/` interdit en écriture à l'éditeur (deny `Edit`, crochet du lanceur), et captures non publiées au gel (seuls leurs sha256 au journal) ?
5. Faut-il un plafond disque cumulé pour `probes/` sur la séance (2 M jetons de captures plein résidu peuvent dépasser des dizaines de Go) ?

## Décisions de l'orchestrateur (2026-11-06)

Anaël a levé ses points d'arrêt de relecture le 2026-11-06 (« Retire mes points d'arrêt et continue »). Les réponses ci-dessous sont prises par l'orchestrateur, au plus prudent ; Anaël peut les revoir à la relecture des PR.

1. Texte brut seulement en s2. L'éditeur peut écrire lui-même le gabarit de chat dans le texte. `token_ids` et `--chat` sont reportés.
2. Option `--probe-budget` de `session-workspace`, 2 000 000 jetons par défaut, inscrite dans la ligne `open`. Une séance ouverte sans ce champ (v1, s1) refuse probe, avec un message qui le dit.
3. Oui : la réservation est consommée même si l'appel plante après elle, comme un lot.
4. Oui : `probes/` est interdit en écriture à l'éditeur (deny `Edit`/`Write` et crochet du lanceur). Les captures ne sont pas publiées au gel, seuls leurs sha256 sont au journal.
5. Oui : plafond disque cumulé de `probes/` sur la séance, 20 Go par défaut (option `--probe-disk-max` de `session-workspace`), vérifié avant calcul avec l'estimation de sortie.

## Réalisation (2026-11-06)

Fait : `probe.py` (commande, plafonds, estimation, crochets, captures `.npy` en memmap, manifest, journal `probe_reserve` puis `probe`), `session-workspace --probe-budget/--probe-disk-max`, deny `Edit(./probes/**)`, `probes/` en lecture pour `run-script` (le crochet interdit déjà toute écriture hors `work/` et `scripts/`), audit (`probe` en liste blanche, `--texts` lié à l'espace), consignes. Tests : `tests/test_probe.py` et réalignements de `test_session_workspace.py` et `test_audit_transcript.py`.

Écarts à la fiche :
- Le budget de probe vit dans la ligne `open` du journal (et `workspace.json`), pas dans `state.json` (`SessionState` inchangé : les séances v1 restent lisibles). `used_probe_tokens` somme les `probe_reserve` du journal, que l'éditeur ne peut pas écrire.
- `open_session` n'inscrit `probe_budget` et `probe_disk_max` que s'ils sont donnés : `test_open_writes_state_and_journal` reste inchangé.
- Les captures sont écrites dans `<espace>/probes/<n>/` (racine de l'espace, hors séance) ; les lignes de tous les textes sont mises bout à bout dans chaque `.npy`, offsets dans `manifest.json`. Noms : `resid_pre_lN`, `resid_lN` (après bloc N), `mlp_act_lN`, `attn_out_lN`, `final_norm`, `top_logit_ids`, `top_logit_values`, `token_logits`, `full_vocab_logits`.
- Estimation de sortie : `mlp_act` compté à `intermediate_size` (la fiche comptait `hidden`), tops à 12 octets (id int64 + valeur float32) ; plafond disque cumulé sur les estimations des réservations.
- 20 meilleures logits : `min(20, vocabulaire)` (modèle de test à 10 jetons).
- Texte lu avec `add_special_tokens=False` (l'éditeur écrit lui-même le gabarit) ; `--device` (auto par défaut) ajouté comme `run-batch`.
- `journal` : `probe` porte aussi le sha256 de chaque `.npy` (`output_files`).
- Non fait (reporté) : `token_ids` en entrée, `--chat`, `torch.set_num_threads`.

## Revue

### Passe 1 (2026-11-06, SHA revu 1574b9a)

Base : `feat/probe` (b84596f) fusionnée avec `origin/main` (docs seulement), sans rebase (story 2.5 empilée).

- Important · medium · second avis 1 · corrigé : `model(...).logits[0].float()` matérialisait [jetons, vocabulaire] en bf16 puis float32 (8 192 x 248 320 : 4 + 8 Go) même avec `--positions last`, OOM après `probe_reserve` donc budget perdu. Forward en `logits_to_keep=1`, logits recalculées par blocs de `LOGIT_CHUNK` = 256 positions depuis la sortie de la norme finale par `get_output_embeddings()`, garde d'égalité exacte avec la logit de la dernière position rendue par le forward (refus si le modèle post-traite ses logits). Test `test_logits_never_materialised_for_all_positions` (blocs de 4, égalité à la référence).
- Important · low · second avis 2 · corrigé : le refus de plus de 64 textes ne donnait pas le nombre demandé (critère 3) ; compte complet puis message « 65 demandés », test réaligné.
- Nit · low · second avis 3 · corrigé : `--texts` lu en `utf-8-sig` (BOM de PowerShell 5 toléré), test ajouté.
- Nit · low · second avis 4 · rejeté : les crochets de la ligne `probe` de `CONSIGNE.md` suivent la convention déjà en place pour `run-batch` (`[--start N]`) ; `CLAUDE.md` donne la forme recopiable sans crochets et `test_audit_accepts_consigne_probe_command` vérifie les deux formes.
- Nit · low · second avis 5 · rejeté : écart consigné et motivé (budget dans la ligne `open`, somme des `probe_reserve` d'un journal non inscriptible par l'éditeur ; aucun compteur `state.json` à recouper).
- Complexité : 0 constat (journal relu deux fois par appel, O(n) ; offsets O(textes²) sur 64 textes au plus).
- Report (GPU/modèle réel) : vérifier sur Qwen3.5 en CUDA que la garde d'égalité de la dernière logit tient (lm_head sans post-traitement, même noyau pour 1 ligne) et mesurer la mémoire crête d'une probe de 8 192 jetons en `--positions all`.
