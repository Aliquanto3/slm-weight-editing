---
status: review
story: "2.8"
title: lm_head détaché, n'éditer que la sortie (CAP-10)
---

# Story 2.8 : lm_head détaché

Contrat : CAP-10 de `SPEC.md`, entrée `id: "8"` de `stories.yaml`, `../spec-slm-weight-editing/fix-report-embed.md` (`_check_tied`). Fiche à relire avant le code.

## Condition préalable

**Fiche conditionnelle.** Elle ne passe en développement que si l'essai de la story 2.7 (`docs/essai-output-weight.md`, résultat dans `docs/resultats/essai-output-weight.md`) conclut « CAP-10 maintenu » : `qwen35-output-test` répond mot pour mot comme le témoin sur les 8 prompts. L'essai n'est pas encore fait par Anaël. S'il échoue, CAP-10 sort de s2 : la story est annulée et rien n'est codé, la v1 reste en place (`_check_tied` refuse tout `lm_head`). L'écart de taille et de temps de réponse relevé par l'essai fait aussi partie de la décision (question 5).

## But

Pendant la séance, l'éditeur peut séparer `lm_head` des embeddings liés par une commande du harnais, une seule fois, avec une ligne au journal. Il peut ensuite éditer la sortie d'un jeton (ici `</think>`) sans toucher son embedding d'entrée, qui se trouve dans le prompt (incident « Embedding d'entrée perturbé » de s1).

## Critères (Given / When / Then)

1. **Given** une séance aux embeddings liés, **when** l'éditeur lance la commande de détachement, **then** `work/` contient `lm_head.weight` égal octet pour octet à `embed_tokens`, `config.json` déclare `tie_word_embeddings: false` (à la racine et dans `text_config` s'il existe), et une ligne `detach_lm_head` est au journal (sha256 avant et après, nom source).
2. **Given** un détachement fait, **when** on le relance, **then** refus (déjà détaché) ; **given** une séance gelée, un verrou présent ou un checkpoint déjà délié, **then** refus.
3. **Given** un lot rejoué juste après le détachement, **then** ses `outputs` sont identiques à ceux du même lot avant (modèle minuscule, glouton).
4. **Given** un `lm_head.weight` ou un `config.json` écrits par un script ou un outil, sans ligne `detach_lm_head`, **when** gel, **then** refus (tenseur en trop, comme aujourd'hui) ; aucun diff produit.
5. **Given** un détachement puis l'édition de quelques lignes de `lm_head`, **when** gel, **then** le diff décrit `lm_head.weight` comme tenseur ajouté, indices et valeurs relatifs à l'`embed_tokens` de la base, sous le plafond de 10 % ; **when** apply-diff, **then** les sha256 publiés reviennent.
6. **Given** ce diff, **when** `report-gguf --edit-diff`, **then** le GGUF contient `output.weight` en Q8_0. Les blocs non édités sont égaux octet pour octet à `token_embd` (témoin de fidélité, comme `fidelite.json`), seuls les blocs des lignes éditées changent, et `token_embd` est inchangé si `embed_tokens` ne l'est pas.
7. **Given** le cas lié (embed visé sans détachement), **then** `_check_tied` refuse toujours un `lm_head` distinct, une config déliée ou un `output.weight` : ses quatre tests restent verts.
8. **Given** `compare-edits` sur ce diff, **then** le rapport affiche « tenseur ajouté : `lm_head.weight` » et compte les poids édités par rapport à l'`embed_tokens` de la base.

## Fichiers et fonctions touchés (vérifiés au 6de87e6)

- Nouvelle commande de l'éditeur (nom proposé : `detach-lm-head`), dans `harness.py` ou un nouveau `lm_head.py`. Elle prend le verrou et `bind_workspace`, comme `run-batch`.
- `workspace.py` : `ALLOWED_SUBCOMMANDS` et les allow ; `audit_transcript.py` : `ALLOWED_SUBCOMMANDS` ; `seance/CONSIGNE.md` : la commande, son effet et son coût (environ 270 Mo de plus au GGUF).
- `freeze.py` : `compute_tensor_diff` refuse aujourd'hui tout tenseur en plus. Il l'accepte désormais pour le seul `lm_head.weight` quand le journal porte `detach_lm_head`, comparé à l'`embed_tokens` de la base. `changed_files` : un `config.json` dont le seul écart est `tie_word_embeddings` devient une transformation connue, pas une entrée de `files_changed`. `replay_tensors` et `apply_diff` rejouent le détachement (ajout du tenseur, réécriture de la config) puis le diff creux. `FORMAT` passe en `edit-diff-v2` (question 1).
- `__init__.py` (CLI `report-gguf --edit-diff`) : sans le changement de `changed_files`, le refus « fichiers hors tenseurs » bloquerait le report.
- `report_gguf.py` : `_HF_TO_GGUF` + `lm_head.weight` → `output.weight` ; ajout de `output.weight` en reprenant `gguf_add_output.add_output_weight` (copie Q8_0 de `token_embd`), puis patch des blocs édités ; `_check_tied` n'est contourné que si le diff porte le détachement.
- `compare_edits.py` : `_check_same` refuse aujourd'hui les noms en trop ; `compare_deltas` et `render_markdown` reçoivent l'entrée « tenseur ajouté ».

## Choix de conception

| Point | Options | Recommandation |
|---|---|---|
| Écriture dans `work/` | (a) réécrire `model.safetensors` avec le tenseur en plus, remplacement atomique ; (b) `lm_head.safetensors` séparé et `model.safetensors.index.json` | (a) : un seul fichier, pas d'index qui change le chargement ; le nom exact attendu par transformers pour Qwen3.5 (`lm_head.weight` à la racine) est à vérifier sur l'en-tête de `models/hf/qwen3.5-base`, sans téléchargement. |
| Format du diff | (a) v2 avec un champ `detached_lm_head: {source, config_keys}` ; (b) v1 avec `lm_head` complet | (a) : (b) publierait un tenseur entier, contraire au plafond. |
| Source de la copie | `embed_tokens` courant de `work/` ; ou celui de la base exigé égal | question 3 (point de protocole). |
| GGUF | `output.weight` copié puis patché ; ou requantifié en entier depuis le bf16 | copie puis patch : le témoin « égal à `token_embd` » ne tient que comme ça. |

## Tests

- À ajouter : critères 1 à 8 (modèle Llama minuscule lié de `test_harness.py`, GGUF synthétique de `test_report_gguf.py` et `test_gguf_add_output.py`) ; `config.json` modifié d'une autre clé encore signalé ; détachement par un script refusé au gel ; diff v1 toujours lisible.
- À réaligner : `test_freeze.py::test_shape_dtype_or_names_changes_are_refused`, `test_changed_non_tensor_files_are_reported`, `test_replay_refuses_non_empty_output_and_unknown_format` (format v2), `test_report_gguf_refuses_a_diff_with_changed_files` ; `test_report_gguf.py::test_report_embed_refuses_a_distinct_lm_head`, `test_report_embed_refuses_untied_config`, `test_report_embed_refuses_a_gguf_with_output_weight` (gardés pour le cas lié) ; `test_compare_edits.py` (cas « noms de tenseurs », ligne 143) ; `test_session_workspace.py::test_settings_json`, `test_audit_transcript.py` (liste blanche).

## Garanties à préserver

- Plafond de 10 % par tenseur inchangé ; `lm_head` comparé à l'`embed_tokens` de la base.
- Le détachement passe par le harnais, une fois. Une écriture directe de `lm_head.weight` ou de `config.json` ne vaut pas détachement (critère 4).
- Les 10 contournements, les 28 scripts interdits et les 73 transcriptions de violation restent en échec. La nouvelle commande est liée à l'espace et n'écrit que dans `work/` et le journal.

## Dépendances

- Story 2.7 (essai Ollama, Anaël). Le rapport réel sur le blob officiel est à refaire par Anaël après fusion : `fidelite.json` doit montrer `output.weight` en ok.
- Elle touche `workspace.py`, `audit_transcript.py` et `CONSIGNE.md`, comme 2.3 à 2.6 : à faire en dernier.

## Questions pour Anaël

1. Diff au format v2 (le détachement est une transformation rejouée par apply-diff), au lieu d'un `lm_head` publié en entier ?
2. Le nom de la commande (`detach-lm-head`) et son coût : un détachement ne consomme ni lot ni jeton de probe, est-ce bien voulu ?
3. Le détachement est-il permis après une édition de `embed_tokens` ? Si oui, copie-t-on l'embed courant (déjà édité) ou celui de la base ?
4. Comment présenter une séance avec `lm_head` détaché dans la comparaison au MIT (mention dans le rapport, compteur à part) ?
5. Si l'essai 2.7 passe mais que l'écart de taille ou de temps de réponse au banc est net, quel seuil fait sortir CAP-10 de s2 ?

## Décisions de l'orchestrateur (2026-11-06)

Anaël a levé ses points d'arrêt de relecture le 2026-11-06 (« Retire mes points d'arrêt et continue »). Les réponses ci-dessous sont prises par l'orchestrateur, au plus prudent ; Anaël peut les revoir à la relecture des PR.

1. Oui : diff au format v2, le détachement est une transformation rejouée par apply-diff.
2. `detach-lm-head`. Il ne consomme ni lot ni jeton de probe, mais il est journalisé et refusé après le gel.
3. Permis à tout moment. On copie l'embed courant de `work/` (éventuellement déjà édité). Le diff rejoue les transformations dans l'ordre, donc le résultat reste vérifiable.
4. Champ `lm_head_detached` dans le rapport et mention « écart de protocole » dans la comparaison au MIT.
5. Story codée sans attendre l'essai 2.7. CAP-10 sort de s2 si l'essai 2.7 échoue, ou si le GGUF détaché dépasse le témoin de plus de 25 % en taille ou de 20 % en temps de réponse médian au banc.

## Réalisation (2026-11-06)

Codée sans attendre l'essai Ollama de 2.7 (décision 5) : la commande ne sert en s2 que si l'essai conclut « CAP-10 maintenu » et si le GGUF détaché reste sous 25 % de taille et 20 % de temps de réponse médian de plus que le témoin.

- Nouvelle commande `detach-lm-head` (module `lm_head.py`, parser et aiguillage dans `harness.py`) : verrou de séance, refus après le gel, une fois, ligne `detach_lm_head` au journal (source, sha256 de l'embed et de `lm_head`, sha256 du fichier réécrit, sha256 avant et après de `config.json`, clés écrites). Écriture : `model.safetensors` réécrit avec le tenseur en plus (option (a), remplacement atomique) ; refus d'un checkpoint à `model.safetensors.index.json`.
- `freeze.py` : `FORMAT` `edit-diff-v2` (v1 toujours lu), champ `detached_lm_head`, entrée `added` pour `lm_head.weight` (toujours émise, même sans édition : le détachement doit se rejouer), `config.json` délié reconnu seulement si la transformation recalculée redonne le sha256 journalisé, rejeu par `apply-diff`.
- `report_gguf.py` : `lm_head.weight` vers `output.weight`, `output.weight` ajouté par `add_output_weight` puis patch par ligne, témoin de fidélité `output.weight` (embed de la base requantifié contre `token_embd`), champs `lm_head_detached` et `protocol_gap` dans `diff.json`. `compare_edits.py` : « tenseur ajouté » et « écart de protocole ». Consigne de séance : commande recopiable, sans chevron ni crochet.

### Écarts à la spec

- Le tenseur ajouté figure toujours au diff, même inchangé (entrée `changed: 0`) ; la spec ne le disait pas, mais sans lui apply-diff ne peut pas rejouer le détachement.
- Un détachement journalisé avec `lm_head.weight` entièrement égal à `embed_tokens` reste un « tenseur modifié » dans le décompte du gel (`tensors_changed`).
- `test_report_embed_refuses_a_distinct_lm_head` et les deux autres tests du cas lié sont inchangés et verts (sans diff détaché, `lm_head` reste refusé) ; seul `gguf_name("lm_head.weight")` change de valeur.
- Le nom du tenseur dans le GGUF est celui de la copie de `token_embd` ; le report ne vérifie pas un `lm_head` hors racine (`model.lm_head.weight` refusé).
- Non fait : le rapport réel sur le blob officiel et l'essai Ollama (à Anaël) ; mesure de taille et de temps de réponse de la décision 5.

## Revue

### Passe 1 (2026-11-06, SHA revu 90cccf8)

Base à jour (main a691faa). Tests ciblés : lm_head, freeze, audit_transcript, session_workspace, report_gguf, harness, compare_edits, gguf_add_output verts (invariants 10 / 28 / 73 inclus) ; les trois tests ajoutés échouent sur 90cccf8.

- Important · medium · second avis 1 · corrigé : un `work/config.json` relié (`tie_word_embeddings: true`) après le détachement passait le gel ; le diff publiait une config déliée alors que les lots mesurés ignoraient `lm_head`. Le gel refuse désormais une config de `work/` qui n'est plus déliée (`_check_detached_config`). Test `test_config_relinked_after_detach_is_refused_at_freeze`.
- Important · medium · second avis 2 · corrigé : tenseur, config et journal écrits sans reprise ; une erreur entre deux bloquait la séance (detach « déjà délié », gel « tenseur en trop »). Le fichier d'origine est gardé par renommage (`.detach-bak`) et work/ est restauré en cas d'erreur avant la ligne de journal. Reste une fenêtre entre deux renommages si le processus est tué. Test `test_detach_interrupted_before_the_journal_line_restores_work`.
- Important · low → medium · second avis 3 · corrigé : un `config.json` modifié avant le détachement donnait un diff que le gel produisait mais qu'apply-diff refusait de rejouer. `detach-lm-head` refuse une config qui diffère de la base à l'ouverture, et le gel refuse un détachement dont la transformation ne se rejoue pas sur l'original. Test `test_detach_refuses_a_config_modified_since_open`.
- Rejeté · low · second avis 4 : « s'il existe » du critère 1 porte sur `text_config`, pas sur la clé ; écrire `text_config.tie_word_embeddings: false` dès que `text_config` existe est le comportement demandé (et le plus sûr pour Qwen3.5).
- Nit · low · second avis 5 · corrigé : la comparaison n'était pas tautologique (elle confrontait la consigne rendue à la liste blanche), mais elle suivait la constante ; liste écrite en dur.
- Complexité : 0 constat (pas de structure qui grossit parcourue à chaque événement en plus de l'existant).

### Passe 2 (2026-11-06, SHA revu 23149d1)

Relu `90cccf8..23149d1`. Tests ciblés verts (lm_head, freeze, audit_transcript, session_workspace, harness, report_gguf), ruff vert.

- Important · medium · revue · corrigé : la sauvegarde `model.safetensors.detach-bak` était dans work/ ; un processus tué après la ligne de journal la laissait, et le gel la publiait comme fichier ajouté (`files_changed`, sha256_before nul : copie complète du modèle qu'apply-diff ne peut pas produire) ; tué entre les deux renommages, work/ n'avait plus de safetensors et rien ne la reprenait. Sauvegardes (tenseur et config) déplacées dans `<séance>/detach-bak/`, hors de work/ (ni haché, ni audité, ni publié) ; le détachement suivant restaure work/ s'il n'y a pas de ligne de journal, nettoie sinon, et retire les temporaires `.tmp` laissés dans work/. Tests `test_detach_killed_between_the_two_renames_is_recovered_by_the_next_detach`, `test_backup_left_after_the_journal_line_is_not_a_work_file`.
- Nit · low · revue · corrigé : une `config.json` absente de la base mais ajoutée dans work/ passait le refus « modifiée depuis l'ouverture » et n'était refusée qu'au gel ; refus dès le détachement.
- Nit · low · revue · rejeté : `os.replace` sous Windows sur le fichier d'embeddings ; `SafeTensor` ne garde aucun descripteur ouvert (lecture par `with`), et le verrou de séance exclut lots et sondes pendant le détachement.
- Nit · low · revue · rejeté : restauration en échec dans le `except` ; l'exception d'origine reste chaînée et la sauvegarde, hors de work/, est reprise au détachement suivant.
- `_check_detached_config` et le refus d'une transformation non rejouable au gel : conformes (une config déliée avec d'autres modifications reste signalée par `files_changed`).
- Complexité : 0 constat.

### Passe 3 (2026-11-06, SHA revu 47a84c5)

Relu `23149d1..47a84c5`, puis fusion de main (e512639 : stories 2.4 et 2.5, crochet de refus, OPERATEUR.md, BLOCAGES.md au gel) par merge commit. Conflit d'`audit_transcript.py` résolu en gardant `ALLOWED_WAYS` / `refusal_message` et `detach-lm-head` dans `ALLOWED_SUBCOMMANDS` et la liste des commandes des voies permises ; la liste fermée des appels permis de `test_espace_lecture.py` inclut la commande detach de la consigne (le crochet la laisse passer). `freeze.py`, CLI et consignes fusionnés sans conflit, vérifiés. Tests ciblés verts (lm_head, freeze, audit_transcript, session_workspace, espace_lecture, harness, report_gguf, probe ; invariants 10 / 28 / 73 inclus), ruff vert.

- Vérifié · `<séance>/detach-bak/` : hors de `work/`, donc ni haché (`work_file_hashes`), ni publié au gel (seuls work/ et l'original sont lus), ni dans `files_changed` ; non écrivable par l'éditeur (crochet : écriture limitée à scripts/, notes/, EDITION.md, work/*.safetensors ; run-script : work/ et scripts/) ; lisible comme tout l'espace, sans effet (copie de la base). Une sauvegarde d'un détachement journalisé n'est jamais restaurée (un seul détachement par séance, `_recover` nettoie seulement) ; un `.tmp` laissé dans `detach-bak/` est déplacé puis retiré de work/ dans le même appel.
- Nit · low · revue · reporté : processus tué avant la ligne de journal, puis l'éditeur écrit dans `work/model.safetensors` avant de relancer `detach-lm-head` : `_recover` restaure la sauvegarde et efface ces écritures sans le dire. Cas rare (état déjà incohérent, refusé au gel) ; piste : ne restaurer le tenseur que si le fichier de work/ manque ou contient `lm_head.weight`, et l'écrire dans la sortie.
- Nit · low · revue · reporté : si `_append_journal` lève après avoir écrit la ligne (fsync), work/ est restauré alors que le journal dit « détaché » : gel et détachement refusés. Fenêtre très étroite, même famille que la passe 1.
- Nit · low · revue · corrigé : docstrings de `workspace.py` disaient « quatre commandes » (cinq désormais).
- Complexité : 0 constat.
