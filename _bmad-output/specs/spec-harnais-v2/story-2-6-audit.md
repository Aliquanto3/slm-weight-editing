---
status: done
story: "2.6"
title: Audit sans les faux positifs de s1, signalement du décodage autorégressif (CAP-9, audit de CAP-3)
---

# Story 2.6 : audit v2

Contrat : CAP-9 et la partie audit de CAP-3 de `SPEC.md`, `incidents-s1.md` (ligne « Faux positifs de l'audit »), entrée `id: "6"` de `stories.yaml`. Fiche à relire avant le code.

## But

L'audit ne signale plus les usages légitimes vus en s1 (`re.compile`, décodage des octets du tokenizer, Grep sur le journal de l'espace) et échoue toujours sur les violations réelles. Il dresse en plus la liste des copies de scripts et des suites d'appels probe qui pourraient faire du décodage autorégressif, pour qu'Anaël les relise avant la mesure.

## Critères (Given / When / Then)

1. **Given** une copie de script avec `re.compile(r"\d+")`, **when** `audit_scripts`, **then** aucun constat `obfuscation`. **Given** `compile(src, "x", "exec")` nu ou `builtins.compile(`, **then** `obfuscation` reste.
2. **Given** le décodage des octets du tokenizer (forme synthétique tirée de s1, par exemple la table `bytes_to_unicode` en `chr(` et `bytes([...]).decode("utf-8")`), **when** il est dans une copie de script, **then** aucun constat.
3. **Given** `Grep` (ou `Read`, `Glob`) sur `sessions/<id>/journal.jsonl` en chemin relatif ou absolu dans l'espace, **when** `check_tool_use`, **then** aucun constat (CAP-5).
4. **Given** un forward numpy génératif synthétique (boucle : logits, `argmax` ou tirage, jeton ajouté à l'entrée de l'itération suivante), **when** `audit_scripts`, **then** la copie figure dans la liste « à relire » avec la ligne et le motif. **Given** le même forward rejoué en forward forcé sur une séquence fixe, **then** elle n'y figure pas.
5. **Given** un script qui échantillonne (`np.random.choice` sur des probabilités, `rng.multinomial`, `softmax` puis tirage), **then** il est listé.
6. **Given** au journal au moins k appels `probe` successifs dont chaque texte prolonge d'un jeton un texte de l'appel précédent, **when** l'audit lit le journal, **then** la suite est listée (numéros d'appels, longueur). **Given** k textes préfixes l'un de l'autre dans un seul appel, **then** rien n'est listé : les textes sont fixés avant le calcul.
7. **Given** les 73 `VIOLATIONS`, les `SCENARIOS` et les copies des 28 scripts interdits, **then** tous restent en échec, nommés avec leur ligne ; `test_clean_transcript_passes` reste vert.
8. **Given** la transcription et le journal de s1 rejoués hors dépôt, **then** il ne reste que la génération numpy et les violations mineures connues (vérification manuelle, question 2).

## Fichiers et fonctions touchés (vérifiés au 6de87e6)

- `audit_transcript.py` :
  - `_OBFUSCATION` : `\bcompile\s*\(` attrape `re.compile(`. Correction ciblée : `(?<![\w.])compile\s*\(` plus `builtins\.compile` ; ou liste blanche `re.compile(` ;
  - `_OBFUSCATION` et `check_code_content` : `\bchr\s*\(`, `codecs.decode`, `bytes.fromhex` sont des candidats pour le décodage d'octets. Le motif exact de s1 reste à confirmer (question 2) ;
  - `_check_paths`, `forbidden_reference`, `broad_reference` : origine du faux positif Grep inconnue. Hypothèses : `_SHORT_NAME` sur un chemin Windows en `~1` (`UTILIS~1`), ou un motif Grep qui contient un terme de `_FORBIDDEN_COMPACT`. À reproduire avant de corriger, sans affaiblir la règle 8.3 hors de l'espace ;
  - nouvelle `review_candidates(session) -> list[Candidate]` (copies et suites de probes), appelée par `audit-session`, avec sortie dans une section « À relire » et dans `--json` ;
  - `RULES` : nouvelle entrée `decodage-autoregressif` (statut : question 1).
- Nouveau module facultatif `audit_decode.py` (analyse AST des copies), pour ne pas grossir `audit_transcript.py` (974 lignes).

## Heuristique (statique, sans garantie : non-goal de SPEC)

- Analyse `ast` de chaque copie. On signale une boucle `for`/`while` dont le corps (a) calcule un indice par `argmax`, `argsort`, `np.random.*`, `rng.*` ou `choice`, (b) l'ajoute à une séquence (`append`, `np.concatenate`, `np.append`, `+=`, affectation indicée), (c) passe cette séquence à un appel du même corps. Tout tirage aléatoire sur des probabilités est aussi signalé.
- Si la copie ne se lit pas en AST, elle est listée « illisible » (repli prudent).
- Un forward forcé (boucle sur les positions d'une séquence fixe, `argmax` seulement pour mesurer) n'est pas signalé : la séquence n'est pas réécrite dans la boucle.
- Suites de probes : comparaison des `token_ids` des manifests de la story 2.4 entre appels consécutifs. Signal renforcé si le jeton ajouté est le premier des 20 meilleurs de l'appel précédent.

## Choix de conception

| Point | Options | Recommandation |
|---|---|---|
| Statut d'un signalement | (a) violation, code 1 ; (b) liste à relire, code 0 si rien d'autre ; (c) code distinct (4) | question 1 ; (c) sépare « à relire » de « échec » sans rien cacher. |
| `re.compile` | lookbehind ; liste blanche | liste blanche `re.compile(` et `regex.compile(` : plus étroit. |
| Lecture du journal par script | inchangé ; `etat-seance` seulement sur une écriture visant le nom | à corriger ici (limite v1 de la story 5, que la lecture libre de 2.5 rendra courante) : question 4. |
| Cas de s1 | synthétiques dans les tests ; rejeu réel hors dépôt | les deux : synthétiques versionnés, rejeu réel noté dans la fiche sans aucun extrait. |

## Tests

- À ajouter (`tests/test_audit_transcript.py`, ou `tests/test_audit_decode.py`) : critères 1 à 7, une paire de scripts (génératif et forcé) écrite à la main d'après la forme de `qwen_np.py` et `qwen_batch.py` sans copier leur code, des contournements simples de l'heuristique documentés comme limites, et le format de la section « À relire ».
- À réaligner : `VIOLATIONS` et `SCENARIOS` si un motif change (aucune entrée ne doit passer en succès), `test_script_copies_are_audited`, `test_clean_transcript_passes`.

## Garanties à préserver

- 73 transcriptions de violation, 28 scripts interdits (par leur copie) et 10 contournements toujours en échec ; chaque assouplissement est couvert par un test négatif voisin (`compile(` nu, `builtins.compile`, `chr(` en commande shell).
- `lecture-hors-espace` inchangée hors de l'espace ; règle 8.3 inchangée hors de l'espace.

## Dépendances

- Story 2.4 : événements `probe` et `token_ids` des manifests, nécessaires pour le critère 6. Story 2.3 : forme de `--thinking` dans la liste blanche. Story 2.5 : périmètre de la lecture libre.
- `audit_transcript.py` est aussi touché par 2.3 et 2.4 (liste blanche) : faire 2.6 après elles, ou réaligner.

## Questions pour Anaël

1. Une copie de script ou une suite de probes signalée fait-elle échouer l'audit, ou seulement apparaître dans une liste à relire (code de sortie distinct) ? Tu tranches ensuite à la relecture.
2. Les trois faux positifs de s1 : peux-tu donner leur forme (appel d'outil, motif, sans contenu privé), ou autorises-tu l'agent à rejouer l'audit sur la transcription s1 locale sans rien committer ?
3. Seuil des suites de probes : un jeton ajouté exactement, ou de 1 à k jetons ? À partir de combien d'appels consécutifs ?
4. Corriger dans cette story le faux positif `etat-seance` d'un script qui lit le journal par `open` et écrit `work/` ?
5. L'acceptation de `chr(` et du décodage d'octets vaut-elle seulement pour le code (copies, outils d'écriture), en gardant le refus dans les commandes shell ?

## Décisions de l'orchestrateur (2026-11-06)

Anaël a levé ses points d'arrêt de relecture le 2026-11-06 (« Retire mes points d'arrêt et continue »). Les réponses ci-dessous sont prises par l'orchestrateur, au plus prudent ; Anaël peut les revoir à la relecture des PR.

1. Une copie de script ou une suite de probes signalée apparaît dans une liste à relire, avec un code de sortie distinct (0 propre, 1 violation, 2 à relire seulement). Elle ne fait pas échouer l'audit.
2. L'agent peut rejouer l'audit sur la transcription s1 locale, sans rien commiter. Les tests reproduisent la forme des faux positifs (outil, motif) sur des données synthétiques, sans contenu privé.
3. Suite signalée : de 1 à 4 jetons ajoutés à chaque appel, à partir de 3 appels consécutifs.
4. Oui, ce faux positif `etat-seance` est corrigé dans cette story.
5. Oui : `chr(` et le décodage d'octets sont acceptés seulement dans le code (copies de scripts, outils d'écriture). Le refus reste dans les commandes shell.

## Réalisation (2026-11-06) et écarts

`audit_transcript.py` : `re.compile(` et `regex.compile(` acceptés (`compile(` nu et `builtins.compile` refusés) ; `chr(` et `bytes.fromhex` acceptés seulement hors shell (`_BYTE_DECODING`) ; `_without_read_lines` : dans le code, une ligne qui ne fait que lire (`open` en lecture, `read_text`, `read_bytes`) ne compte plus pour `etat-seance`. Nouveau `audit_decode.py` (`scan_script` en AST, `scan_probe_suites`, `review_candidates`), appelé par `audit-session` : section « À relire », `--json`, règle `decodage-autoregressif` dans `RULES` (jamais rendue par `check_tool_use`, donc jamais refusée par le crochet). Tests : `tests/test_audit_decode.py`.

Écarts :
- Codes de sortie : 0 propre, 1 violation, 2 à relire seulement (décision 1) ; l'ancien code 2 « entrée illisible » passe à 3 (tests réalignés).
- Critère 3 (Grep sur le journal) : déjà vert depuis la story 2.5 ; seuls des tests de non-régression sont ajoutés. Faux positif d'origine non reproduit.
- Critère 8 (rejeu de la transcription s1) : non fait ici, aucune transcription s1 n'a été cherchée ; à faire par Anaël à la relecture.
- Limites de l'heuristique : une affectation du nom protégé (`p = "journal.jsonl"`) ou un `open` sur plusieurs lignes reste signalé `etat-seance` ; une séquence portée par un attribut d'objet ou une fonction appelée hors de la boucle échappe à la détection ; signal « jeton parmi les 20 meilleurs » non implémenté.
- `allow_abbrev=False` sur les sous-parseurs (report de la revue 2.5) : laissé, change la CLI. `ALLOWED_SUBCOMMANDS` non touchée.

## Revue

### Passe 1 (2026-11-06, SHA revu e0a1cf8)

Base : `origin/main` (2d79040, story 2.8) fusionnée sans conflit avant la revue.

- Important · medium · second avis 1 · corrigé · `_without_read_lines` retirait une ligne d'écriture prise pour une lecture (`open(Path("…journal.jsonl").resolve(), "a")` : `[^)]*` s'arrête à la parenthèse imbriquée) ; même perte sur `p = Path(j); p.read_text()` et `(p := Path(j)).read_text()` suivis de `p.write_text`. L'exemption passe par l'AST (`_line_only_reads`) : une instruction, chaque `open` en mode littéral de lecture, ni `:=` ni affectation d'autre chose qu'un résultat de lecture ; tout le reste reste compté. S'applique aussi au crochet.
- Important · medium · second avis 2 · corrigé · formes de décodage manquées : `max(..., key=...)`, alias `from numpy import argmax as am`, tirage uniforme (`rng.random()`, cumul inverse) ajoutés à l'heuristique.
- Important · medium · second avis 3 · corrigé · une sonde sans rapport intercalée coupait la suite : suites suivies par chaîne de préfixes (index des textes vus).
- Nit · low · second avis 4 · corrigé · rescorage des mêmes préfixes à chaque appel compté comme suite : un texte déjà vu ne prolonge plus rien.
- Nit · low · second avis 5 · corrigé en partie · `choice`/`choices` sans poids (`p`, `weights`) ne sont plus signalés seuls ; `seen.append(pred)` repassé à un appel reste listé (liste à relire, faux positif accepté).
- Nit · low · second avis 6 · rejeté · code 2 partagé avec l'erreur d'usage d'argparse : décision 1 de l'orchestrateur ; une erreur d'usage écrit sur stderr sans sortie standard, ce qui la distingue.
- Nit · low · second avis 7 · rejeté · le lookbehind actuel `(?<!\bre\.)(?<!\bregex\.)` est plus strict que celui de la fiche (`(?<![\w.])`, qui accepte tout `x.compile(`) ; `mod.re.compile(` passe sous les deux.
- Pre-existing · low · revue · reporté · alias de fonctions interdites (`from builtins import compile as c`, `import builtins as re`) : l'audit textuel ne suit pas les alias, avant comme après la story.
- Complexité : 1 constat, corrigé · `_extends` comparait toutes les paires de textes de deux appels ; l'index des textes vus ramène les suites à O(textes × 4).
- Report · critère 8 (rejeu de la transcription s1) toujours à faire par Anaël.

### Passe 2 (2026-11-06, SHA revu a56afda)

Relu : `git diff e0a1cf8..a56afda`. Batterie de 36 formes d'écriture contre `check_code_content` (`r+`, `rb+`, mode conditionnel, `mode=`, `io`/`codecs`/`Path.open`, `os.fdopen`, `opener=`, chaînage `or`, `lambda`, compréhension, `with a, b`, lignes physiques multiples, `\`, `;`, `\n` dans une chaîne, code cassé) : aucune régression par rapport à e0a1cf8 ; les trous ci-dessous existaient déjà ou tiennent au correctif de la passe 1.

- Important · medium · revue · corrigé · le correctif 1 de la passe 1 ne tenait pas seul : `open(Path("…journal.jsonl").resolve(), "a")` sans `.write` sur une autre ligne n'était pas signalé, le déclencheur `_open_writes` gardant `[^)]*` (même cas `open(str(j), "w")`). `_open_writes` ajoute l'AST (texte entier, sinon chaque ligne analysable) à la regex ; `os.open(p, os.O_RDONLY)` reste une lecture.
- Important · medium · revue · corrigé · `_line_only_reads` acceptait tout appel autre qu'`open` sur une ligne de lecture et ne regardait que l'affectation de tête : `os.link(j, open(n).read())`, `open(j, opener=ev)` (ou `opener` en 8e position), `for p in [j]: …`, `with open(j) as f: name = f.name`, `x = [p for p in [j] if …]`, `d[j] = …` passaient. Liste blanche d'appels, `open` limité à `mode`/`buffering`/`encoding`/`errors`/`newline` et trois positionnels, toute affectation de l'arbre vérifiée (cibles nommées seulement), itérables de `for` et de compréhension limités à un nom ou un résultat de lecture, ni `def`/`lambda`/`import`.
- Important · low · revue · corrigé · `scan_probe_suites` : un rescorage intercalé réattribuait le texte à sa propre chaîne (`owner[key] = chain`) et coupait la suite (1, 2, 3 rescorage, 4 : rien signalé). `owner.setdefault`.
- Nit · low · revue · reporté · `f = open(j)` ou `with open(j) as f:` puis `open(f.name, "w")` ailleurs : le nom du fichier lu reste joignable par `.name` ; refuser `with open(j) as f` rendrait au faux positif le cas courant de CAP-9.
- Pre-existing · low · revue · reporté · déclencheur textuel : `sqlite3.connect(j)` ou `zipfile`/`shelve` sans verbe d'écriture reconnu ne déclenchent pas la recherche du nom protégé, avant comme après la story.
- Complexité : 0 constat · `scan_probe_suites` reste en O(textes × `MAX_ADDED` × longueur) ; `_parse_trees` parse le texte une fois, ligne par ligne seulement s'il est mal formé.
- Vérifié sans constat : `_resolve_aliases` (alias `from … import … as` des seuls tireurs), tirages uniformes, `_texts_of` sur manifeste bizarre (dict, liste, `texts` non liste : `None`), listes vides.

### Passe 3 (2026-11-06, SHA revu 708b056)

Relu : `git diff a56afda..708b056`. `origin/main` (2d79040) déjà ancêtre, pas de fusion. Deux batteries : 20 formes de fuite d'un chemin protégé hors de la lecture, 18 scripts d'analyse légitimes (lecture du journal, de `state.json` ou `workspace.json`, écriture dans `work/` ou `notes/`) passés à `check_code_content` et au crochet (`_refusal_hook.decide`).

- Important · medium · revue · corrigé · faux positifs CAP-9 encore courants (8 scripts sur 18 refusés, donc bloquants en `dontAsk`) : liaison `j = "…journal.jsonl"` ou `J = Path(…)` puis lecture par `j`, `n = sum(1 for _ in open(j))`, `enumerate(open(j))`, `print(json.dumps(...))` (`json\.dump` attrapait `dumps`), `pd.read_json(j)`. Une ligne de lecture n'est plus retirée mais effacée seulement aux arguments lus (`open`, `Path(…).read_text`, `pd.read_*`, `np.load` sans `mmap_mode` d'écriture) et aux arguments de `print` sans `file=` ; une liaison d'un chemin protégé est oubliée si chacun de ses autres emplois est ainsi effacé (fichier complet seulement : `Edit` reste compté, liaison sans emploi aussi). Affectations et itérables libérés de la contrainte « résultat de lecture », la garde passant par l'effacement.
- Important · medium · revue · corrigé · trous de la liste blanche de la passe 2 : `[(Path(j), l) for l in open(j)]` puis `open(x[0][0], "w")`, `paths.append((Path(j), Path(j).read_text()))`, `with open(j) as f` puis `open(f.name, "w")` (Nit reporté en passe 2), fonction homonyme `def load(p)` avec `load(open(j).name)`. Effacement ciblé ci-dessus ; aucune ligne allégée si le texte contient `.name`/`.fileno`/`.buffer`/`.raw`, `globals`/`getattr`/`vars`/`__dict__`/`sys.modules`/`inspect`/`gc`, ou redéfinit `open`, `print`, `Path`, `os`, `json`, `np`, `pd`, une méthode admise (`Path.read_text = …`, `def read`), `import *`.
- Nit · low · revue · accepté · prix de la prudence : un script qui lit l'état et écrit ailleurs reste refusé s'il emploie `p.name` (souvent sur un `glob`) ou `map(json.loads, open(j))` ; lecture sur plusieurs lignes physiques également.
- Pre-existing · low · revue · reporté · `\beval\b` d'obfuscation refuse `r["kind"] == "eval"` (chaîne) ; `np.save`, `df.to_csv`, `np.load(…, mmap_mode="r+")` ne déclenchent pas la recherche du nom protégé ; liaison dans un autre fichier (`from paths import J`, module local homonyme de `json`) invisible pour un audit par fichier. Avant comme après la story ; `audit_scripts` relit le script exécuté complet.
- Complexité : 1 constat, corrigé · la recherche des emplois d'une liaison parcourt toutes les lignes ; restreinte aux liaisons d'un nom protégé (300 liaisons sur 600 lignes : 0,24 s).
