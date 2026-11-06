---
status: decided
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
