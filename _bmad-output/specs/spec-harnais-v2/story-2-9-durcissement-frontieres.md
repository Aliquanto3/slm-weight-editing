---
status: done
story: "2.9"
title: Durcissement des frontières 2.5/2.6/2.8 (constats A-12 ; CAP-6, CAP-9, CAP-10)
---

# Story 2.9 : durcissement des frontières 2.5/2.6/2.8

Contrat : CAP-6, CAP-9 et CAP-10 de `SPEC.md`, fiches 2.5, 2.6 et 2.8, action A-12 de `epic-2-retro-2026-11-06.md`. Story à risque (sécurité) : fiche à relire avant le code.

## But

Fermer six brèches entre le crochet (2.5), l'audit (2.6) et le détachement de `lm_head` (2.8) relevées par A-12, sans affaiblir aucun refus existant.

## Critères (Given / When / Then)

### 1. Écriture de l'état de séance sans verbe reconnu

Aujourd'hui `_check_code` ne cherche une cible protégée que si `_WRITE_VERB` ou `_open_writes` répond. `ndarray.tofile`, `pickle.dump`, `os.open(..., os.O_WRONLY)`, `np.memmap(..., mode="w+")` ou `os.truncate` passent.

1. **Given** un script de `scripts/` qui écrit dans `sessions/<id>/state.json` (ou `journal.jsonl`, `registry.jsonl`, `session.lock`, `workspace.json`) par chacune de ces cinq formes, **when** `check_tool_use("Write", …)` puis `audit_scripts`, **then** chaque cas donne `etat-seance`.
2. **Given** la même mention dans un script sans aucun appel d'écriture (chaîne affectée, puis passée à une fonction maison), **then** `etat-seance` aussi : la lecture du journal passe par l'outil Read (CONSIGNE l. 34), pas par un script.
3. **Given** `Read`, `Grep` ou `Glob` sur `sessions/<id>/journal.jsonl`, **then** aucun constat (critère 3 de la fiche 2.6 inchangé).

### 2. Crochet en échec, délai et `_SHADOWING`

4. **Given** une exception dans `decide` ou `context` (`workspace.json` absent, JSON illisible, entrée non UTF-8), **when** `main`, **then** sortie `deny` avec la règle `crochet-en-echec` et le rappel BLOCAGES ; jamais `allow` ni `ask`.
5. **Given** le bloc rendu par `create_workspace`, **then** le crochet porte `onFailure: "block"`, donc un délai dépassé ou un code autre que 0 et 2 bloque l'appel.
6. **Given** une analyse plus longue qu'une échéance interne inférieure à `HOOK_TIMEOUT` (10 s), **then** le crochet rend `deny` (`crochet-en-echec`, « analyse trop longue ») avant le délai de Claude Code.
7. **Given** un contenu de 200 ko construit pour la pire branche de `_SHADOWING` (longues suites `for a, b, c, …` sans `in`, `lambda` sans `:`), **when** `check_code_content`, **then** réponse en moins de 0,5 s ; le temps reste linéaire quand la taille double (mesure sur 100 et 200 ko, rapport < 3).
8. **Given** les cas de redéfinition déjà testés (`print = …`, `def open`, `for open in`, `lambda Path:`), **then** tous restent refusés.

### 3. Prose de `EDITION.md` et `notes/` auditée comme du code

9. **Given** un `Write` ou `Edit` de `EDITION.md` ou de `notes/x.md` dont le texte cite `rm`, `git`, `state.json`, `heldout`, `open(p, "w")` ou `chr(`, **when** `check_tool_use` et le crochet, **then** aucun constat de règle de code ; les règles de chemin (`ecriture-hors-liste`, `etat-seance` sur le chemin cible, `harnais-modifie`) s'appliquent toujours.
10. **Given** le même texte dans `scripts/x.py`, **then** les constats actuels restent.
11. **Given** `notes/x.py`, ou un script de `scripts/` qui lit puis exécute un fichier de `notes/` (`exec`, `runpy`, `importlib`, ajout de `notes` à `sys.path`), **then** refus (`ecriture-hors-liste` pour un `.py` hors de `scripts/`, `obfuscation` ou `api-harnais` pour l'exécution).

### 4. LSP et `filePath`

12. **Given** un appel `LSP` dont `filePath` vise un fichier hors de l'espace (dépôt, `heldout/`, parent, jonction posée dans l'espace), **when** `check_tool_use` et le crochet, **then** `lecture-hors-espace`.
13. **Given** `filePath` dans l'espace, **then** aucun constat. Toute clé de chemin d'un outil de lecture (`file_path`, `path`, `notebook_path`, `filePath`) est contrôlée de la même façon ; test paramétré sur les quatre.

### 5. Reprise de `detach-bak` et ligne de journal

14. **Given** une séance où `<séance>/detach-bak/` existe sans ligne `detach_lm_head` au journal (processus tué), **when** `session-unlock`, `session-freeze`, `run-batch` ou `probe`, **then** `work/` est remis en l'état d'avant le détachement avant toute autre action, et la sortie le dit (ou la commande refuse : question 2) ; `work/` haché au gel égale celui d'avant la tentative.
15. **Given** `detach-bak/` avec la ligne `detach_lm_head` au journal, **then** la sauvegarde est supprimée sans restaurer ; le gel accepte la séance déliée.
16. **Given** une exception levée après l'écriture complète de la ligne de journal (simulée dans `_append_journal` après écriture, ou `KeyboardInterrupt` juste après), **when** `detach_lm_head`, **then** `work/` reste délié, conforme au journal, et `detach-bak/` est supprimé ; un `detach-lm-head` suivant refuse « déjà détaché ».
17. **Given** une exception avant la ligne de journal, **then** comportement actuel : `work/` restauré, aucune ligne.

### 6. `detach-lm-head` dans un espace réel

18. **Given** un espace créé par `create_workspace` sur un checkpoint synthétique lié (`tie_word_embeddings: true`), **when** la commande `detach-lm-head` passe au crochet puis s'exécute depuis l'espace, **then** le crochet ne refuse pas, l'événement est au journal, `detach-bak/` n'existe plus, `session-freeze` réussit et `audit-session` ne rend aucun constat (ni lien, ni état de séance, ni fichier en trop dans `work/`).

## Fichiers et fonctions touchés (vérifiés au 62a83d2)

- `audit_transcript.py` : `_check_code` (l. 688, condition `_WRITE_VERB` l. 739), `_mentions_protected` (l. 304), `_SHADOWING` (l. 459), `check_tool_use` (l. 1035, branche `FILE_WRITE_TOOLS`), `_check_paths` (l. 1017, clés lues l. 1020), `RULES` et `refusal_message` (règle `crochet-en-echec`).
- `_refusal_hook.py` : `main` (refus sur exception au lieu de `return 0`), échéance interne ; docstring à réécrire (elle décrit l'échec ouvert).
- `workspace.py` : bloc du crochet (l. 164), `onFailure: "block"`.
- `lm_head.py` : `_recover` (l. 150) exposé pour les autres commandes, bloc `except BaseException` (l. 244). `harness.py` (`unlock_session` l. 246, `run_batch` l. 779), `freeze.py` (`freeze_session` l. 310), module de probe : appel de la reprise sous le verrou. `lm_head` importe `harness` : placer l'appel sans import circulaire (import local ou petit module commun).
- Tests à réaligner : `test_espace_lecture.py` (« stdin malformé ou exception : code 0 sans sortie » devient `deny`), tests du crochet et de l'audit, `test_lm_head.py`, `test_freeze.py`.

## Contraintes

- Aucun refus existant ne s'affaiblit : 73 `VIOLATIONS`, `SCENARIOS`, 28 scripts interdits, 10 contournements restent en échec ; le crochet ne rend jamais `allow` ni `ask`.
- Tests synthétiques seulement, sans modèle ni téléchargement ; mesure de temps du critère 7 tolérante en CI (seuil large, pas de micro-benchmark fragile).

## Non-goals

- Refonte de l'audit en AST complet (A-11 reste à part).
- Constat 2.8 reporté « `_recover` efface sans le dire des écritures faites après un détachement tué » : seule la sortie qui le dit (critère 14) est dans cette story.
- Relance de la passe `bmad-review` complète sur `d4fa5ab..9204b55`.

## Questions pour Anaël

1. Le crochet passe en échec fermé (critères 4 à 6) : cela annule la décision 2 de la fiche 2.5, que tu avais validée. Une panne du crochet bloque alors toute la séance jusqu'à ton intervention. D'accord ?
2. Critère 14 : reprise automatique avec message, ou refus de la commande tant que `detach-bak/` existe ? Recommandé : reprise avec message pour les quatre. Un refus ne laisse à l'éditeur que `detach-lm-head`, qui reprend puis détache : il serait poussé à délier sans l'avoir choisi.

## Réalisation (dev-story, 2026-11-06)

Tests : `tests/test_durcissement_frontieres.py`, un test (ou groupe paramétré) par critère, nommés `test_cN_…` ; tests réalignés dans `tests/test_espace_lecture.py` (entrée malformée et `workspace.json` absent : `deny crochet-en-echec` ; bloc du crochet avec `onFailure`).

- Point 1 : dans le code (hors shell), plus de condition de verbe ; toute mention d'une cible d'état de séance donne `etat-seance`. Les lignes prouvées en lecture seule par l'AST (`open(p).read()`, `Path(p).read_text()`, `print`…) restent allégées, comme avant : sans cela, les scripts d'analyse de CAP-9 (`LEGIT_ANALYSIS_SCRIPTS`) seraient refusés et `_SHADOWING` (critères 7 et 8) ne servirait plus. Lecture non prouvée (`pathlib.Path(p).open().read()`) : désormais refusée (test déplacé dans les cas refusés).
- Point 2 : `_refusal_hook.main` analyse dans un fil démon borné par `DEADLINE` (6 s, sous `HOOK_TIMEOUT` = 10 s) ; exception, entrée illisible ou échéance : `deny crochet-en-echec` + rappel BLOCAGES ; sortie impossible : code 2. `_SHADOWING` découpé en un motif à essais bornés et des segments possessifs (`_shadows`), équivalence vérifiée sur 300 000 chaînes aléatoires contre l'ancien motif. `_without_read_lines` n'analyse plus que les lignes qui nomment l'état de séance. Seuil du critère 7 : 0,5 s, ×3 si `CI` est posé.
- Point 3 : `_is_prose` (EDITION.md, `notes/` hors `.py`, `.pyw`, `.pyc`, `.pth`, `.ipynb`…) : règles de chemin seulement. Un fichier de code hors de `scripts/` est `ecriture-hors-liste`. Ajout : `sys.path`, `meta_path`, `path_hooks`, `PYTHONPATH`, `from sys import path` donnent `obfuscation` dans le code.
- Point 4 : `READ_PATH_KEYS` = `file_path`, `path`, `notebook_path`, `filePath`.
- Point 5 : `lm_head.recover_interrupted` + `harness.recover_detach` (import local, message `Reprise : …` sur stderr), appelé sous le verrou par `session-unlock` (sous le verrou mort, ou un verrou pris s'il n'y en a pas), `session-freeze`, `run-batch` et `probe`. Après une erreur, `detach_lm_head` relit le journal : ligne présente, la sauvegarde part sans restaurer ; journal illisible, rien n'est touché.
- Reports : la phrase de `CONSIGNE.md` « tout ce dossier se lit librement… par tes scripts » reste vraie pour les lectures prouvées mais ne dit pas qu'une autre mention de l'état de séance est refusée : à préciser dans une story de doc.

## Revue

### Passe 1 (2026-11-06, SHA revu 909481a)

Base à jour (merge-base 62a83d2 = origin/main), pas de rebase.

- Important, high, second avis 1 : `exec` atteint sans `exec(` littéral (`b.__dict__["exec"]`, `vars(builtins)`, `sys.modules`, `getattr(x, "ex"+"ec")`, `print.__self__`, `timeit`, `code`, `pdb`, `read_pickle`, `site.addsitedir`) : un script exécutait la prose de `notes/` sans constat (critère 11). Corrigé : `_DYNAMIC_EXEC` (obfuscation) dans `check_code_content` ; un refus de toute mention de `notes/` aurait fait tomber les `LEGIT_ANALYSIS_SCRIPTS` qui y écrivent et `probe --texts notes/…`. 12 cas ajoutés au test c11.
- Important, medium, second avis 2 : un échec hors de `main` (import de l'audit, `thread.start`, flush) sortait en code 1, non bloquant si `onFailure` n'est pas reconnu. Corrigé : `_blocked` (code 2, motif sur stderr) autour de l'import et du `__main__` ; test en sous-processus `-S`.
- Important, medium, second avis 3 : journal illisible à la reprise : `JSONDecodeError` brute, et `unlock` sans verrou l'avalait. Corrigé : `harness.RecoveryError` (HarnessError), rien n'est touché ; test paramétré sur les quatre commandes. Les autres lectures du journal (`used_batches`, gel) lèvent toujours `JSONDecodeError` : Pre-existing, hors story.
- Nit, low, second avis 4 : échéance comptée depuis `main` : rejeté, le démarrage et l'import tiennent largement dans les 4 s de marge sous `HOOK_TIMEOUT`.
- Nit, low, second avis 5 : seuil de temps du critère 7 : rejeté, seuil ×3 sous `CI` et rapport < 3, déjà tolérants.
- Écart (a) du développeur (lignes en lecture prouvée exemptées, `Path(p).open().read()` refusé) : accepté, aucun refus affaibli. Écart (b) (`sys.path`… et fichiers de code hors de `scripts/`) : accepté, il durcit le critère 11.
- Report : `onFailure: "block"` (Claude Code v2.1.295+) non vérifiable sans réseau ; à vérifier dans une vraie séance : un crochet qui sort en code 1 doit bloquer l'appel.
- Complexité : 0 constat (`_DYNAMIC_EXEC` en temps linéaire).
