---
status: draft
story: "2.5"
title: Espace de séance, lecture libre, OPERATEUR.md, refus expliqués, BLOCAGES.md (CAP-4, CAP-5, CAP-6, consigne de CAP-3)
---

# Story 2.5 : espace de séance v2

Contrat : CAP-4, CAP-5, CAP-6 et la consigne de CAP-3 de `SPEC.md`, entrée `id: "5"` de `stories.yaml`. **Story à risque** (sécurité de l'espace) : fiche à relire avant le code.

## But

Chaque refus dit quelle règle il applique et ce qui est permis à la place. L'éditeur lit librement tout l'espace, par ses outils comme par ses scripts, et note ce qui le bloque dans `notes/BLOCAGES.md`. Les instructions d'Anaël sortent de l'espace. La consigne interdit le décodage autorégressif dans `run-script`.

## Ce que confirme la documentation officielle (lue le 2026-11-05)

| Point | Phrase exacte et source | Conséquence |
|---|---|---|
| Une règle deny ne porte pas de motif | La seule forme de message montrée par la doc est « `Permission to use mcp__workspace__bash has been denied.` » (https://code.claude.com/docs/en/permissions). Rien ne dit qu'on peut personnaliser ce message. | L'explication doit venir d'ailleurs. Que le texte soit identique pour `Bash` reste **à vérifier**. |
| Un crochet PreToolUse rend un motif | « `"deny"` prevents the tool call » ; pour `permissionDecisionReason` : « For `"deny"`, shown to Claude. » (https://code.claude.com/docs/en/hooks) | Un crochet peut expliquer un refus. |
| Le code 2 et stderr | « A hook that blocks by exiting 2 routes the same way as `"deny"`: Claude sees the stderr message as the denial reason. » (hooks) | Deux formes possibles ; on choisit le JSON, plus net. |
| Le crochet passe avant les règles | « A hook that exits with code 2 stops the tool call before permission rules are evaluated, so the block applies even when an allow rule would otherwise let the call proceed. » (permissions) | Quand le crochet refuse, Claude voit son motif et non le refus muet. |
| Un crochet n'affaiblit pas les deny | « PreToolUse hook decisions don't bypass permission rules. Claude Code evaluates deny and ask rules regardless of what a PreToolUse hook returns » (permissions) | Les deny de la v1 restent le filet. |
| Un crochet « allow » élargit les droits | « `"allow"` skips the permission prompt » (hooks) ; `dontAsk` « Auto-denies every call that would otherwise prompt » (permissions) | Un crochet qui répondrait `allow` laisserait passer ce que `dontAsk` refuse. **Le crochet ne répond jamais `allow`.** |
| Un crochet en échec laisse passer | « The action goes ahead » sur un code autre que 0 et 2 ; à l'expiration du délai, « doesn't block the tool call » (hooks) | Le crochet sert à expliquer, il ne garde pas la porte. |
| Les crochets tournent sous Windows | « `sh -c` on macOS and Linux, Git Bash on Windows, or PowerShell when Git Bash isn't installed » ; forme exec : « On Windows, exec form requires `command` to resolve to a real executable such as a `.exe`. » (hooks) | Forme exec possible avec `python.exe`. Chemins reçus « with backslash separators » (hooks). |
| Les crochets d'un dossier non approuvé s'exécutent | Tableau « What runs before you trust a folder » : les crochets des fichiers de réglages sont « Used » dans les deux colonnes (permissions) | Ceux de l'espace tournent sans approbation ; ceux des réglages utilisateur aussi. |
| `dontAsk` et les crochets | « PreToolUse hooks run before the permission prompt, for every tool except `EndConversation` » (permissions). La doc ne dit pas explicitement qu'ils tournent en `dontAsk`. | **À vérifier** par l'essai à blanc. |
| `blockReads…` et les crochets | Rien de dit sur leur combinaison. | **À vérifier** (on s'attend à ce que le crochet refuse avant, avec un motif). |
| Mods personnels | « With neither managed settings nor a Team or Enterprise `/login` as the session's credential, the mod can approve a call that a deny rule refuses or a `PreToolUse` hook blocked. » (permissions) | Risque qui touche **aussi la v1**. Anaël vérifie avant la séance qu'aucun mod n'est installé. |
| Read et Grep, Glob | « Claude makes a best-effort attempt to apply `Read` rules to all built-in tools that read files like Grep and Glob » ; « On Windows, paths are normalized to POSIX form before matching. `C:\Users\alice` becomes `/c/Users/alice` » (permissions) | Un deny `Read(//c/…/OPERATEUR.md)` est possible en plus des autres couches, mais il n'est appliqué qu'« au mieux » (best-effort). |

## Critères (Given / When / Then)

1. **Given** un espace créé, **when** un script lit, liste ou ouvre en `mmap` n'importe quel fichier de l'espace (`notes/`, `EDITION.md`, `CONSIGNE.md`, journal, `probes/`, `runs/`), **then** code 0 ; **when** il lit hors de l'espace (dépôt, `heldout/`, parent, jonction posée dans l'espace vers le dépôt), **then** code 3.
2. **Given** `session-workspace`, **then** `OPERATEUR.md` est écrit hors de l'espace (emplacement : question 1), avec les commandes de gel, d'audit, de déverrouillage et de mesure de cette séance, et aucun fichier de ce nom n'existe dans l'espace. Le lanceur et l'audit refusent sa lecture, `blockReads…` et un deny `Read` la bloquent.
3. **Given** chaque famille de refus (tableau ci-dessous), **when** elle se déclenche, **then** le message contient l'identifiant de la règle et la voie permise ; un test par famille.
4. **Given** le crochet de refus, **when** il reçoit sur l'entrée standard un appel que les deny refusent (`cat x`, `python -c 1`, `Read` hors de l'espace, `WebFetch`), **then** il sort une décision `deny` avec un motif ; **when** il reçoit un appel permis, **then** il sort 0 sans rien écrire. Il ne rend jamais `allow`.
5. **Given** un espace créé, **then** `notes/BLOCAGES.md` existe avec un en-tête et reste inscriptible (`Edit(./notes/**)`) ; la consigne demande de le tenir.
6. **Given** `CONSIGNE.md` rendu, **then** il interdit tout décodage autorégressif dans `run-script` (boucle qui réinjecte une sortie du modèle, échantillonnage) et renvoie à `run-batch --thinking on` et à `probe`.

## Familles de refus et voie permise

| Famille | Couche | Exemple de message visé |
|---|---|---|
| commande hors liste, Python nu, `cd`, variable devant | crochet | « commande-hors-liste : seules run-batch, session-status, run-script, probe ; Python par run-script » |
| lecture hors de l'espace | crochet, lanceur | « lecture-hors-espace : tout l'espace <ws> est lisible, rien au-delà » |
| écriture hors liste, nom protégé | crochet, lanceur | « ecriture-hors-liste : écris dans scripts/, notes/, EDITION.md ou work/*.safetensors » |
| web, sous-agent, skill, MCP | crochet | « web : aucun accès réseau ; note le besoin dans notes/BLOCAGES.md » |
| import, exec, processus (lanceur) | lanceur | « import torch interdit : numpy, safetensors.numpy, stdlib ; modèle par run-batch ou probe » |
| budget, gel, verrou, séance liée (harnais) | harnais | « budget épuisé : 25/25 ; status par session-status » |

## Fichiers et fonctions touchés (vérifiés au 6de87e6)

- `workspace.py` : `build_settings` (clé `hooks.PreToolUse`, deny `Read` d'`OPERATEUR.md`), `create_workspace` (crée `notes/BLOCAGES.md`, écrit `OPERATEUR.md` dehors), `render` (marqueurs), et un nouveau modèle `seance/OPERATEUR.md`.
- Nouveau `src/slm_weight_editing/_refusal_hook.py`, bibliothèque standard seule : il lit le JSON de l'appel et reprend les familles de `DENIED_*` et de `audit_transcript.check_tool_use`. Il n'importe pas l'audit : `-I` interdit le chemin du dépôt (question 2).
- `_sandbox.py` : le message de `deny()` porte la règle et la voie permise ; `check_read` prend l'espace entier comme racine de lecture.
- `script_runner.py` : `cfg["read_roots"]` passe à `[workspace]`, `read_files` devient inutile ; `write_roots` ne change pas.
- `harness.py` : messages de `HarnessError` avec la voie permise (environ 50 `raise`).
- `seance/CONSIGNE.md`, `seance/CLAUDE.md` : lecture libre, BLOCAGES, interdiction du décodage autorégressif.

## Choix de conception

- **Crochet** : (a) forme exec, `command` = `<dépôt>/.venv/Scripts/python.exe`, `args` = `["-I", "-B", "<dépôt>/src/slm_weight_editing/_refusal_hook.py"]` ; (b) `uv run slm-weight-editing explain-refusal` en forme shell ; (c) pas de crochet, explication dans la consigne seule. **Recommandé : (a).** Démarrage rapide, rien du dossier courant n'est importé, aucune dépendance à Git Bash. À mesurer : coût par appel d'outil, et transmission ou non du bloc `env` des réglages au crochet.
- **Lecture libre** : (a) racine de lecture = l'espace entier ; (b) liste élargie (`notes/`, `probes/`, `runs/`, `EDITION.md`). **Recommandé : (a)** (SPEC). `canon()` résout déjà les liens et les jonctions, donc un lien vers l'extérieur reste refusé (test à ajouter).
- **OPERATEUR.md** : voir question 1. Dans tous les cas, l'espace ne garde qu'une mention : « instructions d'Anaël hors de l'espace ».

## Tests

- À ajouter : critères 1 à 6 ; table des familles (paramétrée) ; crochet nourri par les 10 contournements et par les `SCENARIOS` de l'audit (refus avec motif attendu, jamais `allow`) ; une jonction dans l'espace vers le dépôt est refusée.
- À réaligner : `test_session_workspace.py::REASONS` (texte des refus du lanceur ; garder le préfixe `<événement> :` pour limiter la reprise), `test_settings_json`, `test_workspace_layout_and_session`, `test_instructions_are_rendered`, `test_cli_session_workspace` ; `test_audit_transcript.py::test_consigne_states_protocol`, `test_instructions_do_not_leak_reference`.

## Garanties à préserver

- Les 10 contournements, les 28 scripts interdits (aucun ne dépend d'une lecture dans l'espace : à revérifier un par un) et les 73 transcriptions de violation restent en échec.
- Aucune lecture hors de l'espace. Écriture identique à la v1. Aucun deny retiré, et le crochet ne fait qu'ajouter des refus.

## Dépendances

- Elle touche `workspace.py`, `CONSIGNE.md` et `script_runner.py`, comme 2.2, 2.3 et 2.4 : à placer après elles ou dans une autre vague. Il faudra réaligner les commandes de la consigne (`--thinking`, `probe`, délai de `run-script`).
- Vérifications d'Anaël avant la séance, en plus de celles de la story 5 : `/hooks` montre le crochet de l'espace et aucun autre qui réponde `allow` ; aucun mod installé. Essai à blanc : `cat CONSIGNE.md` est refusé avec le motif du crochet.

## Questions pour Anaël

1. Où mettre `OPERATEUR.md` : dans le dépôt sous `models/operateur/<id>.md` (ignoré par git, déjà hors de portée par les trois couches), à côté de l'espace, ou à un chemin `--operator-out` imposé ?
2. Crochet en forme exec avec le `python.exe` du `.venv` du dépôt et la bibliothèque standard seule, ou `uv run` (plus lent) ? Acceptes-tu qu'un crochet en échec laisse passer l'appel, les deny restant la porte ?
3. Tes crochets et tes mods personnels (`~/.claude/settings.json`, agent-cost-kit) tournent aussi dans l'espace, et un crochet `allow` ou un mod peut contourner `dontAsk` ou un deny. Vérification à la main avant la séance, ou lancement avec un `--settings` dédié ?
4. Lecture libre : les scripts peuvent-ils lire aussi `.claude/settings.json`, `workspace.json` (qui contient le chemin du dépôt), les copies `sessions/<id>/scripts/` et `freeze/` ? Ou faut-il en exclure certains ?
5. `notes/BLOCAGES.md` : est-il publié avec le gel (comme le journal public) ou reste-t-il privé ?
