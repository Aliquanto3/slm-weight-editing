---
id: SPEC-harnais-v2
companions:
  - incidents-s1.md
  - probe-captures.md
  - ../spec-slm-weight-editing/SPEC.md
  - ../spec-slm-weight-editing/story-3-harnais.md
  - ../spec-slm-weight-editing/story-4-gel-diff.md
  - ../spec-slm-weight-editing/story-5-protocole-seance.md
  - ../spec-slm-weight-editing/fix-report-embed.md
sources:
  - ../../../docs/CADRAGE.md
---

> **Contrat de référence.** Cette spec et les fichiers de `companions:` forment le contrat complet de ce qu'il faut construire, tester et valider. Les fiches des stories 3, 4 et 5 décrivent le harnais v1 que cette spec modifie ; ce qu'elle ne change pas reste en vigueur.

# Harnais v2 pour la séance s2

## Why

Il s'agit à la fois d'un problème à régler et d'une exigence de méthode. La séance s1 a corrigé la boucle de `</think>` (35,5 % au banc avec raisonnement), mais elle ne compte pas comme réplication du MIT à budget égal. En effet, `run-batch` ne générait que sans raisonnement. Pour observer le défaut, l'éditeur a donc écrit un forward numpy et généré hors budget. La séance a aussi buté sur des frictions du harnais : refus muets, lecture bridée, verrou orphelin, machine saturée, sortie tronquée et faux positifs de l'audit (détail dans `incidents-s1.md`). La séance s2 comptera pour le lot 1. Le harnais doit donc fermer la porte de la génération hors budget tout en donnant à l'éditeur les moyens d'observation légitimes qui lui manquaient.

## Capabilities

- **CAP-1** — probe : forward sans génération, sur budget séparé
  - **intent:** L'éditeur obtient, par une commande du harnais, les logits et les activations du modèle de `work/` sur des textes fixes (forward forcé, aucun jeton généré). Le résultat est écrit dans un fichier de l'espace que ses scripts peuvent lire.
  - **success:** Une probe sur un texte donné rend des captures identiques à un forward de référence sur les mêmes poids. Chaque appel décompte les jetons passés en forward d'un budget de probe fixé avant la séance (2 M pour s2), distinct du budget de lots, et inscrit une ligne au journal. Sont refusés sans calcul : un appel qui dépasserait le budget, plus de 64 textes, un texte de plus de 8 192 jetons, une sortie estimée à plus de 2 Go. Le budget de lots reste inchangé. Le format des captures est fixé dans `probe-captures.md`.
- **CAP-2** — mode raisonnement choisi à chaque lot
  - **intent:** L'éditeur lance `run-batch` avec ou sans raisonnement, pour mesurer le défaut comme pour contrôler les régressions.
  - **success:** `--thinking on` applique le gabarit avec `enable_thinking` et `--thinking off` l'applique sans. Le mode est inscrit dans la ligne `batch` du journal. Un lot sans `--thinking` est refusé, faute de mode explicite.
- **CAP-3** — décodage autorégressif interdit hors du harnais
  - **intent:** Aucune génération ne se fait hors de `run-batch` : la consigne l'interdit dans `run-script` et l'audit signale tout script qui pourrait en contenir.
  - **success:** La consigne de séance énonce l'interdiction. L'audit marque les copies de scripts qui bouclent en réinjectant une sortie du modèle dans son entrée, ou qui échantillonnent. Il marque aussi les suites d'appels probe dont chaque texte prolonge le précédent par un jeton. Il produit la liste de ces copies, qu'Anaël relit avant la mesure. Le forward numpy de s1, rejoué tel quel, est signalé.
- **CAP-4** — refus expliqués et blocages consignés
  - **intent:** Chaque refus nomme la règle enfreinte et la voie permise, qu'il vienne des réglages de l'espace, du lanceur ou du harnais. L'éditeur consigne ce qui le bloque dans `notes/BLOCAGES.md`.
  - **success:** Pour chaque famille de refus, un test vérifie que le message contient la règle et l'alternative autorisée. La consigne demande de tenir `notes/BLOCAGES.md`, qui est inscriptible par l'éditeur et lu par Anaël après la séance.
- **CAP-5** — lecture libre dans l'espace
  - **intent:** L'éditeur lit, liste et recherche librement tout fichier de l'espace de séance, par les outils de Claude Code comme par ses scripts. Les droits d'écriture restent ceux de la v1.
  - **success:** Read, Grep et Glob sur n'importe quel fichier de l'espace, y compris le journal de séance, sont acceptés et ne sont pas signalés par l'audit. Un script lit tout fichier de l'espace. Toute lecture hors de l'espace reste refusée par les trois couches.
- **CAP-6** — OPERATEUR.md hors de l'espace
  - **intent:** Les instructions destinées à Anaël (gel, audit, déverrouillage, mesure) sont générées hors de l'espace de séance, hors de portée de l'éditeur.
  - **success:** `session-workspace` écrit `OPERATEUR.md` hors du dossier de l'espace et n'en laisse aucune copie dedans. Une lecture de ce fichier par l'éditeur est refusée.
- **CAP-7** — verrou identifié et déverrouillage par Anaël
  - **intent:** Le verrou de séance indique quel processus le tient. Anaël peut retirer un verrou orphelin par une commande du harnais, sans manipulation de fichier.
  - **success:** `session.lock` contient le PID et l'heure de prise. Un refus pour verrou présent indique ce PID et précise s'il est vivant. `session-unlock` retire le verrou seulement si son processus est mort, et le journalise. Cette commande est interdite à l'éditeur par les réglages et par l'audit.
- **CAP-8** — exécution des scripts maîtrisée
  - **intent:** `run-script` borne les threads de calcul, applique un délai choisi pour chaque script sous un maximum et écrit la sortie complète dans un fichier de l'espace.
  - **success:** Le nombre de threads BLAS du script est égal au plafond configuré (4 par défaut), qu'un script ne peut pas relever. Le délai vaut 15 min par défaut ; un délai demandé au-dessus du maximum (60 min par défaut) est refusé. Les trois valeurs se fixent à la création de l'espace ; un script qui dépasse son délai est arrêté, et le journal note `timed_out`. La sortie entière est lisible dans un fichier de l'espace. Le journal garde un extrait, la taille totale et le sha256 de ce fichier.
- **CAP-9** — audit sans les faux positifs de s1
  - **intent:** L'audit ne signale plus les usages légitimes relevés en s1 et continue d'échouer sur toutes les violations réelles.
  - **success:** Sont acceptés `re.compile`, le décodage des octets du tokenizer et Grep sur le journal de l'espace. La transcription s1, rejouée, n'échoue que sur ses vraies violations : génération numpy et violations mineures. Les 73 transcriptions de violation et les 28 scripts interdits de la v1 restent en échec.
- **CAP-10** — n'éditer que la sortie : `lm_head` détaché
  - **intent:** Pendant la séance, l'éditeur peut séparer `lm_head` des embeddings liés par une commande du harnais, pour éditer la sortie d'un jeton sans toucher son embedding d'entrée.
  - **success:** Après détachement, `work/` contient `lm_head.weight`, égal octet pour octet à `embed_tokens`, et la config déclare `tie_word_embeddings: false`. Les sorties d'un lot rejoué avant toute édition sont identiques. L'opération est journalisée et ne se fait qu'une fois. Le gel, `apply-diff` et `report-gguf` traitent `lm_head.weight` comme un tenseur ajouté, reporté en `output.weight` Q8_0 dans le GGUF. Avant toute édition, `output.weight` est identique octet pour octet à `token_embd`, ce qui sert de témoin de fidélité. Le diff et `compare-edits` affichent « tenseur ajouté : `lm_head.weight` » et comptent les poids édités par rapport à l'`embed_tokens` de la base. Rejouer le diff redonne les sha256 publiés.

## Constraints

- Garanties inchangées par rapport à la v1. Pas de contamination : ni web, ni journaux du MIT, ni `heldout/`. Banc à l'aveugle : évaluation après le gel, par une commande à part. Budget compté et journalisé, y compris pour probe. Aucun gradient, aucune perte, aucun optimiseur.
- Aucune régression de sécurité. Les trois couches de la v1 sont conservées (réglages de l'espace, lanceur à crochet d'audit, audit de transcription). Les 10 contournements, les 28 scripts interdits et les 73 transcriptions de violation restent en échec. La lecture libre (CAP-5) ne s'étend pas au-delà de l'espace.
- Toute génération passe par `run-batch` : probe ne génère aucun jeton. C'est ce qui garde la séance comparable au MIT à budget de lots égal.
- probe tourne comme `run-batch` : `torch.inference_mode()`, poids rechargés depuis `work/`, sha256 de `work/` journalisés à chaque appel.
- Les commandes d'Anaël (`session-open`, `session-freeze`, `session-workspace`, `session-unlock`, `audit-session`) restent interdites à l'éditeur par les réglages et par l'audit.
- Le détachement de `lm_head` est une commande du harnais. Une écriture directe de `lm_head.weight` ou de `config.json` par un script ne vaut pas détachement.
- Plafond du diff inchangé : 10 % par tenseur, éditions creuses. Un `lm_head` ajouté se compare à l'`embed_tokens` de la base.

## Non-goals

- Bac à sable du système d'exploitation sous Windows natif : la limite de la v1 reste en place.
- Garantie d'exécution contre le décodage autorégressif dans `run-script`. Un forward numpy n'est que du calcul matriciel ; l'interdiction repose sur la consigne, l'audit et la relecture d'Anaël.
- Génération par probe, même comptée.
- Plafond de temps CPU cumulé pour `run-script` sur la séance.
- Lots 2 et 2 bis, ainsi que tout GPU loué.

## Success signal

- La séance s2 se déroule de bout en bout sur le harnais v2. L'éditeur observe le défaut avec raisonnement par `run-batch --thinking on` et `probe`, sans écrire de forward génératif. L'audit passe sans faux positif, et la liste de scripts à relire est vide ou relue par Anaël. s2 peut alors être présentée comme réplication du lot 1, avec deux compteurs publiés (lots et probe).

## Assumptions

- Le budget de probe se compte en jetons passés en forward. Anaël le fixe à la création de l'espace, comme le budget de lots.
- probe accepte tout texte : prompts train, sorties de lots, texte écrit par l'éditeur. Une génération reconstituée par des appels probe successifs coûte un nombre de jetons quadratique : elle est bornée par le budget et signalée par l'audit (CAP-3).
- Ollama 0.40.14 charge un `output.weight` séparé pour qwen3.5. Un essai préalable le vérifie avant la story `lm_head` : GGUF avec `output.weight` copié de `token_embd`, `ollama create` par Anaël, réponses identiques à l'officiel à température 0. En cas d'échec, CAP-10 sort de s2. L'écart de temps de réponse au banc (environ 270 Mo de plus) reste à vérifier.
- Le crochet du lanceur peut autoriser la lecture des `.npy` en `mmap_mode` sans ouvrir d'autre accès.
