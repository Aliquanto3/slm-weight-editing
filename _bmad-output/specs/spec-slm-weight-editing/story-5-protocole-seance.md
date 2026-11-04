---
id: STORY-5-protocole-seance
status: review
parent: SPEC.md
capabilities: [CAP-6]
---

# Story 5 : protocole et garde-fous de la séance d'édition Claude

Spec d'origine : entrée `id: "5"` de `stories.yaml`, exigences transmises par `story-3-harnais.md` (section « Exigence transmise à la story 5 »).

## Livré

- Espace de séance `session/` (modèle versionné, copié par Anaël, `<ID_SEANCE>` et `<BUDGET>` remplacés à la main) : `.claude/settings.json`, `CONSIGNE.md` (prompt de départ), `CLAUDE.md` (rappel court, relu après compaction). Lancé depuis `session/`, le harnais s'appelle avec `--sessions-root ../models/sessions`.
- `src/slm_weight_editing/audit_transcript.py`, sous-commande `audit-session` (`add_parser(sub)` + `set_defaults(func=run)`). Codes : 0 sans violation, 1 avec violations (`fichier:ligne · règle · outil · détail`), 2 entrée illisible.
- Consigne : cible (Qwen3.5-0.8B, boucles tronquées au plafond de 4096 jetons), budget, harnais seule voie d'exécution, édition des safetensors de `work/` par numpy et `safetensors_file`, interdits, plafond ferme de 10 % d'éléments changés par tenseur (`MAX_CHANGED_SHARE` de `harness.py`, décision d'Anaël du 2026-11-04 ; un diff au-delà est refusé au gel), gel par Anaël seulement. Aucune mention de l'édition de référence, de ses tenseurs, couches ou résultats.

## Critères

- Transcription propre (lots et statut par le harnais, lecture du journal et de `work/`, script numpy qui réécrit `work/`) : code 0.
- Une transcription synthétique par violation (73 cas paramétrés), chacune nommée avec sa ligne : web (WebFetch, WebSearch, `mcp__*`, `server_tool_use`, curl, Invoke-WebRequest, urllib), lectures interdites (relatif, `..`, casse, guillemets, accent grave PowerShell, concaténation `'held' + 'out'`, nom court 8.3, motif `../h*out`, variable `${a}out`, Grep sur `..` ou `C:\`, `cd ..`), modèle hors harnais (transformers, torch, `from_pretrained`, `generate`, `forward`, llama.cpp, ollama, pytest), API interne hors liste blanche, sous-commandes hors liste (`apply-diff`, `--help`, `compare-edits`), ouverture (`session-open`, `open_session`) et gel (`session-freeze`, `freeze_session`) par l'agent, écriture ou suppression de `state.json`, `journal.jsonl`, `registry.jsonl`, `session.lock`, `freeze/` ou d'un dossier de séance (Bash, PowerShell, Write, Edit, code Python), copie de `work/` entier, modification de `src/` ou `.claude/`, git, installation, obfuscation (base64, `-EncodedCommand`, subprocess), sous-agent, outil hors liste.
- `--session-dir` avec `--reference` (JSON) ou `--reference-from` (checkpoint de la révision publiée, sha256 par `tensor_hashes`) : tenseurs différents, absents ou en trop et révision différente refusés (copie de `work/` ou base éditée).
- `settings.json` valide et règles `deny` attendues ; `CONSIGNE.md` et `CLAUDE.md` sans RikiyaT, lm_understand, down_proj, couches 15 ou 19, ni sha256 de `reference-mit.md`.

## Choix

- Liste blanche : `uv run slm-weight-editing run-batch|session-status`, `uv run python <script>`, import `slm_weight_editing.safetensors_file` seulement ; outils Read, Grep, Glob, LSP, Bash, PowerShell, Write, Edit, MultiEdit, NotebookEdit, TodoWrite, AskUserQuestion, ToolSearch, plan. Tout autre outil échoue.
- Seuls les blocs `tool_use` des messages assistant sont audités (lignes `assistant` et doublons `api-request-blob`, dédoublonnés par `id`) ; le texte d'Anaël, la réflexion et les `tool_result` ne le sont pas. Les transcriptions de sous-agents (`<id>/subagents/*.jsonl`) sont ajoutées d'office.
- Le journal reste lisible (sorties des lots) ; seule son écriture est refusée.
- Le dépôt entier n'est pas interdit en lecture (le harnais et `work/` y vivent, et une règle `deny` l'emporte sur toute `allow`) : les fichiers qui décrivent l'édition de référence (`_bmad-output/`, `docs/`, `tests/`, `README.md`, `AGENTS.md`, `__init__.py`, `report_gguf.py`, `.git/`) sont refusés un par un.

## Règles `deny` et incertitude

Doc officielle non disponible en local ; syntaxe retenue de mémoire : `Read(//chemin)` absolu (sous Windows `C:\` devient `/c/`), `~/` pour le dossier personnel, motifs façon gitignore, `Edit(...)` pour toutes les écritures par outil, `Bash(*mot*)` avec joker en toute position, `mcp__*`. À vérifier par Anaël avec `/permissions` avant la séance : que `//**/heldout/**` couvre un chemin absolu quelconque, que `Read` couvre Grep et Glob, que les règles `PowerShell(...)`, `claudeMdExcludes` et `disableBypassPermissionsMode` sont reconnues. Les règles Bash ne bloquent pas un chemin construit à l'exécution : seul l'audit fait foi.

## Limites connues de l'audit

- Audit textuel : un chemin ou un import construit à l'exécution (variable Python, `getattr`, fichier lu puis exécuté, liens créés hors transcription) n'est pas vu ; les formes courantes d'obfuscation sont refusées en bloc. Un `cd` dans un dossier de séance suivi d'un nom nu est vu seulement pour les noms protégés et les motifs.
- Faux positifs possibles, à relire à la main : mot « threshold out » (forme compacte `heldout`), comparaison `>` dans un `python -c` qui cite le journal, chemin de `work/` construit par `Path(...) / "work"` avec écriture (la consigne demande un littéral complet), README de `work/`.
- `CLAUDE.md` et `AGENTS.md` du dépôt, chargés automatiquement par Claude Code comme mémoire des dossiers parents, citent l'existence de l'étude (sans tenseurs ni résultats) ; `claudeMdExcludes` doit les écarter, à vérifier par `/memory`.
- Le contrôle de la base demande une référence fournie par Anaël (checkpoint de la révision publiée) ; sans `--session-dir`, la copie de `work/` n'est vue que par l'ouverture de séance par l'agent.
- La transcription elle-même pourrait être réécrite hors de la séance : l'audit suppose qu'Anaël la copie juste après la séance.

## Reports

- Génération de l'espace par une commande (chemins absolus dans `settings.json`) si les motifs `//**/` s'avèrent ignorés sous Windows.
