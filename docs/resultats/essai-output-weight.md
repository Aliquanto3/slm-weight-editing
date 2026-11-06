# Résultat : essai `output.weight` séparé pour qwen3.5:0.8b

Protocole : `docs/essai-output-weight.md`. Essai lancé le 2026-11-06 vers 05:50 par l'agent orchestrateur, sur autorisation explicite d'Anaël (« Ollama est disponible actuellement, tu peux t'en servir »), GPU libre au départ (RTX 3050, 0 Mio occupé).

## Conditions

- Ollama 0.40.14, modèle officiel `qwen3.5:0.8b` (f3817196d142).
- `gguf-add-output --model qwen3.5:0.8b` : `token_embd` en **Q8_0**, `output.weight` [1024, 248320] Q8_0 par **copie d'octets**, +270 172 192 octets.
- Modelfiles essai et témoin en LF, identiques hors `FROM` (vérifié par `diff`).
- `ollama create qwen35-temoin` et `ollama create qwen35-output-test` : succès, chargement sans erreur.
- 8 prompts publics du protocole, température 0, graine 42, `num_predict` 64, une génération de chauffe par modèle. Comparaison sur `response` et `thinking`.

## Résultats

| Contrôle | Résultat par prompt (1 à 8) |
|---|---|
| Officiel relancé deux fois (déterminisme) | identique ×8 |
| Témoin = officiel (Modelfile fidèle) | identique ×8 |
| **Essai = témoin** | **identique ×8** |

| Modèle | Taille (`ollama list`) | Médiane `total_duration` | `eval_count` |
|---|---|---|---|
| qwen3.5:0.8b | 1,0 Go | 606 ms (611 ms au second passage) | 64 ×8 |
| qwen35-temoin | 1,0 Go | 616 ms | 64 ×8 |
| qwen35-output-test | 1,3 Go | 616 ms | 64 ×8 |

Taille du GGUF : 1 036 034 688 octets pour l'officiel, 1 306 206 880 pour l'essai, soit **+26,1 %**. Temps de réponse médian : **+0 %** par rapport au témoin.

Aucun message d'erreur.

## Conclusion

**CAP-10 maintenu** selon le critère du protocole : Ollama charge le `output.weight` séparé et répond mot pour mot comme le témoin.

Réserve pour Anaël : la décision 5 de la fiche 2.8, prise par l'orchestrateur le 2026-11-06, faisait sortir CAP-10 au-delà de +25 % de taille. Les +26,1 % sont structurels : c'est le coût d'une matrice de sortie Q8_0 non liée, connu dès la story 2.7 (+270 Mo). Ce seuil était donc mal choisi. L'orchestrateur propose de ne garder que le critère de temps de réponse (+20 %), qui n'est pas atteint. **La décision sur la taille revient à Anaël.**
