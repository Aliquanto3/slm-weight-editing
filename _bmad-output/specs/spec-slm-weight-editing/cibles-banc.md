# Cibles relevées par le banc

Relevé du 2026-11-04, source : `benchmarks/quality/<campagne>/summary.json` de `slm-quality-bench` (champ `truncated` : réponse arrêtée au plafond, `done_reason == "length"`). Réglages : température 0, plafond de 4 096 tokens (2 048 en RAG). Modèles ≤ 2,5 Md de paramètres :

| Modèle | Md | Tronqué think_off | Tronqué think_on | Score off / on | Lecture |
|---|---|---|---|---|---|
| `qwen3.5:0.8b` | 0,87 | 7,1 % | **72,7 %** | 36,9 / 15,6 | cible n° 1 : le raisonnement fait perdre 21 points |
| `qwen3.5:2b` | 2,3 | 5,7 % | **53,0 %** | 50,6 / 35,4 | cible n° 2, même famille, même défaut |
| `lfm3:1.7b-q4_K_M` | 1,73 | 12,6 % | 33,4 % | 51,1 / 52,6 | candidat, autre famille |
| `lfm2.5-thinking:1.2b` | 1,2 | – | 9,0 % | – / 47,3 | boucle peu |
| `granite4:350m` | 0,35 | 4,0 % | – | 25,1 / – | sans mode raisonnement, boucle peu : score faible par manque de capacité, proche de l'échec du MIT sur Gemma 3 270M |

Au-delà de 2,5 Md, `qwen3.5:4b` (40,4 % en think_on) et `granite4.2:3b` (36,0 %) gardent le même défaut.

Défauts candidats autres que les boucles : seulement ceux qui ont une signature mesurable sur des données d'entraînement (langue, format, fuite du raisonnement), après instrumentation par réponse dans le banc.
