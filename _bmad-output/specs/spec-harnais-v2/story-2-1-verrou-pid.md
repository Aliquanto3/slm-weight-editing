---
status: review
story: "2.1"
title: Verrou de séance avec PID et session-unlock (CAP-7)
---

# Story 2.1 : verrou avec PID et session-unlock

- `session.lock` contient `{"pid", "acquired_at"}` (JSON, heure UTC) ; retiré à la sortie du lot, du gel ou d'`update_state`.
- Refus d'un verrou présent : PID vivant (« un lot tourne déjà »), PID mort (« verrou orphelin », seul Anaël le retire par `session-unlock`), contenu illisible (refus explicite).
- Détection portable de processus vivant `harness.pid_alive` : `ctypes` OpenProcess/GetExitCodeProcess sous Windows, `os.kill(pid, 0)` ailleurs.
- `slm-weight-editing session-unlock --session S [--sessions-root R]` : retire le verrou si le processus est mort ou le contenu illisible, ajoute une ligne `unlock` (pid, lock_acquired_at, at) au journal ; refuse (code 2, rien retiré) si le processus est vivant ; sans verrou, message et code 0 (état voulu atteint).
- Interdit à l'éditeur : deny `*session-unlock*` du settings.json et règle d'audit `deverrouillage-seance`; absent de `ALLOWED_SUBCOMMANDS`.

## Reports

- Fenêtre minime entre création et écriture du verrou : un concurrent y lit un fichier vide (refus « illisible », sans danger) ; `session-unlock` ne retire un verrou illisible qu'après 60 s d'âge.
- Un PID réutilisé par un autre processus après la mort du porteur passe pour vivant (refus prudent).

## Revue

### Passe 1 (2026-11-05, SHA revu 6d10a23)

- Important, medium, second avis 1 : la sortie d'un porteur retirait le verrou repris par un autre. Corrigé : `_session_lock` ne retire que le verrou dont le contenu (PID, heure) est le sien.
- Important, medium, second avis 2 : `session-unlock` retirait un verrou illisible tout juste créé (porteur vivant entre `O_EXCL` et l'écriture). Corrigé : refus tant que le verrou illisible a moins de 60 s.
- Important, medium, second avis 3 : ligne `unlock` journalisée avant le retrait. Corrigé : retrait puis journal ; test d'un `unlink` en échec.
- Important, medium, second avis 4 : écriture ratée du verrou laissant un fichier vide bloquant ; message « à la main ». Corrigé : retrait du verrou si l'écriture échoue ; message renvoyant à `session-unlock`.
- Nit, low, second avis 5 : PID hors bornes. Corrigé : `read_lock` et `pid_alive` refusent un PID hors de 1..2^31-1.
- Nit, low, second avis 6 : `session-unlock` sans `bind_workspace`. Rejeté : même traitement que `session-freeze` (commande d'Anaël lancée depuis le dépôt, racine par défaut).
- Nit, low, second avis 7 : message de `session-freeze` toujours « verrou orphelin ». Corrigé : le refus réel du verrou (vivant, mort ou illisible) est repris.
- Nit, low, second avis 8 : deny littéral, `getattr` composé non pris. Rejeté : alignement voulu sur la v1 (`open_session`, `freeze_session` traités de même : nom de CLI en deny, nom de fonction à l'audit).
- Nit, low, second avis 9 : tests manquants. Corrigé : branche POSIX de `pid_alive`, écriture ratée, ordre journal/retrait, sortie après reprise, PID hors bornes.
- Nit, low, revue : la CLI affichait « PID None mort » pour un verrou illisible. Corrigé.
- Complexité : 0 constat.
