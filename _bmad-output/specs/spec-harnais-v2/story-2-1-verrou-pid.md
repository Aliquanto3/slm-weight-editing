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

- Fenêtre minime entre création et écriture du verrou : un concurrent y lit un fichier vide (refus « illisible », sans danger).
- Un PID réutilisé par un autre processus après la mort du porteur passe pour vivant (refus prudent).
