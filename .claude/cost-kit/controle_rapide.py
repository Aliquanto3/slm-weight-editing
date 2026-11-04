"""Contrôle rapide de fin de story : lint, format, typage, règles d'architecture.

Modèle à copier dans `.claude/cost-kit/controle_rapide.py` et à adapter : remplir COMMANDES avec
les contrôles du projet qui tiennent en une dizaine de secondes. Jamais la suite de tests complète.

Comportement :
- n'affiche que les erreurs (les sorties vertes sont tues, pour ne pas gonfler le contexte) ;
- code de retour 2 s'il y a au moins une erreur, 0 sinon ;
- `--max-blocages N` (défaut 2) : au-delà de N échecs consécutifs enregistrés dans
  `.claude/cost-kit/.blocages`, sort en 0 pour ne pas faire boucler l'agent.

Usage : python .claude/cost-kit/controle_rapide.py [--max-blocages 2]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# À adapter au projet. Exemples :
#   ["uv", "run", "ruff", "check", "."],
#   ["uv", "run", "ruff", "format", "--check", "."],
#   ["uv", "run", "mypy"],
#   ["npm", "run", "lint"],
COMMANDES: list[list[str]] = [
    ["uv", "run", "ruff", "check", "."],
    ["uv", "run", "ruff", "format", "--check", "."],
]
MAX_LIGNES_PAR_COMMANDE = 40


def main() -> int:
    analyseur = argparse.ArgumentParser(description=__doc__)
    analyseur.add_argument("--max-blocages", type=int, default=2)
    options = analyseur.parse_args()
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    compteur = Path(".claude/cost-kit/.blocages")
    echecs = []
    for commande in COMMANDES:
        try:
            resultat = subprocess.run(  # noqa: PLW1510
                commande,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=120,
            )
        except Exception as erreur:  # noqa: BLE001 # toute erreur d'exécution compte comme un échec
            echecs.append((commande, f"{type(erreur).__name__} : {erreur}"))
            continue
        if resultat.returncode != 0:
            sortie = (resultat.stdout + resultat.stderr).strip().splitlines()
            echecs.append((commande, "\n".join(sortie[:MAX_LIGNES_PAR_COMMANDE])))
    if not echecs:
        if compteur.exists():
            compteur.unlink()
        return 0
    blocages = int(compteur.read_text() or 0) + 1 if compteur.exists() else 1
    compteur.parent.mkdir(parents=True, exist_ok=True)
    compteur.write_text(str(blocages))
    for commande, sortie in echecs:
        print(f"[contrôle rapide] {' '.join(commande)} a échoué :\n{sortie}\n", file=sys.stderr)
    if blocages > options.max_blocages:
        print(f"[contrôle rapide] {blocages} échecs consécutifs : on laisse passer, à signaler dans le rapport.", file=sys.stderr)
        compteur.unlink()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
