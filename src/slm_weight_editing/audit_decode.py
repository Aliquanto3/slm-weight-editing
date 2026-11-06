"""Signalement du décodage autorégressif possible (story 2.6, CAP-9) : liste « à relire ».

Analyse statique, sans garantie (non-but de la spec) : elle n'échoue jamais l'audit, elle dresse la
liste des copies de scripts et des suites d'appels `probe` qu'Anaël relit avant la mesure.

- Copies de scripts : analyse `ast`. Une boucle qui choisit un indice (`argmax`, `choice`...),
  l'ajoute à une séquence puis repasse cette séquence à un appel du même corps est signalée ;
  tout tirage aléatoire (`choice`, `multinomial`...) l'est aussi. Un forward forcé (séquence fixe,
  `argmax` seulement pour mesurer) n'est pas signalé. Copie illisible : listée « illisible ».
- Suites de probes : au moins `MIN_CALLS` appels successifs dont chacun prolonge d'un à
  `MAX_ADDED` jetons un texte de l'appel précédent (comparaison des `token_ids` des manifests).
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

RULE = "decodage-autoregressif"
MIN_CALLS = 3  # appels consécutifs d'une suite signalée
MAX_ADDED = 4  # jetons ajoutés au plus à chaque appel

# Fonctions qui choisissent un indice de jeton ; celles qui tirent au hasard sont signalées seules.
_PICKERS = {
    "argmax",
    "argsort",
    "argpartition",
    "topk",
    "top_k",
    "searchsorted",
    "choice",
    "choices",
    "multinomial",
    "categorical",
}
_SAMPLERS = {"choice", "choices", "multinomial", "categorical"}
_NOT_SEQUENCE_USE = {"len", "print", "range", "enumerate", "isinstance", "str", "repr"}
_APPEND_METHODS = {"append", "extend", "insert"}


@dataclass(frozen=True)
class Candidate:
    """Copie de script ou suite de probes à relire."""

    source: str
    line: int
    kind: str  # « script », « illisible » ou « probe »
    detail: str

    def render(self) -> str:
        place = f"{self.source}:{self.line}" if self.line else self.source
        return f"{place} · {RULE} · {self.kind} · {self.detail}"

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "line": self.line,
            "rule": RULE,
            "kind": self.kind,
            "detail": self.detail,
        }


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _picks(node: ast.AST) -> str | None:
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and _call_name(n) in _PICKERS:
            return _call_name(n)
    return None


def _target_name(target: ast.AST) -> str | None:
    while isinstance(target, ast.Subscript | ast.Attribute):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def _loop_feedback(loop: ast.For | ast.While) -> str | None:
    """Motif d'une boucle dont le jeton choisi revient à l'entrée du calcul suivant."""
    statements = [n for stmt in loop.body for n in ast.walk(stmt)]
    picked: dict[str, str] = {}
    changed = True
    while changed:  # l'indice choisi, puis ce qui s'en déduit (`nxt = int(tok)`)
        changed = False
        for n in statements:
            if not isinstance(n, ast.Assign):
                continue
            how = _picks(n.value) or next(
                (picked[name] for name in _names(n.value) & picked.keys()), None
            )
            for target in n.targets:
                name = _target_name(target)
                if how and name and name not in picked and name not in _names(n.value):
                    picked[name] = how
                    changed = True
    if not picked:
        return None
    for stmt in statements:
        sequence, pick, node = None, None, None
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if _call_name(call) in _APPEND_METHODS and isinstance(call.func, ast.Attribute):
                sequence, node = _target_name(call.func.value), call
                used = set().union(*[_names(a) for a in call.args]) if call.args else set()
                pick = next(iter(used & picked.keys()), None)
        elif isinstance(stmt, ast.Assign | ast.AugAssign):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            names = _names(stmt.value)
            pick = next(iter(names & picked.keys()), None)
            for target in targets:
                name = _target_name(target)
                is_update = (
                    isinstance(stmt, ast.AugAssign)
                    or isinstance(target, ast.Subscript)
                    or name in names
                )
                if name and name not in picked and is_update:
                    sequence, node = name, stmt
        if not (sequence and pick and node):
            continue
        inside = {id(n) for n in ast.walk(node)}
        for n in statements:
            if not isinstance(n, ast.Call) or id(n) in inside:
                continue
            if _call_name(n) in _NOT_SEQUENCE_USE:
                continue
            if isinstance(n.func, ast.Attribute) and _target_name(n.func.value) == sequence:
                continue
            args = [*n.args, *(k.value for k in n.keywords)]
            if any(sequence in _names(a) for a in args):
                return (
                    f"« {picked[pick]} » ajouté à « {sequence} » (ligne {node.lineno}), "
                    f"repassé à « {_call_name(n) or 'un appel'} » dans la même boucle"
                )
    return None


def scan_script(source: str) -> list[tuple[int, str, str]]:
    """Signalements (ligne, genre, motif) d'une copie de script ; genre `script` ou `illisible`."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return [
            (getattr(exc, "lineno", None) or 0, "illisible", f"copie non lisible en AST : {exc}")
        ]
    found: dict[tuple[int, str], str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in _SAMPLERS:
            found[(node.lineno, "tirage")] = f"tirage aléatoire « {_call_name(node)} »"
        if isinstance(node, ast.For | ast.While):
            motif = _loop_feedback(node)
            if motif:
                found[(node.lineno, "boucle")] = f"boucle à rétroaction : {motif}"
    return [(line, "script", motif) for (line, _), motif in sorted(found.items())]


def _texts_of(manifest: Path) -> list[list[int]] | None:
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        texts = [t["token_ids"] for t in data["texts"]]
    except OSError, ValueError, KeyError, TypeError:
        return None
    if all(isinstance(t, list) and all(isinstance(i, int) for i in t) for t in texts):
        return texts
    return None


def _extends(current: list[list[int]], previous: list[list[int]]) -> bool:
    return any(
        1 <= len(c) - len(p) <= MAX_ADDED and c[: len(p)] == p
        for c in current
        for p in previous
        if p
    )


def scan_probe_suites(probes: list[tuple[int, list[list[int]]]]) -> list[tuple[list[int], int]]:
    """Suites (numéros d'appels, longueur) d'appels successifs qui prolongent le précédent.

    `probes` : (numéro d'appel, textes en `token_ids`) dans l'ordre du journal. Une suite compte
    au moins `MIN_CALLS` appels ; les textes préfixes l'un de l'autre dans un seul appel ne font
    pas une suite (ils sont fixés avant le calcul).
    """
    suites: list[tuple[list[int], int]] = []
    run: list[int] = []
    for i, (number, texts) in enumerate(probes):
        if i and _extends(texts, probes[i - 1][1]):
            run = run or [probes[i - 1][0]]
            run.append(number)
            continue
        if len(run) >= MIN_CALLS:
            suites.append((run, len(run)))
        run = []
    if len(run) >= MIN_CALLS:
        suites.append((run, len(run)))
    return suites


def review_candidates(session: str | Path, workspace: str | Path | None = None) -> list[Candidate]:
    """Copies de scripts et suites de probes d'une séance à relire (jamais une violation)."""
    session = Path(session)
    workspace = Path(workspace) if workspace is not None else session.parent.parent
    journal = session / "journal.jsonl"
    found: list[Candidate] = []
    try:
        lines = journal.read_text(encoding="utf-8").splitlines()
    except OSError:
        return found
    probes: list[tuple[int, list[list[int]]]] = []
    for number, raw in enumerate(lines, start=1):
        try:
            entry = json.loads(raw) if raw.strip() else None
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("event") == "script":
            copy = session / str(entry.get("copy", ""))
            try:
                data = copy.read_bytes() if copy.is_file() and copy.parent != session else None
            except OSError:
                data = None
            if data is None:
                continue  # copie absente : déjà une violation `script-copie`
            name = str(entry.get("script"))
            for line, kind, motif in scan_script(data.decode("utf-8", "replace")):
                found.append(Candidate(f"{name} (journal ligne {number})", line, kind, motif))
        elif entry.get("event") == "probe" and isinstance(entry.get("probe"), int):
            texts = _texts_of(workspace / "probes" / str(entry["probe"]) / "manifest.json")
            if texts is not None:
                probes.append((entry["probe"], texts))
    for numbers, length in scan_probe_suites(probes):
        detail = (
            f"suite de {length} appels probe successifs (n° {', '.join(map(str, numbers))}) : "
            f"chaque appel prolonge de 1 à {MAX_ADDED} jetons un texte du précédent"
        )
        found.append(Candidate(journal.as_posix(), 0, "probe", detail))
    return found
