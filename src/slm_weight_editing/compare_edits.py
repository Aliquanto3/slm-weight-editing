"""Comparaison chiffrée de deux éditions d'un même modèle (SPEC CAP-7).

Axes : tenseurs touchés (communs, propres à chacun, nombre de poids), cosinus des deltas aplatis
et des directions principales (premiers vecteurs singuliers du delta), puis, si des `summary.json`
de slm-quality-bench (`qb.report/1`) sont fournis, écart de score et de taux tronqué par piste et
par catégorie (usage), avec intervalles de confiance.

Méthode des intervalles : taux tronqué = Wilson (95 %) ; score = intervalle `ci_low`/`ci_high`
repris tel quel du résumé du banc (calculé par le banc, non recalculé ici).
Écart entre deux modèles traités comme
indépendants : méthode de Newcombe (MOVER), `d - sqrt((p1 - l1)^2 + (u2 - p2)^2)` pour la borne
basse et `d + sqrt((u1 - p1)^2 + (p2 - l2)^2)` pour la haute. Les résumés de deux campagnes n'ont
pas d'appariement : l'intervalle est donc conservateur.

Le rapport ne lit que des champs numériques nommés du résumé : jamais de question, de réponse
ni de chemin.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from slm_weight_editing.safetensors_file import SafeTensor, SafetensorsError, open_checkpoint

DEFAULT_OUT = "models/comparaison-editions/rapport.md"
Z95 = 1.959963984540054
_NUMERIC = {"F32": "<f4", "F64": "<f8", "F16": "<f2"}


class CompareError(ValueError):
    """Entrées incompatibles ou illisibles."""


def bytes_to_f64(data: bytes, dtype: str, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Octets bruts (BF16, F16, F32, F64) en float64, à la forme donnée."""
    if dtype == "BF16":
        u32 = np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16
        arr = u32.view(np.float32)
    elif dtype in _NUMERIC:
        arr = np.frombuffer(data, dtype=_NUMERIC[dtype])
    else:
        raise CompareError(f"{name} : dtype {dtype} non géré")
    return arr.astype(np.float64).reshape(shape)


def load_f64(st: SafeTensor) -> np.ndarray:
    """Tenseur safetensors (BF16, F16, F32, F64) en float64, à sa forme."""
    return bytes_to_f64(st.read_bytes(), st.dtype, st.shape, st.name)


def _check_same(original: dict[str, SafeTensor], edit: dict[str, SafeTensor], label: str) -> None:
    missing = sorted(set(original) - set(edit))
    extra = sorted(set(edit) - set(original))
    if missing or extra:
        raise CompareError(
            f"{label} : noms de tenseurs différents de l'original "
            f"(manquants : {missing[:3]}, en trop : {extra[:3]})"
        )
    for name, ref in original.items():
        got = edit[name]
        if got.shape != ref.shape:
            raise CompareError(f"{label} : {name} a la forme {got.shape}, original {ref.shape}")
        if got.dtype != ref.dtype:
            raise CompareError(f"{label} : {name} a le dtype {got.dtype}, original {ref.dtype}")


def deltas_from_checkpoints(
    original: dict[str, SafeTensor], edit: dict[str, SafeTensor], label: str
) -> dict[str, np.ndarray]:
    """Deltas (édité - original, float64) des seuls tenseurs dont au moins un poids diffère."""
    _check_same(original, edit, label)
    out = {}
    for name, ref in original.items():
        if ref.read_bytes() == edit[name].read_bytes():
            continue
        delta = load_f64(edit[name]) - load_f64(ref)
        if np.any(delta != 0):
            out[name] = delta
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def principal_directions(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Premiers vecteurs singuliers (gauche, droit) du delta, ou None si de rang 0.

    Réservé aux tenseurs d'au moins 2 dimensions (voir `compare_deltas` pour les autres).
    """
    if delta.ndim < 2:
        raise CompareError("directions principales : tenseur d'au moins 2 dimensions attendu")
    if not np.any(delta):
        return None
    mat = delta.reshape(delta.shape[0], -1) if delta.ndim != 2 else delta
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    if s.size == 0 or s[0] == 0:
        return None
    return u[:, 0], vt[0, :]


def compare_deltas(
    deltas_a: dict[str, np.ndarray], deltas_b: dict[str, np.ndarray], total_weights: int
) -> dict:
    """Axe « poids » à partir de deux dicts nom -> delta (point d'entrée pour la story 4)."""

    def side(deltas: dict[str, np.ndarray]) -> dict:
        per = {n: int(np.count_nonzero(d)) for n, d in deltas.items()}
        total = sum(per.values())
        return {
            "tensors": per,
            "modified_weights": total,
            "model_share": total / total_weights if total_weights else None,
        }

    a, b = side(deltas_a), side(deltas_b)
    shared = []
    for name in sorted(set(deltas_a) & set(deltas_b)):
        da, db = deltas_a[name], deltas_b[name]
        if da.shape != db.shape:
            raise CompareError(f"{name} : formes de delta différentes {da.shape} / {db.shape}")
        entry: dict = {
            "name": name,
            "modified_a": a["tensors"][name],
            "modified_b": b["tensors"][name],
            "flat_cosine": cosine(da, db),
            "left_cosine": None,
            "right_cosine": None,
            "note": None,
        }
        if da.ndim < 2:
            # Biais, normes, scalaires : pas de direction principale (le droit vaudrait 1.0).
            entry["note"] = "tenseur 1-D : directions principales n/a"
            shared.append(entry)
            continue
        if not (np.all(np.isfinite(da)) and np.all(np.isfinite(db))):
            entry["note"] = "delta non fini (NaN ou inf) : directions non définies"
            shared.append(entry)
            continue
        pa, pb = principal_directions(da), principal_directions(db)
        if pa is None or pb is None:
            who = "A et B" if pa is None and pb is None else ("A" if pa is None else "B")
            entry["note"] = f"delta nul ou de rang 0 pour {who} : directions non définies"
        else:
            # Valeur absolue : le signe d'un vecteur singulier est arbitraire.
            entry["left_cosine"] = abs(float(np.dot(pa[0], pb[0])))
            entry["right_cosine"] = abs(float(np.dot(pa[1], pb[1])))
        shared.append(entry)
    return {
        "total_weights": total_weights,
        "a": a,
        "b": b,
        "common": shared,
        "only_a": sorted(set(deltas_a) - set(deltas_b)),
        "only_b": sorted(set(deltas_b) - set(deltas_a)),
    }


def _side_deltas(ref, edit, diff, label: str) -> dict[str, np.ndarray]:
    """Deltas d'un côté : checkpoint édité, ou diff de séance rejoué en mémoire."""
    if (edit is None) == (diff is None):
        raise CompareError(f"{label} : un checkpoint édité ou un diff, pas les deux ni aucun")
    if diff is None:
        return deltas_from_checkpoints(ref, open_checkpoint(edit), label)
    from slm_weight_editing.freeze import DiffError, diff_deltas

    try:
        return diff_deltas(diff, ref)
    except DiffError as exc:
        raise CompareError(f"{label} : {exc}") from exc


def _added_tensors(diff) -> list[str]:
    """Tenseurs ajoutés par un diff de séance (lm_head détaché), absents de l'original."""
    if diff is None:
        return []
    from slm_weight_editing.freeze import DiffError, load_diff

    try:
        meta, _ = load_diff(diff)
    except DiffError:
        return []  # l'erreur est levée par `_side_deltas`, avec son libellé
    return sorted(e["name"] for e in meta["tensors"] if e.get("added"))


def compare_checkpoints(original, edit_a, edit_b, diff_a=None, diff_b=None) -> dict:
    ref = open_checkpoint(original)
    total = sum(math.prod(t.shape) for t in ref.values())
    da = _side_deltas(ref, edit_a, diff_a, "édition A")
    db = _side_deltas(ref, edit_b, diff_b, "édition B")
    result = compare_deltas(da, db, total)
    for key, diff in (("a", diff_a), ("b", diff_b)):
        added = _added_tensors(diff)
        # Le delta d'un tenseur ajouté est mesuré contre l'embed_tokens de la base.
        result[key]["added"] = added
        result[key]["lm_head_detached"] = bool(added)
    return result


# --- Banc -------------------------------------------------------------------------------------


def wilson(successes: float, n: int, z: float = Z95) -> tuple[float, float, float]:
    """(p, bas, haut) de l'intervalle de Wilson d'une proportion."""
    if n <= 0:
        raise CompareError("Wilson : effectif nul")
    p = successes / n
    z2 = z * z
    den = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def newcombe_diff(
    one: tuple[float, float, float], two: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Écart `one - two` et son intervalle (Newcombe/MOVER) ; entrées (p, bas, haut)."""
    p1, l1, u1 = one
    p2, l2, u2 = two
    d = p1 - p2
    low = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    high = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, low, high


def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _int(x) -> int:
    return x if isinstance(x, int) and not isinstance(x, bool) else 0


def read_summary(path: str | Path, tag: str | None) -> dict[tuple[str, str], dict]:
    """Chiffres d'un modèle par (piste, catégorie) d'un `summary.json` `qb.report/1`.

    Structure lue (slm_quality_bench/report/campaign.py, schéma REPORT_V1) :
    `tracks[].track`, `tracks[].scopes[].usage` (nul = périmètre global) et, dans
    `scopes[].models[]`, les champs numériques `mean`, `ci_low`, `ci_high`, `graded`, `truncated`.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CompareError(f"summary.json illisible ({type(exc).__name__})") from exc
    if not isinstance(data, dict) or data.get("schema") != "qb.report/1":
        raise CompareError("summary.json : schéma qb.report/1 attendu")
    out: dict[tuple[str, str], dict] = {}
    seen: set[str] = set()
    for track in data.get("tracks", []):
        for scope in track.get("scopes", []):
            entries = scope.get("models", [])
            seen.update(e["model"] for e in entries if isinstance(e.get("model"), str))
            if tag is None:
                if len(entries) != 1:
                    raise CompareError("summary.json : plusieurs modèles, préciser --tag-a/--tag-b")
                entry = entries[0]
            else:
                found = [e for e in entries if e.get("model") == tag]
                if not found:
                    continue
                entry = found[0]
            usage = scope.get("usage")
            out[(str(track.get("track")), "global" if usage is None else str(usage))] = {
                "mean": _num(entry.get("mean")),
                "ci_low": _num(entry.get("ci_low")),
                "ci_high": _num(entry.get("ci_high")),
                "graded": _int(entry.get("graded")),
                "truncated": _int(entry.get("truncated")),
            }
    if tag is not None and not out:
        raise CompareError(f"summary.json : modèle {tag!r} absent (présents : {sorted(seen)})")
    return out


def compare_summaries(a: dict[tuple[str, str], dict], b: dict[tuple[str, str], dict]) -> list[dict]:
    """Une ligne par (piste, catégorie) de A ou de B ; `missing` vaut "A" ou "B" si absente."""
    rows = []
    for key in sorted(set(a) | set(b)):
        row: dict = {
            "track": key[0],
            "category": key[1],
            "score": None,
            "truncation": None,
            "missing": None,
        }
        if key not in a or key not in b:
            row["missing"] = "A" if key not in a else "B"
            rows.append(row)
            continue
        ea, eb = a[key], b[key]
        if all(e[k] is not None for e in (ea, eb) for k in ("mean", "ci_low", "ci_high")):
            d, lo, hi = newcombe_diff(
                (ea["mean"], ea["ci_low"], ea["ci_high"]),
                (eb["mean"], eb["ci_low"], eb["ci_high"]),
            )
            row["score"] = {
                "a": ea["mean"],
                "b": eb["mean"],
                "diff": d,
                "ci_low": lo,
                "ci_high": hi,
            }
        if ea["graded"] > 0 and eb["graded"] > 0:
            wa = wilson(ea["truncated"], ea["graded"])
            wb = wilson(eb["truncated"], eb["graded"])
            d, lo, hi = newcombe_diff(wa, wb)
            row["truncation"] = {"a": wa[0], "b": wb[0], "diff": d, "ci_low": lo, "ci_high": hi}
        rows.append(row)
    return rows


# --- Rapport ----------------------------------------------------------------------------------


def _f(x: float | None, nd: int = 4) -> str:
    return "n/d" if x is None else f"{x:.{nd}f}"


def _pct(x: float | None) -> str:
    return "n/d" if x is None else f"{100 * x:.1f} %"


def render_markdown(result: dict, label_a: str, label_b: str) -> str:
    w = result["weights"]
    lines = ["# Comparaison de deux éditions", "", "## Poids", ""]
    lines.append(f"Poids du modèle : {w['total_weights']}.")
    lines += [
        "",
        "| Édition | Tenseurs touchés | Poids modifiés | Part du modèle |",
        "|---|---|---|---|",
    ]
    for lab, s in ((label_a, w["a"]), (label_b, w["b"])):
        lines.append(
            f"| {lab} | {len(s['tensors'])} | {s['modified_weights']} | {_pct(s['model_share'])} |"
        )
    lines += ["", f"Tenseurs touchés par les deux : {len(w['common'])}."]
    for lab, side in ((label_a, w["a"]), (label_b, w["b"])):
        for name in side.get("added", []):
            lines.append(
                f"- {lab} : tenseur ajouté : `{name}`, poids comptés par rapport à "
                "l'`embed_tokens` de la base"
            )
    detached = [lab for lab, side in ((label_a, w["a"]), (label_b, w["b"])) if side.get("added")]
    if detached:
        lines += [
            "",
            f"Écart de protocole : {', '.join(detached)} a détaché `lm_head` des embeddings liés "
            "(l'édition du MIT garde les embeddings liés) ; ces comparaisons ne sont pas à "
            "protocole égal.",
        ]
    if w["common"]:
        lines += [
            "",
            f"| Tenseur | Poids {label_a} | Poids {label_b} | Cosinus deltas | "
            "Cosinus gauche | Cosinus droit | Note |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in w["common"]:
            lines.append(
                f"| {c['name']} | {c['modified_a']} | {c['modified_b']} | "
                f"{_f(c['flat_cosine'])} | {_f(c['left_cosine'])} | {_f(c['right_cosine'])} | "
                f"{c['note'] or ''} |"
            )
        lines += ["", "Cosinus gauche et droit : valeur absolue (signe arbitraire en SVD)."]
    for lab, key, side in ((label_a, "only_a", w["a"]), (label_b, "only_b", w["b"])):
        lines += ["", f"Tenseurs propres à {lab} : {len(w[key])}."]
        lines += [f"- {n} : {side['tensors'][n]} poids" for n in w[key]]
    bench = result.get("bench")
    if bench is not None:
        lines += [
            "",
            "## Banc",
            "",
            f"Écarts = {label_a} - {label_b}, intervalle à 95 % (Newcombe, modèles traités comme "
            "indépendants ; Wilson pour le taux tronqué).",
            "",
            "| Piste | Catégorie | Score A / B | Écart score [IC] | Tronqué A / B | "
            "Écart tronqué [IC] |",
            "|---|---|---|---|---|---|",
        ]
        for r in bench:
            if r["missing"] is not None:
                lab = label_a if r["missing"] == "A" else label_b
                absent = f"absente de {lab}"
                lines.append(
                    f"| {r['track']} | {r['category']} | {absent} | n/d | {absent} | n/d |"
                )
                continue
            s, t = r["score"], r["truncation"]
            sc = "n/d" if s is None else f"{_f(s['a'], 3)} / {_f(s['b'], 3)}"
            sd = (
                "n/d"
                if s is None
                else f"{s['diff']:+.3f} [{s['ci_low']:+.3f} ; {s['ci_high']:+.3f}]"
            )
            tr = "n/d" if t is None else f"{_pct(t['a'])} / {_pct(t['b'])}"
            td = (
                "n/d"
                if t is None
                else f"{100 * t['diff']:+.1f} pt "
                f"[{100 * t['ci_low']:+.1f} ; {100 * t['ci_high']:+.1f}]"
            )
            lines.append(f"| {r['track']} | {r['category']} | {sc} | {sd} | {tr} | {td} |")
        if not bench:
            lines.append("")
            lines.append("Aucune piste ni catégorie dans les résumés.")
    return "\n".join(lines) + "\n"


def compare(
    original,
    edit_a,
    edit_b,
    summaries=(),
    tags=(None, None),
    label_a="A",
    label_b="B",
    diff_a=None,
    diff_b=None,
) -> tuple[dict, str]:
    if len(summaries) > 2:
        raise CompareError("au plus deux summary.json")
    if not summaries and any(t is not None for t in tags):
        raise CompareError("--tag-a/--tag-b sans --summary")
    if len(summaries) == 1 and (None in tags or tags[0] == tags[1]):
        raise CompareError("un seul summary.json : --tag-a et --tag-b distincts requis")
    result: dict = {
        "labels": {"a": label_a, "b": label_b},
        "weights": compare_checkpoints(original, edit_a, edit_b, diff_a, diff_b),
        "bench": None,
    }
    if summaries:
        sa = read_summary(summaries[0], tags[0])
        sb = read_summary(summaries[-1], tags[1])
        result["bench"] = compare_summaries(sa, sb)
    return result, render_markdown(result, label_a, label_b)


def write_report(result: dict, markdown: str, out: str | Path) -> tuple[Path, Path]:
    out = Path(out)
    if out.suffix.lower() == ".json":
        raise CompareError("--out désigne le rapport Markdown, pas le .json (écrit à côté)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    js = out.with_suffix(".json")
    js.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out, js


def add_parser(sub) -> None:
    p = sub.add_parser(
        "compare-edits",
        help="compare deux éditions d'un même modèle (poids, puis scores du banc)",
        description="Rapport Markdown et JSON : tenseurs touchés, cosinus des deltas, écarts.",
    )
    p.add_argument("--original", required=True, help="checkpoint original (fichier ou dossier)")
    p.add_argument("--edit-a", help="checkpoint de l'édition A (ou --diff-a)")
    p.add_argument("--edit-b", help="checkpoint de l'édition B (ou --diff-b)")
    p.add_argument("--diff-a", help="diff.json de séance rejoué en mémoire, à la place de --edit-a")
    p.add_argument("--diff-b", help="diff.json de séance rejoué en mémoire, à la place de --edit-b")
    p.add_argument("--label-a", default="A", help="nom de l'édition A dans le rapport")
    p.add_argument("--label-b", default="B", help="nom de l'édition B dans le rapport")
    p.add_argument(
        "--summary",
        action="append",
        help="summary.json du banc (une fois : les deux modèles y figurent ; deux fois : A puis B)",
    )
    p.add_argument("--tag-a", help="tag du modèle A dans son summary.json")
    p.add_argument("--tag-b", help="tag du modèle B dans son summary.json")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"rapport Markdown (défaut {DEFAULT_OUT})")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        result, md = compare(
            args.original,
            args.edit_a,
            args.edit_b,
            summaries=args.summary or (),
            tags=(args.tag_a, args.tag_b),
            label_a=args.label_a,
            label_b=args.label_b,
            diff_a=args.diff_a,
            diff_b=args.diff_b,
        )
        md_path, js_path = write_report(result, md, args.out)
    except (CompareError, SafetensorsError, OSError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    print(f"Rapport : {md_path}")
    print(f"Chiffres : {js_path}")
    return 0
