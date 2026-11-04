"""Gel d'une séance d'édition et diff vérifiable rejouable bit à bit (SPEC CAP-2, CAP-1).

`session-freeze` compare `work/` à l'original de la séance et écrit, dans `<séance>/freeze/` :
- `diff.json` : métadonnées et sha256 (format FORMAT) ;
- `diff.bin` : pour chaque tenseur modifié, les indices plats des éléments changés (u4 ou u8
  petit-boutiste) puis leurs valeurs après édition, en octets bruts du dtype d'origine ;
- `journal.jsonl` : copie du journal au moment du gel, dont le sha256 figure dans `diff.json`.
Il pose `frozen` (définitif : le harnais refuse ensuite tout lot) et ajoute un événement `freeze`.

`apply-diff` rejoue le diff sur l'original : le checkpoint produit redonne exactement les sha256
publiés (vérifiés avant écriture, refus sinon) ; les autres fichiers de l'original sont recopiés.

Plafond : un tenseur dont plus de MAX_CHANGED_SHARE des éléments changent est refusé, car son
diff reviendrait à publier le tenseur complet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from slm_weight_editing import harness
from slm_weight_editing.harness import HarnessError
from slm_weight_editing.safetensors_file import SafeTensor, SafetensorsError, open_checkpoint

FORMAT = "slm-weight-editing/edit-diff-v1"
MAX_CHANGED_SHARE = 0.10
DIFF_FILE = "diff.json"
VALUES_FILE = "diff.bin"
FREEZE_DIR = "freeze"
_FREEZE_RETRIES = 20
_FREEZE_DELAY = 0.5
# Largeur en octets de chaque dtype safetensors (comparaison bit à bit via un entier non signé).
_WIDTH = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}  # fmt: skip


class FreezeError(HarnessError):
    """Gel refusé (code 2)."""


class DiffError(ValueError):
    """Diff illisible, altéré ou incompatible avec l'original."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _uint(dtype: str, name: str) -> np.dtype:
    if dtype not in _WIDTH:
        raise FreezeError(f"{name} : dtype {dtype} non géré")
    return np.dtype(f"<u{_WIDTH[dtype]}")


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# --- Diff -------------------------------------------------------------------------------------


def compute_tensor_diff(
    base: dict[str, SafeTensor], work: dict[str, SafeTensor]
) -> list[tuple[dict, np.ndarray, np.ndarray]]:
    """Tenseurs qui diffèrent : (entrée sans offsets, indices plats, valeurs après en uint)."""
    missing, extra = sorted(set(base) - set(work)), sorted(set(work) - set(base))
    if missing or extra:
        raise FreezeError(
            f"work/ et l'original n'ont pas les mêmes tenseurs "
            f"(manquants : {missing[:3]}, en trop : {extra[:3]})"
        )
    out = []
    for name in sorted(base):
        b, w = base[name], work[name]
        if b.shape != w.shape or b.dtype != w.dtype:
            raise FreezeError(
                f"{name} : forme {list(w.shape)} {w.dtype} dans work/, "
                f"{list(b.shape)} {b.dtype} dans l'original"
            )
        raw_b, raw_w = b.read_bytes(), w.read_bytes()
        if raw_b == raw_w:
            continue
        u = _uint(b.dtype, name)
        arr_b, arr_w = np.frombuffer(raw_b, u), np.frombuffer(raw_w, u)
        idx = np.flatnonzero(arr_b != arr_w)
        numel = arr_b.size
        if len(idx) > MAX_CHANGED_SHARE * numel:
            raise FreezeError(
                f"{name} : {len(idx)}/{numel} éléments changés "
                f"({len(idx) / numel:.1%}), plafond {MAX_CHANGED_SHARE:.0%} : le diff "
                "reviendrait à publier le tenseur complet"
            )
        entry = {
            "name": name,
            "shape": list(b.shape),
            "dtype": b.dtype,
            "numel": numel,
            "changed": int(len(idx)),
            "sha256_before": _sha(raw_b),
            "sha256_after": _sha(raw_w),
        }
        out.append((entry, idx, arr_w[idx]))
    return out


def changed_files(base_files: dict[str, str], work_files: dict[str, str]) -> list[dict]:
    """Fichiers non safetensors qui diffèrent de la base (modifiés, ajoutés ou supprimés)."""
    return [
        {"path": p, "sha256_before": base_files.get(p), "sha256_after": work_files.get(p)}
        for p in sorted(set(base_files) | set(work_files))
        if base_files.get(p) != work_files.get(p)
    ]


def _pack_values(parts: list[tuple[dict, np.ndarray, np.ndarray]]) -> tuple[list[dict], bytes]:
    chunks, offset, entries = [], 0, []
    for entry, idx, vals in parts:
        idx_dtype = "<u4" if entry["numel"] <= 2**32 else "<u8"
        idx_bytes = idx.astype(idx_dtype).tobytes()
        val_bytes = vals.tobytes()
        entries.append(
            entry
            | {
                "index_dtype": idx_dtype.lstrip("<"),
                "indices_offset": offset,
                "values_offset": offset + len(idx_bytes),
            }
        )
        chunks += [idx_bytes, val_bytes]
        offset += len(idx_bytes) + len(val_bytes)
    return entries, b"".join(chunks)


def freeze_session(
    session_id: str,
    original: str | Path,
    *,
    sessions_root: str | Path = harness.DEFAULT_SESSIONS_ROOT,
    out: str | Path | None = None,
) -> dict:
    """Gèle la séance et écrit le diff ; renvoie le contenu de diff.json."""
    directory = harness.session_dir(session_id, sessions_root)
    harness.read_state(directory)
    journal = harness.read_journal(directory)
    if any(e.get("event") == "freeze" for e in journal):
        raise FreezeError(f"séance {session_id} déjà gelée : diff déjà produit")
    opened = next((e for e in journal if e.get("event") == "open"), None)
    if opened is None:
        raise FreezeError("journal sans ligne open")
    original = Path(original)
    if not original.is_dir():
        raise FreezeError(f"original introuvable : {original}")
    out_dir = Path(out) if out else directory / FREEZE_DIR

    # 1. L'original passé doit être celui de l'ouverture ; rien n'est écrit avant.
    got = harness.tensor_hashes(original)
    want = opened["base_tensors_sha256"]
    if got != want:
        wrong = sorted(n for n in set(got) | set(want) if got.get(n) != want.get(n))
        raise FreezeError(
            f"l'original ne correspond pas à la base de la séance ({len(wrong)} tenseurs "
            f"différents, par exemple {wrong[:3]}) ; gel refusé"
        )
    if harness.work_file_hashes(original) != opened["base_files_sha256"]:
        raise FreezeError("les fichiers de l'original diffèrent de la base de la séance")

    # 2. Diff calculé avant le gel : un plafond dépassé ne fige rien.
    work = directory / harness.WORK_DIR
    parts = compute_tensor_diff(open_checkpoint(original), open_checkpoint(work))
    work_files = harness.work_file_hashes(work)
    files = changed_files(opened["base_files_sha256"], work_files)

    # 3. Gel (sous le verrou du harnais, réessai borné si un lot tourne), puis work/ doit être
    # resté identique à ce qui a été comparé.
    _freeze_state(directory)
    expected = dict(want) | {e["name"]: e["sha256_after"] for e, _, _ in parts}
    if harness.tensor_hashes(work) != expected or harness.work_file_hashes(work) != work_files:
        raise FreezeError("work/ a changé pendant le gel ; relancer session-freeze")

    entries, values = _pack_values(parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    journal_path = directory / harness.JOURNAL_FILE
    journal_sha = harness._sha256_file(journal_path)
    shutil.copyfile(journal_path, out_dir / harness.JOURNAL_FILE)
    state = harness.read_state(directory)
    diff = {
        "format": FORMAT,
        "session_id": session_id,
        "base_revision": state.base_revision,
        "budget": state.budget,
        "batches_used": harness.used_batches(directory, state),
        "journal": {"file": harness.JOURNAL_FILE, "sha256": journal_sha},
        "max_changed_share": MAX_CHANGED_SHARE,
        "values_file": {"name": VALUES_FILE, "sha256": _sha(values), "bytes": len(values)},
        "tensors": entries,
        "files_changed": files,
        "frozen_at": state.updated_at,
    }
    text = (json.dumps(diff, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _write_atomic(out_dir / VALUES_FILE, values)
    _write_atomic(out_dir / DIFF_FILE, text)
    harness._append_journal(
        directory,
        {
            "event": "freeze",
            "at": harness._now(),
            "diff_sha256": _sha(text),
            "values_sha256": diff["values_file"]["sha256"],
            "journal_sha256": journal_sha,
            "tensors_changed": len(entries),
            "files_changed": len(files),
        },
    )
    return diff


def _freeze_state(directory: Path) -> None:
    for attempt in range(_FREEZE_RETRIES):
        try:
            harness.update_state(directory, frozen=True)
            return
        except HarnessError as exc:
            if not (directory / harness.LOCK_FILE).exists():
                raise
            if attempt == _FREEZE_RETRIES - 1:
                raise FreezeError(
                    f"un lot tourne encore après {_FREEZE_RETRIES} essais ; gel non posé "
                    f"(attendre la fin du lot, ou supprimer {harness.LOCK_FILE} s'il est périmé)"
                ) from exc
            time.sleep(_FREEZE_DELAY)


# --- Rejeu ------------------------------------------------------------------------------------


def load_diff(path: str | Path) -> tuple[dict, bytes]:
    """Lit diff.json et son fichier de valeurs (sha256 vérifié)."""
    path = Path(path)
    try:
        meta = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise DiffError(f"diff illisible : {path} ({exc})") from exc
    found = meta.get("format") if isinstance(meta, dict) else None
    if found != FORMAT:
        raise DiffError(f"{path} : format {found!r}, {FORMAT} attendu")
    try:
        vf = meta["values_file"]
        values = (path.parent / vf["name"]).read_bytes()
        meta["tensors"]
    except (KeyError, TypeError) as exc:
        raise DiffError(f"{path} : diff invalide ({exc!r})") from exc
    except OSError as exc:
        raise DiffError(f"fichier de valeurs illisible : {exc}") from exc
    if _sha(values) != vf.get("sha256") or len(values) != vf.get("bytes"):
        raise DiffError(f"{vf['name']} : sha256 différent de celui du diff (fichier altéré)")
    return meta, values


def _entry_arrays(entry: dict, values: bytes) -> tuple[np.ndarray, np.ndarray]:
    try:
        n = entry["changed"]
        idx_t = np.dtype("<" + entry["index_dtype"])
        val_t = _uint(entry["dtype"], entry["name"])
        idx = np.frombuffer(values, idx_t, n, entry["indices_offset"])
        vals = np.frombuffer(values, val_t, n, entry["values_offset"])
    except (KeyError, TypeError, ValueError, FreezeError) as exc:
        raise DiffError(f"entrée de diff invalide : {entry.get('name')!r} ({exc})") from exc
    if n and int(idx.max()) >= entry["numel"]:
        raise DiffError(f"{entry['name']} : indice hors du tenseur")
    return idx, vals


def replay_tensors(meta: dict, values: bytes, base: dict[str, SafeTensor]):
    """Génère (entrée, octets avant, octets après) en vérifiant les sha256 avant et après."""
    for entry in meta["tensors"]:
        name = entry["name"]
        st = base.get(name)
        if st is None:
            raise DiffError(f"{name} absent de l'original")
        if list(st.shape) != entry["shape"] or st.dtype != entry["dtype"]:
            raise DiffError(f"{name} : forme ou dtype différents de l'original")
        before = st.read_bytes()
        if _sha(before) != entry["sha256_before"]:
            raise DiffError(f"{name} : l'original n'a pas le sha256 avant du diff ; rejeu refusé")
        idx, vals = _entry_arrays(entry, values)
        arr = np.frombuffer(before, _uint(st.dtype, name)).copy()
        arr[idx] = vals
        after = arr.tobytes()
        if _sha(after) != entry["sha256_after"]:
            raise DiffError(f"{name} : le rejeu ne redonne pas le sha256 après publié")
        yield entry, before, after


def diff_deltas(diff_path: str | Path, base: dict[str, SafeTensor]) -> dict[str, np.ndarray]:
    """Deltas (après - avant, float64) rejoués en mémoire : entrée de compare_deltas."""
    from slm_weight_editing.compare_edits import bytes_to_f64

    meta, values = load_diff(diff_path)
    out = {}
    for entry, before, after in replay_tensors(meta, values, base):
        shape, dtype, name = tuple(entry["shape"]), entry["dtype"], entry["name"]
        delta = bytes_to_f64(after, dtype, shape, name) - bytes_to_f64(before, dtype, shape, name)
        if np.any(delta != 0):
            out[name] = delta
    return out


def apply_diff(diff_path: str | Path, original: str | Path, out: str | Path) -> dict:
    """Rejoue le diff sur l'original dans `out` ; renvoie le diff chargé (`files_changed`)."""
    meta, values = load_diff(diff_path)
    original, out = Path(original), Path(out)
    if not original.is_dir():
        raise DiffError(f"original introuvable : {original}")
    if out.exists() and any(out.iterdir()):
        raise DiffError(f"sortie non vide : {out}")
    base = open_checkpoint(original)
    patched = {e["name"]: after for e, _, after in replay_tensors(meta, values, base)}
    out.mkdir(parents=True, exist_ok=True)
    try:
        for src in sorted(p for p in original.rglob("*") if p.is_file()):
            dst = out / src.relative_to(original)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        for name, after in patched.items():
            st = base[name]
            with (out / st.path.relative_to(original)).open("r+b") as f:
                f.seek(st.start)
                f.write(after)
        result = open_checkpoint(out)
        for entry in meta["tensors"]:
            if _sha(result[entry["name"]].read_bytes()) != entry["sha256_after"]:
                raise DiffError(f"{entry['name']} : sha256 après différent dans la sortie")
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return meta


# --- CLI --------------------------------------------------------------------------------------


def add_parser(sub: argparse._SubParsersAction) -> None:
    fr = sub.add_parser(
        "session-freeze",
        help="gèle une séance et écrit le diff vérifiable de l'édition",
        description=(
            "Pose le gel (définitif : plus aucun lot), compare work/ à l'original et écrit "
            f"{DIFF_FILE}, {VALUES_FILE} et la copie du journal dans <séance>/{FREEZE_DIR}/. "
            f"Refuse un tenseur dont plus de {MAX_CHANGED_SHARE:.0%} des éléments changent."
        ),
    )
    fr.add_argument("--session", required=True)
    fr.add_argument("--original", required=True, help="checkpoint de base de la séance (dossier)")
    fr.add_argument("--out", help=f"dossier de sortie (défaut : <séance>/{FREEZE_DIR})")
    fr.add_argument("--sessions-root", default=harness.DEFAULT_SESSIONS_ROOT)
    ap = sub.add_parser(
        "apply-diff",
        help="rejoue un diff sur l'original et produit le checkpoint édité",
        description="Vérifie les sha256 avant et après ; recopie les autres fichiers de la base.",
    )
    ap.add_argument("--diff", required=True, help="diff.json produit par session-freeze")
    ap.add_argument("--original", required=True, help="checkpoint de base (dossier)")
    ap.add_argument("--out", required=True, help="dossier de sortie (sous models/, vide ou absent)")
    for p in (fr, ap):
        p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        if args.command == "session-freeze":
            diff = freeze_session(
                args.session, args.original, sessions_root=args.sessions_root, out=args.out
            )
            where = args.out or harness.session_dir(args.session, args.sessions_root) / FREEZE_DIR
            print(
                f"Séance {args.session} gelée : {len(diff['tensors'])} tenseurs modifiés, "
                f"{sum(t['changed'] for t in diff['tensors'])} éléments"
            )
            print(f"Diff : {Path(where) / DIFF_FILE}")
            for f in diff["files_changed"]:
                print(f"Attention : {f['path']} diffère de la base (non rejoué par apply-diff)")
        else:
            meta = apply_diff(args.diff, args.original, args.out)
            print(f"Checkpoint rejoué : {args.out} ({len(meta['tensors'])} tenseurs vérifiés)")
            for f in meta["files_changed"]:
                print(f"Attention : {f['path']} différait de la base dans la séance (non rejoué)")
    except (HarnessError, DiffError, SafetensorsError, OSError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return getattr(exc, "exit_code", 1)
    return 0
