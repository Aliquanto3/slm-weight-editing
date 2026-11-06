"""Gel d'une séance d'édition et diff vérifiable rejouable bit à bit (SPEC CAP-2, CAP-1).

`session-freeze` compare `work/` à l'original de la séance et écrit, dans `<séance>/freeze/` :
- `diff.json` : métadonnées et sha256 (format FORMAT) ;
- `diff.bin` : pour chaque tenseur modifié, les indices plats des éléments changés (u4 ou u8
  petit-boutiste) puis leurs valeurs après édition, en octets bruts du dtype d'origine ;
- `journal.jsonl` : version publique du journal au moment du gel (chemins locaux masqués par
  `public_path`), dont le sha256 figure dans `diff.json` ; le journal brut reste dans la séance.
Tout le gel se fait sous le verrou de séance du harnais. Il pose `frozen` (définitif : le
harnais refuse ensuite tout lot) et ajoute un événement `freeze`.

`apply-diff` rejoue le diff sur l'original : le checkpoint produit redonne exactement les sha256
publiés (vérifiés avant écriture, refus sinon) ; les autres fichiers de l'original sont recopiés.

Plafond : un tenseur dont plus de MAX_CHANGED_SHARE des éléments changent est refusé, car son
diff reviendrait à publier le tenseur complet (décision du 2026-11-04 : éditions creuses
imposées par la consigne de séance). Un tenseur de 9 éléments ou moins ne peut donc changer
d'aucun élément.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from slm_weight_editing import harness
from slm_weight_editing.harness import MAX_CHANGED_SHARE, HarnessError
from slm_weight_editing.lm_head import (
    LM_HEAD,
    append_tensor,
    detach_event,
    transform_config,
    write_config,
)
from slm_weight_editing.report_gguf import public_text
from slm_weight_editing.safetensors_file import SafeTensor, SafetensorsError, open_checkpoint

FORMAT = "slm-weight-editing/edit-diff-v2"
# v1 (sans `detached_lm_head`) reste lisible par apply-diff, compare-edits et report-gguf.
FORMATS = ("slm-weight-editing/edit-diff-v1", FORMAT)
DIFF_FILE = "diff.json"
VALUES_FILE = "diff.bin"
FREEZE_DIR = "freeze"
_FREEZE_RETRIES = 20
_FREEZE_DELAY = 0.5
_INDEX_DTYPES = {"u4": 4, "u8": 8}
# Champs du journal qui portent un chemin local (open : base, batch : fichier de données).
_PATH_FIELDS = ("base_checkpoint", "data_path")
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


def _diff_pair(
    name: str, b: SafeTensor, w: SafeTensor, *, extra: dict | None = None, keep_equal: bool = False
) -> tuple[dict, np.ndarray, np.ndarray] | None:
    """Entrée de diff de `w` contre `b` (None si identiques, sauf `keep_equal`)."""
    if b.shape != w.shape or b.dtype != w.dtype:
        raise FreezeError(
            f"{name} : forme {list(w.shape)} {w.dtype} dans work/, "
            f"{list(b.shape)} {b.dtype} dans l'original"
        )
    raw_b, raw_w = b.read_bytes(), w.read_bytes()
    if raw_b == raw_w and not keep_equal:
        return None
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
    } | (extra or {})
    return entry, idx, arr_w[idx]


def compute_tensor_diff(
    base: dict[str, SafeTensor], work: dict[str, SafeTensor], detached: dict | None = None
) -> list[tuple[dict, np.ndarray, np.ndarray]]:
    """Tenseurs qui diffèrent : (entrée sans offsets, indices plats, valeurs après en uint).

    `detached` (ligne `detach_lm_head` du journal) autorise le seul `lm_head.weight` en plus ;
    il est comparé à l'`embed_tokens` de la base et figure toujours au diff (entrée `added`),
    même sans édition : le détachement lui-même doit se rejouer.
    """
    source = detached["source"] if detached else None
    allowed = {LM_HEAD} if detached else set()
    missing = sorted(set(base) - set(work))
    extra = sorted(set(work) - set(base) - allowed)
    if detached and (LM_HEAD not in work or source not in base):
        missing.append(LM_HEAD if LM_HEAD not in work else str(source))
    if missing or extra:
        raise FreezeError(
            f"work/ et l'original n'ont pas les mêmes tenseurs "
            f"(manquants : {missing[:3]}, en trop : {extra[:3]})"
        )
    out = []
    for name in sorted(base):
        pair = _diff_pair(name, base[name], work[name])
        if pair is not None:
            out.append(pair)
    if detached:
        added = {"added": True, "source": source}
        out.append(_diff_pair(LM_HEAD, base[source], work[LM_HEAD], extra=added, keep_equal=True))
    return out


IGNORED_DIR = ".cache"


def is_ignored_file(path: object) -> bool:
    """Métadonnées de téléchargement HF (`.cache/...`) : jamais une modification de la séance.

    Un chemin qui contient `..` n'est jamais ignoré (`.cache/../config.json` sortirait du cache).
    """
    if not isinstance(path, str):
        return False
    parts = path.replace("\\", "/").split("/")
    return parts[0] == IGNORED_DIR and ".." not in parts


def relevant_files(files: list[dict]) -> list[dict]:
    """Entrées `files_changed` hors `.cache/` (les seules qui comptent pour apply et report)."""
    return [f for f in files if not is_ignored_file(f.get("path"))]


def changed_files(
    base_files: dict[str, str], work_files: dict[str, str], known: dict | None = None
) -> list[dict]:
    """Fichiers non safetensors qui diffèrent de la base (modifiés, ajoutés ou supprimés).

    Les chemins sous `.cache/` sont ignorés (métadonnées de téléchargement HF absentes de work/).
    `known` (champ `config` d'un détachement vérifié : path, sha256_before, sha256_after) retire
    le seul `config.json` dont l'écart est cette transformation, rejouée par apply-diff.
    """

    def is_known(path: str) -> bool:
        return (
            known is not None
            and path == known["path"]
            and base_files.get(path) == known["sha256_before"]
            and work_files.get(path) == known["sha256_after"]
        )

    return [
        {"path": p, "sha256_before": base_files.get(p), "sha256_after": work_files.get(p)}
        for p in sorted(set(base_files) | set(work_files))
        if base_files.get(p) != work_files.get(p) and not is_ignored_file(p) and not is_known(p)
    ]


def _known_config(original: Path, detached: dict | None) -> dict | None:
    """Config du détachement, si la transformation redonne bien le sha256 journalisé."""
    if detached is None:
        return None
    cfg = detached["config"]
    path = original / cfg["path"]
    try:
        before = path.read_bytes()
        after, _ = transform_config(before.decode("utf-8"))
    except OSError, ValueError, HarnessError:
        return None
    if _sha(before) != cfg["sha256_before"] or _sha(after.encode("utf-8")) != cfg["sha256_after"]:
        return None
    return cfg


def _check_detached_config(original: Path, work: Path, detached: dict | None) -> dict | None:
    """Config du détachement, vérifiée : rejouable sur l'original et toujours déliée dans work/.

    Sinon le modèle publié par apply-diff (config déliée) ne serait pas celui mesuré pendant la
    séance (lots tournés avec `lm_head` ignoré si work/ est relié) : gel refusé.
    """
    if detached is None:
        return None
    known = _known_config(original, detached)
    if known is None:
        raise FreezeError(
            f"{detached['config']['path']} de l'original ne redonne pas la transformation "
            "journalisée par detach-lm-head : diff non rejouable, gel refusé"
        )
    path = work / known["path"]
    try:
        text = path.read_text("utf-8")
        untied, _ = transform_config(text)
        still_untied = json.loads(untied) == json.loads(text)
    except OSError, ValueError, HarnessError:
        still_untied = False
    if not still_untied:
        raise FreezeError(
            f"work/{known['path']} ne déclare plus tie_word_embeddings: false après "
            "detach-lm-head : le modèle mesuré ignorait lm_head, gel refusé"
        )
    return known


def _detached_field(detached: dict | None) -> dict | None:
    """Champ `detached_lm_head` de diff.json (source, nom et transformation de la config)."""
    if detached is None:
        return None
    keys = ("path", "sha256_before", "sha256_after", "keys")
    return {
        "source": detached["source"],
        "lm_head": detached["lm_head"],
        "config": {k: detached["config"][k] for k in keys},
    }


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


def public_path(value: str) -> str:
    """Chemin publiable : relatif au dossier courant s'il est dessous, sinon `.../<nom>`.

    Le contenu reste identifié par les sha256 du journal ; seul l'emplacement local disparaît.
    """
    path = Path(value)
    if not path.is_absolute():
        return public_text(path.as_posix())
    cwd, home = Path.cwd().resolve(), Path.home().resolve()
    # Un dossier courant au-dessus du dossier personnel (racine du disque, /home) laisserait
    # le nom d'utilisateur dans le chemin relatif : on masque alors comme hors du dossier.
    if path.is_relative_to(cwd) and (cwd.is_relative_to(home) or not path.is_relative_to(home)):
        return path.relative_to(cwd).as_posix()
    return f".../{path.name}"


def public_journal(raw: bytes) -> bytes:
    """Journal publiable : chemins locaux masqués, dossier personnel remplacé par `~`."""
    lines = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        for key in _PATH_FIELDS:
            if isinstance(entry.get(key), str):
                entry[key] = public_path(entry[key])
        lines.append(public_text(json.dumps(entry, ensure_ascii=False)) + "\n")
    return "".join(lines).encode("utf-8")


def freeze_session(
    session_id: str,
    original: str | Path,
    *,
    sessions_root: str | Path = harness.DEFAULT_SESSIONS_ROOT,
    out: str | Path | None = None,
) -> dict:
    """Gèle la séance et écrit le diff ; renvoie le contenu de diff.json.

    Tout se passe sous le verrou de séance (réessai borné si un lot tourne) : deux gels
    concurrents ne peuvent écrire ni deux diffs ni deux événements freeze.
    """
    directory = harness.session_dir(session_id, sessions_root)
    harness.read_state(directory)
    for attempt in range(_FREEZE_RETRIES):
        lock = harness._session_lock(directory)
        try:
            lock.__enter__()
        except HarnessError as exc:
            if attempt == _FREEZE_RETRIES - 1:
                raise FreezeError(f"gel non posé après {_FREEZE_RETRIES} essais : {exc}") from exc
            time.sleep(_FREEZE_DELAY)
            continue
        try:
            return _freeze_locked(directory, session_id, Path(original), out)
        finally:
            lock.__exit__(None, None, None)
    raise AssertionError("inaccessible")


def _freeze_locked(directory: Path, session_id: str, original: Path, out) -> dict:
    journal = harness.read_journal(directory)
    if any(e.get("event") == "freeze" for e in journal):
        raise FreezeError(f"séance {session_id} déjà gelée : diff déjà produit")
    opened = next((e for e in journal if e.get("event") == "open"), None)
    if opened is None:
        raise FreezeError("journal sans ligne open")
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
    detached = detach_event(journal)
    parts = compute_tensor_diff(open_checkpoint(original), open_checkpoint(work), detached)
    work_files = harness.work_file_hashes(work)
    known = _check_detached_config(original, work, detached)
    files = changed_files(opened["base_files_sha256"], work_files, known)

    # 3. Gel (verrou déjà tenu : aucun lot ne peut démarrer), puis work/ doit être resté
    # identique à ce qui a été comparé (l'éditeur écrit work/ sans passer par le verrou).
    _freeze_state(directory)
    resume = "séance déjà gelée, relancer session-freeze pour écrire le diff"
    expected = dict(want) | {e["name"]: e["sha256_after"] for e, _, _ in parts}
    if harness.tensor_hashes(work) != expected or harness.work_file_hashes(work) != work_files:
        raise FreezeError(f"work/ a changé pendant le gel ; {resume}")

    try:
        entries, values = _pack_values(parts)
        out_dir.mkdir(parents=True, exist_ok=True)
        public = public_journal((directory / harness.JOURNAL_FILE).read_bytes())
        journal_sha = _sha(public)
        state = harness.read_state(directory)
        blocages = _public_blocages(directory)
        diff = {
            "format": FORMAT,
            "session_id": session_id,
            "base_revision": state.base_revision,
            "budget": state.budget,
            "batches_used": harness.used_batches(directory, state),
            "journal": {"file": harness.JOURNAL_FILE, "sha256": journal_sha},
            "max_changed_share": MAX_CHANGED_SHARE,
            "values_file": {"name": VALUES_FILE, "sha256": _sha(values), "bytes": len(values)},
            "detached_lm_head": _detached_field(detached),
            "tensors": entries,
            "files_changed": files,
            "frozen_at": state.updated_at,
        }
        if blocages is not None:
            diff["blocages"] = {"file": BLOCAGES_FILE, "sha256": _sha(blocages)}
        text = (json.dumps(diff, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        _write_atomic(out_dir / harness.JOURNAL_FILE, public)
        _write_atomic(out_dir / VALUES_FILE, values)
        if blocages is not None:
            _write_atomic(out_dir / BLOCAGES_FILE, blocages)
        _write_atomic(out_dir / DIFF_FILE, text)
    except (OSError, ValueError, HarnessError) as exc:
        raise FreezeError(f"écriture du diff impossible ({exc}) ; {resume}", 1) from exc
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


BLOCAGES_FILE = "BLOCAGES.md"


def _public_blocages(directory: Path) -> bytes | None:
    """`notes/BLOCAGES.md` de l'espace de la séance (story 2.5), chemins personnels masqués.

    Séance hors d'un espace (`<espace>/sessions/<id>`, à côté de `workspace.json`) : None.
    """
    workspace = directory.resolve().parent.parent
    path = workspace / "notes" / BLOCAGES_FILE
    if not (workspace / harness.WORKSPACE_FILE).is_file() or not path.is_file():
        return None
    return public_text(path.read_text(encoding="utf-8", errors="replace")).encode("utf-8")


def _freeze_state(directory: Path) -> None:
    """Pose `frozen` ; l'appelant tient déjà le verrou de séance."""
    state = harness.read_state(directory)
    if not state.frozen:
        new = dataclasses.replace(state, frozen=True, updated_at=harness._now())
        harness._write_state(directory, new)


# --- Rejeu ------------------------------------------------------------------------------------


def load_diff(path: str | Path) -> tuple[dict, bytes]:
    """Lit diff.json et son fichier de valeurs (sha256 vérifié)."""
    path = Path(path)
    try:
        meta = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise DiffError(f"diff illisible : {path} ({exc})") from exc
    found = meta.get("format") if isinstance(meta, dict) else None
    if found not in FORMATS:
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
    files = meta.get("files_changed")
    if (
        not isinstance(meta["tensors"], list)
        or not isinstance(files, list)
        or not all(isinstance(f, dict) and isinstance(f.get("path"), str) for f in files)
    ):
        raise DiffError(f"{path} : diff invalide (tensors ou files_changed mal formés)")
    detached = _check_detached(meta.get("detached_lm_head"), path)
    meta["detached_lm_head"] = detached
    for entry in meta["tensors"]:
        _check_entry(entry, len(values))
        if entry.get("added") is not None and (
            entry["added"] is not True
            or detached is None
            or entry["name"] != LM_HEAD
            or entry.get("source") != detached["source"]
        ):
            raise DiffError(f"{entry['name']} : tenseur ajouté sans détachement correspondant")
    if detached is not None and not any(e.get("added") for e in meta["tensors"]):
        raise DiffError(f"{path} : détachement de lm_head sans tenseur {LM_HEAD}")
    return meta, values


def _check_detached(value: object, path: Path) -> dict | None:
    """Champ `detached_lm_head` (v2) : None s'il est absent (v1) ou nul."""
    if value is None:
        return None
    cfg = value.get("config") if isinstance(value, dict) else None
    ok = (
        isinstance(cfg, dict)
        and isinstance(value.get("source"), str)
        and value.get("lm_head") == LM_HEAD
        and all(isinstance(cfg.get(k), str) for k in ("path", "sha256_before", "sha256_after"))
        and isinstance(cfg.get("keys"), list)
    )
    if not ok or Path(cfg["path"]).name != cfg["path"]:
        raise DiffError(f"{path} : detached_lm_head mal formé")
    return value


def _is_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _check_entry(entry: object, size: int) -> None:
    """Structure d'une entrée : types, numel = produit de la forme, plages dans diff.bin."""
    if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
        raise DiffError(f"entrée de diff invalide : {entry!r:.80}")
    name = entry["name"]
    shape, numel, n = entry.get("shape"), entry.get("numel"), entry.get("changed")
    if not isinstance(shape, list) or not all(_is_int(d) for d in shape):
        raise DiffError(f"{name} : forme invalide dans le diff")
    if not _is_int(numel) or numel != math.prod(shape) or not _is_int(n) or n > numel:
        raise DiffError(f"{name} : numel ou changed incohérents avec la forme")
    if entry.get("index_dtype") not in _INDEX_DTYPES or entry.get("dtype") not in _WIDTH:
        raise DiffError(f"{name} : index_dtype ou dtype non géré")
    if entry["index_dtype"] == "u4" and numel > 2**32:
        raise DiffError(f"{name} : index_dtype u4 trop étroit pour {numel} éléments")
    for key, width in (
        ("indices_offset", _INDEX_DTYPES[entry["index_dtype"]]),
        ("values_offset", _WIDTH[entry["dtype"]]),
    ):
        offset = entry.get(key)
        if not _is_int(offset) or offset + n * width > size:
            raise DiffError(f"{name} : {key} hors de {VALUES_FILE}")
    for key in ("sha256_before", "sha256_after"):
        if not isinstance(entry.get(key), str):
            raise DiffError(f"{name} : {key} absent du diff")


def _entry_arrays(entry: dict, values: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Indices et valeurs d'une entrée validée par `_check_entry`."""
    n = entry["changed"]
    idx = np.frombuffer(values, "<" + entry["index_dtype"], n, entry["indices_offset"])
    vals = np.frombuffer(values, _uint(entry["dtype"], entry["name"]), n, entry["values_offset"])
    if n > 1 and not bool(np.all(idx[1:] > idx[:-1])):
        raise DiffError(f"{entry['name']} : indices non triés ou en double")
    if n and int(idx[-1]) >= entry["numel"]:
        raise DiffError(f"{entry['name']} : indice hors du tenseur")
    return idx, vals


def replay_tensors(meta: dict, values: bytes, base: dict[str, SafeTensor]):
    """Génère (entrée, octets avant, octets après) en vérifiant les sha256 avant et après."""
    for entry in meta["tensors"]:
        name = entry["name"]
        # Tenseur ajouté (lm_head détaché) : comparé à l'embed_tokens de la base.
        source = entry["source"] if entry.get("added") else name
        st = base.get(source)
        if st is None:
            raise DiffError(f"{source} absent de l'original")
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
    patched, added = {}, {}
    for e, _, after in replay_tensors(meta, values, base):
        (added if e.get("added") else patched)[e["name"]] = after
    detached = meta["detached_lm_head"]
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
        if detached is not None:
            _replay_detach(detached, added[LM_HEAD], base, original, out)
        result = open_checkpoint(out)
        for entry in meta["tensors"]:
            if _sha(result[entry["name"]].read_bytes()) != entry["sha256_after"]:
                raise DiffError(f"{entry['name']} : sha256 après différent dans la sortie")
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return meta


def _replay_detach(
    detached: dict, lm_head: bytes, base: dict[str, SafeTensor], original: Path, out: Path
) -> None:
    """Rejoue le détachement dans `out` : tenseur ajouté, puis `config.json` délié."""
    embed = base[detached["source"]]
    cfg = detached["config"]
    config = out / cfg["path"]
    try:
        before = config.read_bytes()
        new_text, _ = transform_config(before.decode("utf-8"))
    except (OSError, ValueError, HarnessError) as exc:
        raise DiffError(f"{cfg['path']} illisible dans l'original : {exc}") from exc
    if _sha(before) != cfg["sha256_before"]:
        raise DiffError(f"{cfg['path']} : l'original n'a pas le sha256 avant du détachement")
    if _sha(new_text.encode("utf-8")) != cfg["sha256_after"]:
        raise DiffError(f"{cfg['path']} : le rejeu ne redonne pas le sha256 après publié")
    target = out / embed.path.relative_to(original)
    append_tensor(target, LM_HEAD, embed.dtype, list(embed.shape), lm_head)
    write_config(config, new_text)


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
            if diff.get("detached_lm_head"):
                print("lm_head détaché (écart de protocole, rejoué par apply-diff)")
            print(f"Diff : {Path(where) / DIFF_FILE}")
            for f in relevant_files(diff["files_changed"]):
                print(f"Attention : {f['path']} diffère de la base (non rejoué par apply-diff)")
        else:
            meta = apply_diff(args.diff, args.original, args.out)
            print(f"Checkpoint rejoué : {args.out} ({len(meta['tensors'])} tenseurs vérifiés)")
            for f in relevant_files(meta["files_changed"]):
                print(f"Attention : {f['path']} différait de la base dans la séance (non rejoué)")
    except (HarnessError, DiffError, SafetensorsError, OSError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return getattr(exc, "exit_code", 1)
    return 0
