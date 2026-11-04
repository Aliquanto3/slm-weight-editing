"""Harnais d'édition : seule voie d'exécution du modèle pour l'agent éditeur (SPEC CAP-5).

Une séance vit dans `models/sessions/<id>/` :
- `state.json` : budget de lots (10, 25 ou 50, fixé à l'ouverture, immuable), lots consommés,
  gel (posé par la story 4), checkpoint et révision de base, dates ;
- `work/` : copie du checkpoint de base (safetensors, config, tokenizer) que l'agent modifie ;
- `journal.jsonl` : une ligne `open` à l'ouverture (avec le sha256 de la base), puis par lot une
  ligne `reserve` (lot décompté avant la génération) et une ligne `batch` (sorties).
Le registre `<sessions_root>/registry.jsonl` (ajout seul) garde chaque identifiant ouvert.
Un lot est consommé dès sa réservation, même si la génération plante (décompte conservateur).

Chaque lot recharge les poids depuis `work/`, génère en glouton (max_new_tokens fixé à 4096)
sous `torch.inference_mode()`, paramètres figés : ni gradient, ni perte, ni optimiseur.
Données : fichiers JSONL locaux des splits train de GSM8K et MATH, jamais sous `heldout/`.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

ALLOWED_BUDGETS = (10, 25, 50)
MAX_PROBLEMS = 512
MAX_NEW_TOKENS = 4096
DATASETS = ("gsm8k", "math")
TRAIN_SPLIT = "train"
FORBIDDEN_DIR = "heldout"
DEFAULT_SESSIONS_ROOT = "models/sessions"
STATE_FILE = "state.json"
JOURNAL_FILE = "journal.jsonl"
WORK_DIR = "work"
LOCK_FILE = "session.lock"
REGISTRY_FILE = "registry.jsonl"
_REPLACE_RETRIES = 20
_REPLACE_DELAY = 0.1
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# Champs de problème reconnus : `question` (GSM8K) ou `problem` (MATH).
_PROMPT_KEYS = ("question", "problem")


class HarnessError(RuntimeError):
    """Refus du harnais (protocole) ou séance illisible."""

    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


@dataclasses.dataclass(frozen=True)
class SessionState:
    session_id: str
    budget: int
    used: int
    frozen: bool
    base_checkpoint: str
    base_revision: str | None
    created_at: str
    updated_at: str

    @property
    def remaining(self) -> int:
        return self.budget - self.used


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def session_dir(session_id: str, sessions_root: str | Path = DEFAULT_SESSIONS_ROOT) -> Path:
    if not _SESSION_ID.match(session_id) or ".." in session_id:
        raise HarnessError(f"identifiant de séance invalide : {session_id!r}")
    return Path(sessions_root) / session_id


def _check_budget(budget: object) -> int:
    if type(budget) is not int or budget not in ALLOWED_BUDGETS:
        raise HarnessError(f"budget {budget!r} refusé : 10, 25 ou 50 lots seulement")
    return budget


def read_state(directory: str | Path) -> SessionState:
    """État validé d'une séance (réutilisé par le gel, story 4)."""
    path = Path(directory) / STATE_FILE
    try:
        raw = json.loads(path.read_text("utf-8"))
        state = SessionState(**{f.name: raw[f.name] for f in dataclasses.fields(SessionState)})
    except FileNotFoundError as exc:
        raise HarnessError(f"séance introuvable : {path}", 1) from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HarnessError(f"état de séance illisible : {path} ({exc})", 1) from exc
    _check_budget(state.budget)
    if type(state.used) is not int or not 0 <= state.used <= state.budget:
        raise HarnessError(f"compteur de lots invalide dans {path} : {state.used!r}", 1)
    if type(state.frozen) is not bool:
        raise HarnessError(f"champ frozen invalide dans {path}", 1)
    return state


def _write_state(directory: Path, state: SessionState) -> None:
    """Écriture atomique ; refuse tout changement de budget d'une séance existante."""
    path = directory / STATE_FILE
    if path.exists() and read_state(directory).budget != state.budget:
        raise HarnessError("le budget d'une séance ouverte ne se modifie pas")
    tmp = path.with_suffix(".tmp")
    text = json.dumps(dataclasses.asdict(state), indent=2, ensure_ascii=False)
    tmp.write_text(text, "utf-8", newline="\n")
    # Sous Windows, os.replace échoue (WinError 5) tant qu'un autre processus tient state.json.
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            if attempt == _REPLACE_RETRIES - 1:
                tmp.unlink(missing_ok=True)
                raise HarnessError(f"écriture de {path} impossible ({exc})", 1) from exc
            time.sleep(_REPLACE_DELAY)


@contextmanager
def _session_lock(directory: Path):
    """Verrou exclusif de séance : un lot ou un gel à la fois."""
    lock = directory / LOCK_FILE
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise HarnessError(f"un lot tourne déjà (sinon supprimer {lock})") from exc
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def update_state(directory: str | Path, **changes: object) -> SessionState:
    """Seul changement admis hors lot : poser `frozen` à vrai (gel, story 4)."""
    directory = Path(directory)
    if set(changes) - {"frozen"}:
        raise HarnessError(f"champs non modifiables : {sorted(set(changes) - {'frozen'})}")
    if "frozen" in changes and type(changes["frozen"]) is not bool:
        raise HarnessError("frozen doit être un booléen")
    read_state(directory)
    # Sous le verrou : un lot en cours ne peut ni annuler le gel ni démarrer après lui.
    with _session_lock(directory):
        state = read_state(directory)
        if changes.get("frozen") is False and state.frozen:
            raise HarnessError("une séance gelée ne se dégèle pas")
        new = dataclasses.replace(state, **changes, updated_at=_now())
        _write_state(directory, new)
        return new


def _append_journal(directory: Path, entry: dict) -> None:
    with (directory / JOURNAL_FILE).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_journal(directory: str | Path) -> list[dict]:
    path = Path(directory) / JOURNAL_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def used_batches(directory: str | Path, state: SessionState | None = None) -> int:
    """Lots consommés : `state.used`, recoupé avec les réservations du journal.

    Refuse la séance si `state.json` compte moins de lots que le journal (remise à zéro à la
    main) ou si le budget diffère de celui de la ligne `open`.
    """
    directory = Path(directory)
    state = read_state(directory) if state is None else state
    journal = read_journal(directory)
    opened = next((e for e in journal if e.get("event") == "open"), None)
    if opened is None or opened.get("budget") != state.budget:
        raise HarnessError("budget de state.json différent de celui de l'ouverture")
    reserved = sum(1 for e in journal if e.get("event") == "reserve")
    if state.used < reserved:
        raise HarnessError(
            f"compteur incohérent : state.json compte {state.used} lots, le journal {reserved}"
        )
    return max(state.used, reserved)


def _registered_ids(sessions_root: Path) -> set[str]:
    path = sessions_root / REGISTRY_FILE
    if not path.exists():
        return set()
    return {
        json.loads(line).get("session_id")
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    }


def _is_under_session(base: Path, sessions_root: Path) -> bool:
    """Vrai si `base` est sous le dossier des séances ou dans un dossier de séance quelconque."""
    roots = {sessions_root.resolve(), Path(DEFAULT_SESSIONS_ROOT).resolve()}
    if any(base == r or r in base.parents for r in roots):
        return True
    return any(
        (d / STATE_FILE).exists() or (d / JOURNAL_FILE).exists() for d in (base, *base.parents)
    )


def open_session(
    session_id: str,
    budget: int,
    base_checkpoint: str | Path,
    *,
    base_revision: str | None = None,
    sessions_root: str | Path = DEFAULT_SESSIONS_ROOT,
) -> SessionState:
    """Ouvre une séance : budget écrit une fois pour toutes, checkpoint copié dans `work/`."""
    _check_budget(budget)
    directory = session_dir(session_id, sessions_root)
    root = Path(sessions_root)
    if directory.exists() or session_id in _registered_ids(root):
        raise HarnessError(f"séance déjà ouverte : {directory} (budget non modifiable)")
    base = Path(base_checkpoint)
    if not base.is_dir() or not any(base.glob("*.safetensors")):
        raise HarnessError(f"checkpoint de base sans safetensors : {base}", 1)
    if _is_under_session(base.resolve(), root):
        raise HarnessError(f"checkpoint de base refusé : {base} est issu d'une séance")
    base_tensors = tensor_hashes(base)
    base_files = work_file_hashes(base)
    work = directory / WORK_DIR
    work.mkdir(parents=True)
    for file in sorted(base.iterdir()):
        if file.is_file():
            shutil.copy2(file, work / file.name)
    now = _now()
    state = SessionState(session_id, budget, 0, False, str(base.resolve()), base_revision, now, now)
    _write_state(directory, state)
    _append_journal(
        directory,
        {
            "event": "open",
            "session_id": session_id,
            "budget": budget,
            "base_checkpoint": state.base_checkpoint,
            "base_revision": base_revision,
            "base_tensors_sha256": base_tensors,
            "base_files_sha256": base_files,
            "at": now,
        },
    )
    with (root / REGISTRY_FILE).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"session_id": session_id, "opened_at": now}) + "\n")
    return state


def check_data_request(dataset: str, split: str, data_path: str | Path) -> Path:
    """Contrôles faits avant toute lecture du fichier de données."""
    if dataset not in DATASETS:
        raise HarnessError(f"jeu {dataset!r} refusé : gsm8k ou math seulement")
    if split != TRAIN_SPLIT:
        raise HarnessError(f"split {split!r} refusé : train seulement")
    raw = Path(data_path)
    candidates = (raw, Path(os.path.abspath(raw)), raw.resolve())
    for candidate in candidates:
        if any(part.lower() == FORBIDDEN_DIR for part in candidate.parts):
            raise HarnessError(f"chemin refusé : {data_path} est sous {FORBIDDEN_DIR}/")
    resolved = candidates[2]
    if TRAIN_SPLIT not in resolved.name.lower():
        raise HarnessError(f"fichier refusé : {resolved.name} ne porte pas « train » dans son nom")
    return resolved


def load_problems(path: Path, start: int = 0, count: int | None = None) -> list[dict]:
    """Problèmes `[start, start + count)` d'un JSONL ; au plus MAX_PROBLEMS."""
    if start < 0 or (count is not None and count < 1):
        raise HarnessError("start doit être positif et count au moins 1")
    try:
        lines = [ln for ln in path.read_text("utf-8").splitlines() if ln.strip()]
    except OSError as exc:
        raise HarnessError(f"données illisibles : {path} ({exc})", 1) from exc
    selected = lines[start:] if count is None else lines[start : start + count]
    if len(selected) > MAX_PROBLEMS:
        raise HarnessError(f"lot de {len(selected)} problèmes refusé : {MAX_PROBLEMS} au plus")
    if not selected:
        raise HarnessError("lot vide")
    problems = []
    for offset, line in enumerate(selected):
        index = start + offset
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise HarnessError(f"{path}:{index + 1} : JSON invalide", 1) from exc
        if not isinstance(record, dict):
            raise HarnessError(f"{path}:{index + 1} : objet JSON attendu", 1)
        if record.get("split") != TRAIN_SPLIT:
            raise HarnessError(f"{path}:{index + 1} : split {record.get('split')!r} refusé")
        prompt = next((record[k] for k in _PROMPT_KEYS if isinstance(record.get(k), str)), None)
        if prompt is None:
            raise HarnessError(f"{path}:{index + 1} : champ question ou problem absent", 1)
        problems.append({"index": index, "id": str(record.get("id", index)), "prompt": prompt})
    return problems


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hashes(work: Path) -> dict[str, str]:
    """sha256 des octets bruts de chaque tenseur safetensors de `work/`."""
    from slm_weight_editing.safetensors_file import open_checkpoint

    index = open_checkpoint(work)
    return {name: hashlib.sha256(index[name].read_bytes()).hexdigest() for name in sorted(index)}


def work_file_hashes(work: Path) -> dict[str, str]:
    """sha256 des autres fichiers du checkpoint (config, generation_config, tokenizer...)."""
    work = Path(work)
    return {
        f.relative_to(work).as_posix(): _sha256_file(f)
        for f in sorted(work.rglob("*"))
        if f.is_file() and f.suffix != ".safetensors"
    }


def _generation_settings(prompt_format: str, dtype: str) -> dict:
    return {
        "do_sample": False,
        "num_beams": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_format": prompt_format,
        "dtype": dtype,
        "inference_mode": True,
        "requires_grad": False,
    }


def _generate(work: Path, problems: list[dict]) -> tuple[dict, list[dict]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(work, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(work, local_files_only=True, dtype="auto")
    model.eval()
    model.requires_grad_(False)
    eos = model.config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else ({eos} if eos is not None else set())
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(eos_ids, default=0)
    model.generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=sorted(eos_ids) or None,
        pad_token_id=pad,
    )
    use_chat = bool(getattr(tokenizer, "chat_template", None))
    outputs = []
    with torch.inference_mode():
        for problem in problems:
            if use_chat:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": problem["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = problem["prompt"]
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=not use_chat)
            out = model.generate(**enc)
            new = out[0, enc["input_ids"].shape[1] :].tolist()
            reason = "stop" if new and new[-1] in eos_ids else "length"
            outputs.append(
                {
                    "id": problem["id"],
                    "text": tokenizer.decode(new, skip_special_tokens=True),
                    "new_tokens": len(new),
                    "done_reason": reason,
                }
            )
    settings = _generation_settings("chat_template" if use_chat else "raw", str(model.dtype))
    return settings, outputs


def run_batch(
    session_id: str,
    dataset: str,
    split: str,
    data_path: str | Path,
    *,
    start: int = 0,
    count: int | None = None,
    sessions_root: str | Path = DEFAULT_SESSIONS_ROOT,
) -> dict:
    """Lance un lot : contrôles, rechargement de `work/`, génération, journal, décompte."""
    resolved = check_data_request(dataset, split, data_path)
    directory = session_dir(session_id, sessions_root)
    read_state(directory)
    with _session_lock(directory):
        state = read_state(directory)
        if state.frozen:
            raise HarnessError("séance gelée : plus aucun lot")
        used = used_batches(directory, state)
        if used >= state.budget:
            raise HarnessError(f"budget épuisé : {used}/{state.budget} lots")
        problems = load_problems(resolved, start, count)
        work = directory / WORK_DIR
        hashes = tensor_hashes(work)
        files = work_file_hashes(work)
        data_sha256 = _sha256_file(resolved)
        number = used + 1
        # Réservation avant la génération : un échec d'écriture ne génère rien, un plantage
        # pendant la génération consomme le lot.
        _write_state(directory, dataclasses.replace(state, used=number, updated_at=_now()))
        started = _now()
        _append_journal(
            directory,
            {"event": "reserve", "batch": number, "budget": state.budget, "at": started},
        )
        settings, outputs = _generate(work, problems)
        entry = {
            "event": "batch",
            "batch": number,
            "budget": state.budget,
            "remaining": state.budget - number,
            "settings": settings,
            "dataset": dataset,
            "split": split,
            "data_path": str(resolved),
            "data_sha256": data_sha256,
            "problems": problems,
            "outputs": outputs,
            "tensors_sha256": hashes,
            "files_sha256": files,
            "started_at": started,
            "finished_at": _now(),
        }
        _append_journal(directory, entry)
        return entry


def add_parser(sub: argparse._SubParsersAction) -> None:
    root_help = f"dossier des séances (défaut : {DEFAULT_SESSIONS_ROOT})"
    op = sub.add_parser(
        "session-open",
        help="ouvre une séance d'édition à budget de lots fixé",
        description="Copie le checkpoint de base dans work/ et fixe le budget (10, 25 ou 50).",
    )
    op.add_argument("--session", required=True, help="identifiant de la séance")
    op.add_argument("--budget", required=True, type=int, help="10, 25 ou 50 lots")
    op.add_argument("--base-checkpoint", required=True, help="dossier safetensors HF de base")
    op.add_argument("--base-revision", help="révision HF du checkpoint de base")
    op.add_argument("--sessions-root", default=DEFAULT_SESSIONS_ROOT, help=root_help)
    rb = sub.add_parser(
        "run-batch",
        help="lance un lot d'inférence sur un split train (compté contre le budget)",
        description=(
            f"Génération gloutonne, max_new_tokens {MAX_NEW_TOKENS}, au plus {MAX_PROBLEMS} "
            "problèmes, poids rechargés depuis work/ ; une ligne par lot dans journal.jsonl."
        ),
    )
    rb.add_argument("--session", required=True)
    rb.add_argument("--dataset", required=True, help="gsm8k ou math")
    rb.add_argument("--split", required=True, help="train seulement")
    rb.add_argument("--data", required=True, help="fichier JSONL local du split train")
    rb.add_argument("--start", type=int, default=0, help="premier problème (défaut 0)")
    rb.add_argument("--count", type=int, help="nombre de problèmes (défaut : jusqu'à la fin)")
    rb.add_argument("--sessions-root", default=DEFAULT_SESSIONS_ROOT, help=root_help)
    st = sub.add_parser("session-status", help="budget, lots consommés et gel d'une séance")
    st.add_argument("--session", required=True)
    st.add_argument("--sessions-root", default=DEFAULT_SESSIONS_ROOT, help=root_help)
    for p in (op, rb, st):
        p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        if args.command == "session-open":
            state = open_session(
                args.session,
                args.budget,
                args.base_checkpoint,
                base_revision=args.base_revision,
                sessions_root=args.sessions_root,
            )
            print(f"Séance {state.session_id} ouverte : budget {state.budget} lots")
        elif args.command == "run-batch":
            entry = run_batch(
                args.session,
                args.dataset,
                args.split,
                args.data,
                start=args.start,
                count=args.count,
                sessions_root=args.sessions_root,
            )
            print(
                f"Lot {entry['batch']}/{entry['budget']} : {len(entry['problems'])} problèmes, "
                f"reste {entry['remaining']} lots"
            )
        else:
            directory = session_dir(args.session, args.sessions_root)
            state = read_state(directory)
            used = used_batches(directory, state)
            print(
                f"Séance {state.session_id} : {used}/{state.budget} lots, "
                f"reste {state.budget - used}, gelée : {'oui' if state.frozen else 'non'}"
            )
    except HarnessError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return exc.exit_code
    except OSError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    return 0
