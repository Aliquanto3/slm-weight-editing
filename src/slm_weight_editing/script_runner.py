"""Lanceur `run-script` : seule voie d'exécution de Python pour l'agent éditeur (story 5, CAP-6).

`uv run slm-weight-editing run-script scripts/x.py --session <id>
--sessions-root sessions`, depuis l'espace de séance. Le script (sous `scripts/` de l'espace) tourne
dans un sous-processus `python -I -B -X utf8`, environnement purgé, dossier courant = l'espace,
derrière le crochet d'audit de `_sandbox.py` (voir sa docstring et la fiche de la story 5).

Chaque exécution tient le verrou de séance (pas de script pendant un lot), refuse une séance
gelée, copie le script dans `<séance>/scripts/<sha256>.py` et ajoute au journal un événement
`script` (sha256, copie, code de retour, sortie tronquée) sans consommer de lot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

from slm_weight_editing import harness
from slm_weight_editing._sandbox import REFUSED

SCRIPTS_DIR = "scripts"
DATA_DIR = "data"
COPIES_DIR = "scripts"
MAX_OUTPUT = 20_000  # extrait de la sortie gardé dans le journal
PROBES_DIR = "probes"  # captures de `probe`, lisibles par les scripts, écrites par le harnais
RUNS_DIR = "runs"  # sortie complète de chaque exécution, lisible par l'éditeur
MAX_LOG_BYTES = 50 * 1024 * 1024  # au-delà, le fichier de sortie est coupé (disque protégé)
DEFAULT_THREADS = 4
DEFAULT_TIMEOUT = 900  # 15 min
DEFAULT_MAX_TIMEOUT = 3600  # 60 min
TIMEOUT_CEILING = 86_400  # 24 h : plafond de tout délai (Timer fini)
# Variables lues par les bibliothèques de calcul natives au chargement de numpy.
_THREAD_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_BOOT = Path(__file__).with_name("_sandbox.py")
# Seules variables transmises : de quoi démarrer l'interpréteur sous Windows.
_ENV_KEEP = ("SYSTEMROOT", "SystemRoot", "WINDIR")


def _python_roots() -> list[str]:
    names = ("prefix", "exec_prefix", "base_prefix", "base_exec_prefix")
    return sorted({str(Path(getattr(sys, n)).resolve()) for n in names})


def _inside(path: Path, base: Path) -> bool:
    return path == base or base in path.parents


def child_env(threads: int = DEFAULT_THREADS) -> dict[str, str]:
    """Environnement du sous-processus : rien de l'appelant (PYTHONPATH, PYTHONHOME...).

    Les variables de threads sont posées avant le démarrage de l'enfant, donc avant le
    préchargement de numpy par l'amorce ; un script les modifie sans effet après coup.
    """
    env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
    env.update({k: str(threads) for k in _THREAD_VARS})
    return env


def script_limits(info: dict | None) -> tuple[int, float, float]:
    """(threads, délai par défaut, délai maximum) de `workspace.json` ; défauts hors espace.

    Les valeurs relues sont validées comme à la création (fichier abîmé : refus net).
    """
    info = info or {}
    try:
        limits = (
            info.get("script_threads", DEFAULT_THREADS),
            float(info.get("script_timeout", DEFAULT_TIMEOUT)),
            float(info.get("script_max_timeout", DEFAULT_MAX_TIMEOUT)),
        )
    except (TypeError, ValueError) as exc:
        raise harness.HarnessError(
            f"workspace.json : réglages run-script illisibles ({exc})"
        ) from exc
    check_limits(*limits)
    return limits


def check_limits(threads: int, timeout: float, max_timeout: float) -> None:
    """Valide les trois réglages (création de l'espace, lecture) ; lève HarnessError.

    Délais finis et bornés à `TIMEOUT_CEILING` : au-delà, `threading.Timer` lève OverflowError
    dans son thread et le script tournerait sans délai.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise harness.HarnessError(f"--script-threads refusé : {threads} (entier, au moins 1)")
    for flag, value in (("--script-timeout", timeout), ("--script-max-timeout", max_timeout)):
        if not 0 < value <= TIMEOUT_CEILING:
            raise harness.HarnessError(
                f"{flag} refusé : {value:g} s (strictement positif, au plus {TIMEOUT_CEILING} s)"
            )
    if timeout > max_timeout:
        raise harness.HarnessError(
            f"--script-timeout refusé : {timeout:g} s dépasse le délai maximum {max_timeout:g} s"
        )


def _kill_tree(proc: subprocess.Popen) -> None:
    """Arrête le script et tous ses descendants."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        import signal

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    proc.kill()


def _resolve_script(workspace: Path, script: str) -> Path:
    scripts = (workspace / SCRIPTS_DIR).resolve()
    raw = Path(script)
    candidate = raw if raw.is_absolute() else workspace / raw
    path = candidate.resolve()
    if not _inside(path, scripts) or path == scripts:
        raise harness.HarnessError(f"script refusé : {script} n'est pas sous {SCRIPTS_DIR}/")
    if any(p.is_symlink() for p in (candidate, *candidate.parents) if _inside(p, workspace)):
        raise harness.HarnessError(f"script refusé : lien symbolique dans {script}")
    if path.suffix != ".py" or not path.is_file():
        raise harness.HarnessError(f"script refusé : {script} n'est pas un fichier .py")
    return path


def _bounded_read(stream, limit: int, log=None, log_limit: int = MAX_LOG_BYTES):
    """Lit tout le flux ; garde `limit` octets, écrit au plus `log_limit` octets dans `log`.

    Renvoie (extrait, total lu, octets écrits dans le fichier, sha256 de ces octets).
    """
    kept, total, written, digest = bytearray(), 0, 0, hashlib.sha256()
    while chunk := stream.read(65536):
        total += len(chunk)
        if len(kept) < limit:
            kept += chunk[: limit - len(kept)]
        if log is not None and written < log_limit:
            part = chunk[: log_limit - written]
            log.write(part)
            digest.update(part)
            written += len(part)
    return bytes(kept), total, written, digest.hexdigest()


def run_script(
    script: str,
    session_id: str,
    *,
    sessions_root: str | Path | None = None,
    workspace: str | Path | None = None,
    timeout: float | None = None,
    max_output: int = MAX_OUTPUT,
    max_log_bytes: int = MAX_LOG_BYTES,
) -> dict:
    """Exécute `scripts/<x>.py` sous le crochet d'audit ; renvoie l'événement `script`."""
    workspace = Path.cwd() if workspace is None else Path(workspace)
    workspace = workspace.resolve()
    path = _resolve_script(workspace, script)
    found = harness.find_workspace(workspace)
    if found is not None and found[0] != workspace:
        raise harness.HarnessError(f"run-script se lance depuis l'espace de séance {found[0]}")
    threads, default_timeout, max_timeout = script_limits(found[1] if found else None)
    if timeout is None:
        timeout = default_timeout
    if not timeout > 0:
        raise harness.HarnessError(f"--timeout refusé : {timeout:g} s (strictement positif)")
    if timeout > max_timeout:
        raise harness.HarnessError(
            f"--timeout refusé : {timeout:g} s dépasse le délai maximum de l'espace "
            f"({max_timeout:g} s) ; valeur permise : au plus {max_timeout:g} s"
        )
    root = harness.bind_workspace(
        session_id, sessions_root, cwd=workspace, default_root=harness.WORKSPACE_SESSIONS_DIR
    )
    directory = harness.session_dir(session_id, root).resolve()
    if not _inside(directory, workspace):
        raise harness.HarnessError(f"séance refusée : {directory} hors de l'espace de séance")
    work = directory / harness.WORK_DIR
    harness.read_state(directory)
    with harness._session_lock(directory):
        state = harness.read_state(directory)
        if state.frozen:
            raise harness.HarnessError("séance gelée : plus aucun script")
        # Lecture unique : la copie journalisée (vérifiée par son sha256) est ce qui s'exécute,
        # pas le fichier de scripts/, que l'éditeur pourrait changer pendant le lancement.
        source = path.read_bytes()
        sha256 = hashlib.sha256(source).hexdigest()
        copies = directory / COPIES_DIR
        copies.mkdir(exist_ok=True)
        copy = copies / f"{sha256}.py"
        if not copy.is_file() or hashlib.sha256(copy.read_bytes()).hexdigest() != sha256:
            copy.write_bytes(source)
        cfg = {
            "script": str(path),
            "source": str(copy),
            "sha256": sha256,
            "workspace": str(workspace),
            "read_roots": [
                str(work),
                str(workspace / SCRIPTS_DIR),
                str(workspace / DATA_DIR),
                str(workspace / PROBES_DIR),
            ],
            "read_files": [str(directory / harness.JOURNAL_FILE), str(directory / "state.json")],
            "write_roots": [str(work), str(workspace / SCRIPTS_DIR)],
            "python_roots": _python_roots(),
        }
        for r in cfg["python_roots"]:
            if _inside(workspace, Path(r)):
                raise harness.HarnessError(f"espace de séance sous l'interpréteur ({r}) refusé")
        started = harness._now()
        cmd = [sys.executable, "-I", "-B", "-X", "utf8", str(_BOOT), json.dumps(cfg)]
        # Fichier de sortie créé avant le lancement : une collision ou une erreur disque refuse
        # le script sans laisser d'enfant orphelin.
        runs = workspace / RUNS_DIR
        runs.mkdir(exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        log_path = runs / f"{stamp}-{path.stem}.log"
        with open(log_path, "xb") as log:
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=workspace,
                    env=child_env(threads),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=sys.platform != "win32",
                )
            except BaseException:
                log.close()
                log_path.unlink(missing_ok=True)  # pas de fichier vide sans entrée au journal
                raise
            killed = threading.Event()

            def _kill() -> None:
                killed.set()
                _kill_tree(proc)

            timer = threading.Timer(timeout, _kill)
            timer.start()
            try:
                output, total, written, log_sha256 = _bounded_read(
                    proc.stdout, max_output, log, max_log_bytes
                )
                returncode = proc.wait()
            except BaseException:
                # Ctrl-C ou erreur d'écriture : le délai serait annulé, l'arbre resterait orphelin.
                _kill_tree(proc)
                proc.wait()
                raise
            finally:
                timer.cancel()
                proc.stdout.close()
        timed_out = killed.is_set()
        text = output.decode("utf-8", "replace")
        entry = {
            "event": "script",
            "script": path.relative_to(workspace).as_posix(),
            "sha256": sha256,
            "copy": f"{COPIES_DIR}/{sha256}.py",
            "returncode": returncode,
            "refused": returncode == REFUSED,
            "timed_out": timed_out,
            "output": text,
            "output_bytes": total,
            "log": f"{RUNS_DIR}/{log_path.name}",
            "log_bytes": written,
            "log_sha256": log_sha256,
            "log_truncated": written < total,
            "timeout": timeout,
            "started_at": started,
            "finished_at": harness._now(),
        }
        harness._append_journal(directory, entry)
    return entry


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "run-script",
        help="exécute un script Python de l'éditeur sous le crochet d'audit (aucun lot consommé)",
        description=(
            "Lance scripts/<x>.py de l'espace de séance dans un python -I isolé : lecture de "
            "work/, scripts/ et data/, écriture de work/ et scripts/, numpy et la bibliothèque "
            "standard seulement ; ni modèle, ni réseau, ni sous-processus. Journalisé."
        ),
    )
    p.add_argument("script", help="chemin du script sous scripts/")
    p.add_argument("--session", required=True)
    p.add_argument(
        "--sessions-root", help="dossier des séances (celui de workspace.json dans un espace)"
    )
    p.add_argument(
        "--timeout",
        type=float,
        help="secondes ; défaut et maximum fixés à la création de l'espace (15 et 60 min)",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        entry = run_script(
            args.script, args.session, sessions_root=args.sessions_root, timeout=args.timeout
        )
    except harness.HarnessError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return exc.exit_code
    except OSError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(entry["output"])
    if entry["output_bytes"] > len(entry["output"].encode("utf-8")):
        print(
            f"\n[extrait : {entry['output_bytes']} octets au total ; "
            f"sortie complète dans {entry['log']}]"
        )
    if entry["log_truncated"]:
        print(f"[fichier coupé à {entry['log_bytes']} octets]", file=sys.stderr)
    if entry["timed_out"]:
        print(f"Erreur : script arrêté après {entry['timeout']:g} s", file=sys.stderr)
    status = "refusé" if entry["refused"] else f"code {entry['returncode']}"
    print(f"Script {entry['script']} ({entry['sha256'][:12]}) : {status}", file=sys.stderr)
    code = entry["returncode"]
    return code if code > 0 else (1 if code else 0)
