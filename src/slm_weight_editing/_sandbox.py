"""Amorce du lanceur `run-script` (story 5) : exécutée par `python -I -B -X utf8 _sandbox.py <cfg>`.

Avant le code de l'éditeur : numpy préchargé, `safetensors` remplacé par une version Python pure
(aucune entrée-sortie native hors de l'audit), chemins d'import hors des racines lisibles retirés,
puis un `sys.addaudithook` qui tue le processus (code `REFUSED`) au premier événement interdit :
ouverture hors des racines permises, écriture hors de `work/` et `scripts/`, import hors de la
bibliothèque standard, de numpy et des modules de `scripts/`, sous-processus, socket, ctypes, etc.

Le crochet n'utilise que des objets capturés à sa création (fonctions C de `nt`/`posix`, méthodes
de `str`) : remplacer `os.path`, `builtins` ou un module ne le trompe pas. Ce n'est pas un bac à
sable absolu (une extension C chargée avant lui ou un défaut de CPython y échappent) : c'est une
couche de plus, avec les réglages de Claude Code et l'audit de transcription.

Ce fichier ne dépend d'aucun autre module du paquet (il n'est pas importable par l'éditeur).
"""

import sys

REFUSED = 3

# Événements refusés d'office (préfixes) : processus, réseau, code natif, Windows, introspection.
DENY_PREFIXES = (
    "subprocess.",
    "_posixsubprocess.",
    "os.system",
    "os.exec",
    "os.spawn",
    "os.posix_spawn",
    "os.startfile",
    "os.fork",
    "os.kill",
    "os.add_dll_directory",
    "os.chdir",
    "os.fchdir",
    "os.symlink",
    "os.link",
    "pty.",
    "socket.",
    "ctypes.",
    "_winapi.",
    "_wmi.",
    "winreg.",
    "msvcrt.",
    "sqlite3.",
    "webbrowser.",
    "ftplib.",
    "http.",
    "urllib.",
    "smtplib.",
    "poplib.",
    "imaplib.",
    "nntplib.",
    "telnetlib.",
    "gc.get_",
    "sys._current_frames",
    "sys.remote_exec",
    "cpython.remote_debugger_script",
    "sys.addaudithook",
    "code.__new__",
)
# Modules jamais importables (nom de tête), même s'ils font partie de la bibliothèque standard.
DENY_TOP = frozenset(
    (
        "socket _socket ssl _ssl http urllib ftplib smtplib poplib imaplib nntplib telnetlib "
        "xmlrpc socketserver asyncio subprocess _posixsubprocess ctypes _ctypes multiprocessing "
        "_multiprocessing _posixshmem concurrent webbrowser _overlapped winreg _winreg _wmi "
        "sqlite3 _sqlite3 dbm _dbm _gdbm shelve tkinter _tkinter turtle turtledemo idlelib "
        "ensurepip venv _interpreters _interpqueues _interpchannels _xxsubinterpreters "
        "_xxinterpchannels _remote_debugging test safetensors slm_weight_editing torch "
        "transformers tokenizers llama_cpp cffi _cffi_backend requests httpx huggingface_hub "
        "hf_xet"
    ).split()
)
DENY_NUMPY = ("numpy.f2py", "numpy.distutils", "numpy.ctypeslib", "numpy.testing")
PROTECTED = frozenset(
    ("state.json", "journal.jsonl", "registry.jsonl", "session.lock", "workspace.json", "freeze")
)
# Règle et voie permise de chaque refus (story 2.5, CAP-6) : identifiants de RULES de l'audit
# (vérifié par test ; ce fichier n'importe rien du paquet).
WAYS = {
    "lecture-hors-espace": "tout l'espace de séance est lisible, rien au-delà",
    "ecriture-hors-liste": "un script écrit dans work/ ou scripts/ seulement",
    "etat-seance": "état par session-status ; journal et état en lecture seule",
    "import-interdit": "numpy, safetensors.numpy, bibliothèque standard ; modèle par run-batch "
    "ou probe",
    "script-refuse": "numpy, safetensors.numpy et bibliothèque standard ; ni processus ni réseau",
}
# Fichiers chargeables comme code natif ou bytecode : jamais écrits par un script.
NATIVE_SUFFIXES = (".pyd", ".so", ".dll", ".dylib", ".pyc", ".pyo", ".pth")

# Événements à chemins : (indices lus, indices écrits, indices dir_fd : None ou -1 par défaut).
PATH_EVENTS = {
    "os.listdir": ((0,), (), ()),
    "os.scandir": ((0,), (), ()),
    "os.remove": ((), (0,), (1,)),
    "os.rmdir": ((), (0,), (1,)),
    "os.mkdir": ((), (0,), (2,)),
    "os.truncate": ((), (0,), ()),
    "os.chmod": ((), (0,), (2,)),
    "os.chown": ((), (0,), (3,)),
    "os.utime": ((), (0,), (3,)),
    "os.chflags": ((), (0,), ()),
    "os.setxattr": ((), (0,), ()),
    "os.removexattr": ((), (0,), ()),
    "os.rename": ((), (0, 1), (2, 3)),
    "shutil.rmtree": ((), (0,), (1,)),
    "shutil.chown": ((), (0,), ()),
    "shutil.move": ((), (0, 1), ()),
    "shutil.copyfile": ((0,), (1,), ()),
    "shutil.copytree": ((0,), (1,), ()),
    "shutil.copymode": ((0,), (1,), ()),
    "shutil.copystat": ((0,), (1,), ()),
    "shutil.make_archive": ((2,), (0,), ()),
    "shutil.unpack_archive": ((0,), (1,), ()),
}


def make_hook(cfg, stdlib, local_modules, exec_roots):
    """Crochet d'audit : tout ce qu'il utilise est capturé ici, rien n'est relu en global."""
    import os
    import types

    win = sys.platform == "win32"
    _type, _str, _bytes, _int, _len = type, str, bytes, int, len
    _OSError, _TypeError, _ValueError = OSError, TypeError, ValueError
    _write, _exit, _fspath = os.write, os._exit, os.fspath
    _decode, _lower = bytes.decode, str.lower
    _code = type(make_hook.__code__)
    enc, errs = sys.getfilesystemencoding(), sys.getfilesystemencodeerrors()
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
    deny_prefixes, deny_top, deny_numpy = DENY_PREFIXES, DENY_TOP, DENY_NUMPY
    protected, refused, native, ways = PROTECTED, REFUSED, NATIVE_SUFFIXES, dict(WAYS)
    path_events = types.MappingProxyType(dict(PATH_EVENTS))
    if win:
        import nt

        _full, _final = nt._getfullpathname, nt._getfinalpathname
        sep = "\\"
    else:
        import posix

        _lstat, _readlink, _getcwd = posix.lstat, posix.readlink, posix.getcwd
        sep = "/"

    def canon(p):
        if _type(p) is _bytes:
            p = _decode(p, enc, errs)
        if win:
            full = _full(p)
            cur, tail = full, ""
            while True:
                try:
                    final = _final(cur)
                    break
                except _OSError:
                    i = cur.rfind("\\")
                    head = cur[:i] if i > 0 else cur
                    if head.endswith(":"):
                        head += "\\"
                    if i <= 0 or head == cur:
                        return _lower(full)
                    tail = "\\" + cur[i + 1 :] + tail
                    cur = head
            if final.startswith("\\\\?\\UNC\\"):
                final = "\\\\" + final[8:]
            elif final.startswith("\\\\?\\"):
                final = final[4:]
            return _lower(final.rstrip("\\") + tail)
        if not p.startswith("/"):
            p = _getcwd() + "/" + p
        parts = [x for x in p.split("/") if x and x != "."]
        resolved, i, links = "", 0, 0
        while i < _len(parts):
            part = parts[i]
            i += 1
            if part == "..":
                resolved = resolved[: resolved.rfind("/")] if resolved else ""
                continue
            candidate = resolved + "/" + part
            try:
                mode = _lstat(candidate).st_mode
            except _OSError:
                resolved = candidate
                continue
            if mode & 0o170000 == 0o120000:
                links += 1
                if links > 40:
                    return candidate
                target = _readlink(candidate)
                if target.startswith("/"):
                    resolved = ""
                parts = [x for x in target.split("/") if x and x != "."] + parts[i:]
                i = 0
                continue
            resolved = candidate
        return resolved or "/"

    def roots(paths):
        return tuple(canon(p) for p in paths)

    # Lecture libre de tout l'espace (CAP-5) et de l'interpréteur ; rien au-delà.
    read_roots = roots(cfg["read_roots"]) + roots(cfg["python_roots"])
    write_roots = roots(cfg["write_roots"])
    code_roots = roots(exec_roots)

    def under(path, bases):
        for base in bases:
            if path == base or path.startswith(base.rstrip(sep) + sep):
                return base
        return None

    def deny(event, detail, rule="script-refuse"):
        msg = (
            f"\nrun-script : refus ({event} : {detail}) ; règle {rule}, permis : {ways[rule]} ;"
            " arrêt du script\n"
        )
        try:
            _write(2, msg.encode("utf-8", "replace"))
        except _OSError:
            pass
        _exit(refused)

    def check_read(event, p):
        if _type(p) is _int:
            return
        c = canon(_fspath(p) if p is not None else ".")
        if under(c, read_roots) is None:
            deny(event, f"lecture hors des dossiers permis : {c}", "lecture-hors-espace")

    def check_write(event, p):
        if _type(p) is _int:
            return
        c = canon(_fspath(p))
        base = under(c, write_roots)
        if base is None:
            deny(event, f"écriture hors de work/ et scripts/ : {c}", "ecriture-hors-liste")
        if _lower(c).endswith(native) or "__pycache__" in c:
            deny(event, f"fichier de code natif ou compilé : {c}", "ecriture-hors-liste")
        rel = c[_len(base) :]
        if ":" in rel:
            # Flux NTFS (`state.json::$DATA`, `x:flux`) : cible réelle indéterminable ici.
            deny(event, f"flux de fichier interdit : {c}", "ecriture-hors-liste")
        for part in rel.split(sep):
            # Windows retire points et espaces finaux (`state.json.` est `state.json`).
            if _lower(part).rstrip(". ") in protected:
                deny(event, f"fichier de séance protégé : {c}", "etat-seance")

    def hook(event, args):
        if event.startswith(deny_prefixes):
            deny(event, "interdit dans run-script")
        if event == "import":
            name = args[0]
            if _type(name) is not _str:
                deny(event, "nom de module illisible", "import-interdit")
            top = name.partition(".")[0]
            if top in deny_top or top.startswith("_test"):
                deny(event, f"module {name} interdit", "import-interdit")
            if top == "numpy":
                if name.startswith(deny_numpy):
                    deny(event, f"module {name} interdit", "import-interdit")
                return
            if top not in stdlib and top not in local_modules:
                deny(
                    event,
                    f"module {name} hors de la bibliothèque standard, de numpy et scripts/",
                    "import-interdit",
                )
            return
        if event == "exec":
            # importlib.import_module ne lève pas d'événement import : le code d'un module se
            # contrôle à son exécution (bibliothèque standard, numpy et scripts/ seulement).
            # Un code compilé depuis une chaîne (`<string>`, amorce `<safetensors.numpy>`, modules
            # gelés `<frozen ...>`) passe : ses imports et ses ouvertures restent contrôlés un à un.
            obj = args[0]
            if _type(obj) is _code and obj.co_filename[:1] != "<":
                c = canon(obj.co_filename)
                if under(c, code_roots) is None:
                    detail = f"code hors de la bibliothèque standard, numpy et scripts/ : {c}"
                    deny(event, detail, "import-interdit")
            return
        if event == "open":
            path, mode, flags = args[0], args[1], args[2]
            writing = (_type(flags) is _int and flags & write_flags) or (
                _type(mode) is _str and ("w" in mode or "a" in mode or "x" in mode or "+" in mode)
            )
            try:
                if writing:
                    check_write(event, path)
                else:
                    check_read(event, path)
            except (_OSError, _TypeError, _ValueError) as exc:
                deny(event, f"chemin illisible ({exc})")
            return
        spec = path_events.get(event)
        if spec is None:
            return
        reads, writes, dirfds = spec
        try:
            for k in dirfds:
                if k < _len(args) and args[k] is not None and args[k] != -1:
                    deny(event, "dir_fd interdit")
            for k in reads:
                if k < _len(args):
                    check_read(event, args[k])
            for k in writes:
                if k < _len(args):
                    check_write(event, args[k])
        except (_OSError, _TypeError, _ValueError) as exc:
            deny(event, f"chemin illisible ({exc})")

    return hook, canon


_SHIM = '''
"""safetensors (numpy) en Python pur pour run-script : load, load_file, save, save_file.

BF16 (absent de numpy) se lit en uint16 (motifs bruts) ; save et save_file l'écrivent avec
`dtypes={"nom": "BF16"}` pour un tableau uint16.
"""
import json as _json

import numpy as _np

_CODES = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "BF16": "<u2", "I64": "<i8", "I32": "<i4",
          "I16": "<i2", "I8": "i1", "U64": "<u8", "U32": "<u4", "U16": "<u2", "U8": "u1",
          "BOOL": "?"}
_NAMES = {_np.dtype(v).str: k for k, v in _CODES.items() if k != "BF16"}


def load(data):
    data = bytes(data)
    n = int.from_bytes(data[:8], "little")
    header = _json.loads(data[8 : 8 + n])
    base, out = 8 + n, {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        arr = _np.frombuffer(data[base + start : base + end], dtype=_CODES[info["dtype"]])
        out[name] = arr.reshape(info["shape"]).copy()
    return out


def load_file(filename, device="cpu"):
    with open(filename, "rb") as f:
        return load(f.read())


def save(tensors, metadata=None, *, dtypes=None):
    dtypes = dtypes or {}
    header, chunks, offset = {}, [], 0
    for name in sorted(tensors):
        arr = _np.ascontiguousarray(tensors[name])
        arr = arr.astype(arr.dtype.newbyteorder("<"), copy=False)
        code = dtypes.get(name) or _NAMES.get(arr.dtype.str)
        if code is None or (code == "BF16" and arr.dtype.str != "<u2"):
            raise ValueError(f"{name} : dtype {arr.dtype} non pris en charge")
        raw = arr.tobytes()
        header[name] = {"dtype": code, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    if metadata:
        header["__metadata__"] = dict(metadata)
    text = _json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    return len(text).to_bytes(8, "little") + text + b"".join(chunks)


def save_file(tensors, filename, metadata=None, *, dtypes=None):
    data = save(tensors, metadata, dtypes=dtypes)
    with open(filename, "wb") as f:
        f.write(data)
'''


def _install_shim():
    import types

    package = types.ModuleType("safetensors")
    package.__path__ = []
    module = types.ModuleType("safetensors.numpy")
    exec(compile(_SHIM, "<safetensors.numpy>", "exec"), module.__dict__)
    package.numpy = module
    sys.modules["safetensors"] = package
    sys.modules["safetensors.numpy"] = module


def _main():
    import builtins
    import json
    import os
    import traceback
    import types

    cfg = json.loads(sys.argv[1])
    flags = sys.flags
    if not (flags.isolated and flags.ignore_environment and flags.dont_write_bytecode):
        os.write(2, "run-script : interpréteur non isolé (-I -B attendus)\n".encode())
        os._exit(REFUSED)
    script = cfg["script"]
    # La copie journalisée par le lanceur, relue et vérifiée : c'est elle qui s'exécute.
    with open(cfg["source"], "rb") as f:
        source = f.read()
    import hashlib

    if hashlib.sha256(source).hexdigest() != cfg["sha256"]:
        os.write(2, "run-script : copie du script altérée, refus\n".encode())
        os._exit(REFUSED)
    import numpy  # (préchargé : son import tire ctypes, interdit ensuite)

    _install_shim()
    scripts = cfg["write_roots"][1]
    local = frozenset(
        n[:-3] if n.endswith(".py") else n
        for n in os.listdir(scripts)
        if n.endswith(".py") or os.path.isfile(os.path.join(scripts, n, "__init__.py"))
    )
    exec_roots = (sys.base_prefix, sys.base_exec_prefix, os.path.dirname(numpy.__file__), scripts)
    hook, canon = make_hook(cfg, frozenset(sys.stdlib_module_names), local, exec_roots)
    # Chemins d'import hors des racines lisibles (src/ du dépôt installé en éditable) retirés.
    allowed = tuple(canon(p) for p in (*cfg["python_roots"], scripts))
    sep = "\\" if sys.platform == "win32" else "/"
    keep = []
    for entry in sys.path:
        c = canon(entry or ".")
        if any(c == a or c.startswith(a.rstrip(sep) + sep) for a in allowed):
            keep.append(entry)
    sys.path[:] = [scripts, *keep]
    main = types.ModuleType("__main__")
    main.__file__ = script
    main.__builtins__ = builtins
    code = compile(source, script, "exec")
    del source, f, keep, allowed, canon, exec_roots, numpy
    sys.argv = [script]
    sys.modules["__main__"] = main
    sys.addaudithook(hook)
    del hook
    try:
        exec(code, main.__dict__)
    except SystemExit:
        raise
    except BaseException as exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__.tb_next)
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    _main()
