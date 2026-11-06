"""Détachement de `lm_head` des embeddings liés, dans `work/` d'une séance (SPEC CAP-10).

`detach-lm-head` copie `embed_tokens` en `lm_head.weight` (octet pour octet) dans le fichier
safetensors de `work/` qui porte `embed_tokens`, et passe `tie_word_embeddings` à false dans
`config.json` (racine et `text_config`). Le modèle répond comme avant (même copie en entrée et en
sortie), mais l'éditeur peut ensuite éditer la sortie d'un jeton sans toucher son embedding
d'entrée. Une seule fois par séance, sous le verrou, avec une ligne `detach_lm_head` au journal ;
ni lot ni jeton de probe consommés ; refusé après le gel.

Le gel et `apply-diff` rejouent cette transformation (`freeze.py`) : seul un détachement journalisé
par cette commande est reconnu, un `lm_head.weight` ou un `config.json` écrits à la main ne le
sont pas. Ce module ne dépend pas de `freeze` (qui l'importe).

Cette commande ne sert en s2 que si l'essai Ollama de la story 2.7 conclut que CAP-10 est maintenu
(`docs/essai-output-weight.md`).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

from slm_weight_editing import harness
from slm_weight_editing.harness import HarnessError
from slm_weight_editing.safetensors_file import SafetensorsError, open_checkpoint

EVENT = "detach_lm_head"
LM_HEAD = "lm_head.weight"
CONFIG_FILE = "config.json"
TIE_KEY = "tie_word_embeddings"
_INDEX_FILE = "model.safetensors.index.json"
_COPY_CHUNK = 16 * 1024 * 1024


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detach_event(journal: list[dict]) -> dict | None:
    """Ligne `detach_lm_head` du journal, ou None."""
    return next((e for e in journal if e.get("event") == EVENT), None)


def is_lm_head(name: str) -> bool:
    return name == LM_HEAD or name.endswith("." + LM_HEAD)


def find_embed(names) -> str:
    """Nom de l'unique tenseur `embed_tokens.weight`."""
    found = sorted(
        n for n in names if n == "embed_tokens.weight" or n.endswith(".embed_tokens.weight")
    )
    if len(found) != 1:
        raise HarnessError(f"embed_tokens.weight attendu une fois dans work/, trouvé : {found}")
    return found[0]


def _sections(data: dict) -> list[dict]:
    text = data.get("text_config")
    return [data] + ([text] if isinstance(text, dict) else [])


def config_tie_values(text: str) -> list[object]:
    """Valeurs de `tie_word_embeddings` déclarées (racine, puis `text_config`)."""
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise HarnessError(f"{CONFIG_FILE} illisible : {exc}") from exc
    if not isinstance(data, dict):
        raise HarnessError(f"{CONFIG_FILE} : objet JSON attendu")
    return [s[TIE_KEY] for s in _sections(data) if TIE_KEY in s]


def transform_config(text: str) -> tuple[str, list[str]]:
    """`config.json` délié (texte déterministe) et clés écrites.

    Le résultat ne dépend que du contenu JSON d'entrée : le gel et apply-diff le recalculent
    pour vérifier le sha256 publié.
    """
    config_tie_values(text)
    data = json.loads(text)
    data[TIE_KEY] = False
    keys = [TIE_KEY]
    if isinstance(data.get("text_config"), dict):
        data["text_config"][TIE_KEY] = False
        keys.append(f"text_config.{TIE_KEY}")
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n", keys


def append_tensor(
    path: Path,
    name: str,
    dtype: str,
    shape: list[int],
    data: bytes,
    *,
    backup: Path | None = None,
) -> None:
    """Réécrit un fichier safetensors avec un tenseur de plus en fin de données (atomique).

    `backup` : l'ancien fichier y est déplacé (renommage) juste avant le remplacement, pour
    pouvoir annuler l'opération.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        with path.open("rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
            data_start = 8 + n
            size = f.seek(0, os.SEEK_END) - data_start
            if name in header:
                raise HarnessError(f"{name} existe déjà dans {path.name}")
            header[name] = {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [size, size + len(data)],
            }
            raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
            raw += b" " * (-len(raw) % 8)
            with tmp.open("wb") as out:
                out.write(struct.pack("<Q", len(raw)) + raw)
                f.seek(data_start)
                shutil.copyfileobj(f, out, _COPY_CHUNK)
                out.write(data)
        if backup is not None:
            os.replace(path, backup)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_config(path: Path, text: str) -> None:
    write_config_bytes(path, text.encode("utf-8"))


def write_config_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def detach_lm_head(
    session_id: str, *, sessions_root: str | Path = harness.DEFAULT_SESSIONS_ROOT
) -> dict:
    """Détache `lm_head` dans `work/` et journalise ; renvoie la ligne de journal."""
    directory = harness.session_dir(session_id, sessions_root)
    harness.read_state(directory)
    with harness._session_lock(directory):
        if harness.read_state(directory).frozen:
            raise HarnessError("séance gelée : détachement de lm_head refusé")
        if detach_event(harness.read_journal(directory)) is not None:
            raise HarnessError("lm_head déjà détaché dans cette séance : refus")
        work = directory / harness.WORK_DIR
        try:
            index = open_checkpoint(work)
        except SafetensorsError as exc:
            raise HarnessError(f"work/ illisible : {exc}") from exc
        heads = [n for n in index if is_lm_head(n)]
        if heads:
            raise HarnessError(f"checkpoint déjà délié : {heads[0]} présent dans work/")
        if (work / _INDEX_FILE).exists():
            raise HarnessError(f"{_INDEX_FILE} présent : checkpoint multi-fichiers non géré")
        embed_name = find_embed(index)
        config = work / CONFIG_FILE
        if not config.is_file():
            raise HarnessError(f"{CONFIG_FILE} absent de work/ : détachement impossible")
        config_bytes = config.read_bytes()
        ties = config_tie_values(config_bytes.decode("utf-8"))
        if any(v is False for v in ties):
            raise HarnessError(f"checkpoint déjà délié : {CONFIG_FILE} déclare {TIE_KEY}: false")
        if not any(v is True for v in ties):
            raise HarnessError(
                f"embeddings non liés : {CONFIG_FILE} ne déclare pas {TIE_KEY}: true"
            )
        journal = harness.read_journal(directory)
        opened = next((e for e in journal if e.get("event") == "open"), {})
        base_sha = opened.get("base_files_sha256", {}).get(CONFIG_FILE)
        if base_sha is not None and base_sha != _sha(config_bytes):
            # Le gel ne pourrait pas rejouer la transformation sur la config de la base.
            raise HarnessError(
                f"{CONFIG_FILE} de work/ modifié depuis l'ouverture : détachement refusé"
            )
        embed = index[embed_name]
        raw = embed.read_bytes()
        file_before = harness._sha256_file(embed.path)
        new_text, keys = transform_config(config_bytes.decode("utf-8"))
        backup = embed.path.with_name(embed.path.name + ".detach-bak")
        # Tenseur, config et journal : une erreur entre deux écritures remet work/ en l'état,
        # sinon la séance resterait déliée sans ligne au journal (refusée au gel et au détachement).
        try:
            append_tensor(embed.path, LM_HEAD, embed.dtype, list(embed.shape), raw, backup=backup)
            write_config(config, new_text)
            entry = {
                "event": EVENT,
                "at": harness._now(),
                "source": embed_name,
                "lm_head": LM_HEAD,
                "shape": list(embed.shape),
                "dtype": embed.dtype,
                "embed_sha256": _sha(raw),
                "lm_head_sha256": _sha(open_checkpoint(work)[LM_HEAD].read_bytes()),
                "file": embed.path.name,
                "file_sha256_before": file_before,
                "file_sha256_after": harness._sha256_file(embed.path),
                "config": {
                    "path": CONFIG_FILE,
                    "sha256_before": _sha(config_bytes),
                    "sha256_after": _sha(new_text.encode("utf-8")),
                    "keys": keys,
                },
            }
            harness._append_journal(directory, entry)
        except BaseException:
            if backup.exists():
                os.replace(backup, embed.path)
            if config.read_bytes() != config_bytes:
                write_config_bytes(config, config_bytes)
            raise
        backup.unlink(missing_ok=True)
        return entry


def run_detach(args) -> int:
    try:
        root = harness.bind_workspace(args.session, args.sessions_root)
        entry = detach_lm_head(args.session, sessions_root=root)
    except HarnessError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return exc.exit_code
    except OSError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    print(
        f"lm_head détaché de {entry['source']} : {entry['lm_head']} {entry['shape']} "
        f"{entry['dtype']} ajouté à work/, {TIE_KEY}: false ({', '.join(entry['config']['keys'])})"
    )
    return 0
