"""Ajoute un `output.weight` Q8_0, copié de `token_embd.weight`, à un GGUF qui n'en a pas.

Préalable de CAP-10 : vérifier qu'Ollama charge un `output.weight` séparé pour qwen3.5. Le
GGUF produit est le GGUF d'entrée, octet pour octet (métadonnées, informations de tenseurs,
alignement, données), plus une information de tenseur et ses données en fin de fichier. Seuls
le nombre de tenseurs et le décalage des données changent dans l'en-tête ; les décalages
relatifs des tenseurs existants restent valides.

Aucune métadonnée de liaison des embeddings n'est modifiée : llama.cpp déduit la liaison de
l'absence de `output.weight`. Les clés dont le nom contient « tie » sont seulement signalées.
"""

from __future__ import annotations

import argparse
import mmap
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

from slm_weight_editing.gguf_file import GGUFError, _Cursor, read_gguf
from slm_weight_editing.ollama_store import OllamaStoreError, default_store, load_model
from slm_weight_editing.q8_0 import QK8_0, bf16_bytes_to_f32, quantize_q8_0
from slm_weight_editing.report_gguf import ReportError, render_modelfile

SOURCE, TARGET = "token_embd.weight", "output.weight"
GGML_Q8_0 = 8
_COPY_CHUNK = 16 * 1024 * 1024


class AddOutputError(ValueError):
    """Entrée ou sortie refusée."""


def _to_float32(raw: bytes, type_name: str) -> np.ndarray:
    if type_name == "F32":
        return np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if type_name == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if type_name == "BF16":
        return bf16_bytes_to_f32(raw)
    raise AddOutputError(f"type {type_name} de {SOURCE} non pris en charge")


def _header_ends(path: Path) -> tuple[int, int]:
    """Positions (fin des métadonnées, fin des informations de tenseurs) dans le fichier."""
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        cur = _Cursor(buf)
        cur.take(8)
        n_tensors, n_kv = cur.unpack("Q"), cur.unpack("Q")
        for _ in range(n_kv):
            cur.string()
            cur.value(cur.unpack("I"))
        kv_end = cur.pos
        for _ in range(n_tensors):
            cur.string()
            for _ in range(cur.unpack("I")):
                cur.unpack("Q")
            cur.unpack("I")
            cur.unpack("Q")
        return kv_end, cur.pos


def _same_file(a: Path, b: Path) -> bool:
    return a.resolve() == b.resolve() or (b.exists() and a.exists() and a.samefile(b))


def add_output_weight(src: str | Path, dst: str | Path, store: str | Path | None = None) -> dict:
    """Écrit dans `dst` une copie de `src` avec `output.weight` ; renvoie un résumé."""
    src, dst = Path(src), Path(dst)
    if _same_file(src, dst):
        raise AddOutputError("la sortie ne doit pas être le fichier d'entrée")
    if dst.resolve().is_relative_to(default_store(store).resolve()):
        raise AddOutputError(f"la sortie ne doit pas être dans le magasin d'Ollama : {dst}")
    if dst.exists():
        raise AddOutputError(f"la sortie existe déjà : {dst}")
    gguf = read_gguf(src)
    if TARGET in gguf.tensors:
        raise AddOutputError(f"{src} a déjà un {TARGET} : refus")
    embd = gguf.tensor(SOURCE)
    if len(embd.dims) != 2:
        raise AddOutputError(f"{SOURCE} : 2 dimensions attendues, {len(embd.dims)} trouvées")
    if embd.dims[0] % QK8_0:
        raise AddOutputError(f"{SOURCE} : ne0={embd.dims[0]} non multiple de {QK8_0}")
    raw = gguf.read_tensor_bytes(SOURCE)
    if embd.type_name == "Q8_0":
        data, mode = raw, "copie d'octets"
    else:
        data, mode = quantize_q8_0(_to_float32(raw, embd.type_name)), "requantifié"
    name = TARGET.encode("utf-8")
    info = struct.pack("<Q", len(name)) + name + struct.pack("<I", len(embd.dims))
    info += b"".join(struct.pack("<Q", d) for d in embd.dims) + struct.pack("<I", GGML_Q8_0)

    kv_end, info_end = _header_ends(src)
    size = src.stat().st_size
    align = gguf.alignment
    rel_new = -(-(size - gguf.data_offset) // align) * align
    info += struct.pack("<Q", rel_new)
    new_data_offset = -(-(info_end + len(info)) // align) * align
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with src.open("rb") as fin, dst.open("wb") as out:
            head = bytearray(fin.read(info_end))
            struct.pack_into("<Q", head, 8, len(gguf.tensors) + 1)
            out.write(head)
            out.write(info)
            out.write(bytes(new_data_offset - info_end - len(info)))
            fin.seek(gguf.data_offset)
            shutil.copyfileobj(fin, out, _COPY_CHUNK)
            out.write(bytes(new_data_offset + rel_new - out.tell()))
            out.write(data)
    except BaseException:
        dst.unlink(missing_ok=True)
        raise
    return {
        "output": str(dst),
        "mode": mode,
        "source_type": embd.type_name,
        "dims": list(embd.dims),
        "tensors_before": len(gguf.tensors),
        "added_bytes": dst.stat().st_size - size,
        "tie_keys": sorted(k for k in gguf.metadata if "tie" in k.lower()),
    }


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "gguf-add-output",
        help="ajoute un output.weight Q8_0 (copie de token_embd) à un GGUF (essai CAP-10)",
        description=(
            "Écrit un nouveau GGUF identique à l'entrée plus output.weight en Q8_0 copié de "
            "token_embd.weight. Refuse un GGUF qui a déjà un output.weight, et toute sortie "
            "qui écraserait l'entrée, un fichier existant ou le magasin d'Ollama."
        ),
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--gguf", help="chemin du GGUF d'entrée")
    src.add_argument("--model", help="modèle Ollama dont on lit le blob GGUF (ex. qwen3.5:0.8b)")
    p.add_argument(
        "--ollama-models", help="magasin d'Ollama (sinon OLLAMA_MODELS, sinon ~/.ollama)"
    )
    p.add_argument("--out", help="GGUF de sortie (défaut : models/<nom>-output.gguf)")
    p.set_defaults(func=run)


def modelfile_paths(dst: str | Path) -> tuple[Path, Path]:
    """Modelfile de l'essai et Modelfile témoin (GGUF officiel), à côté de la sortie."""
    dst = Path(dst)
    return dst.parent / f"{dst.stem}.Modelfile", dst.parent / f"{dst.stem}-temoin.Modelfile"


def run(args: argparse.Namespace) -> int:
    modelfiles: list[Path] = []
    try:
        official = None
        if args.model:
            official = load_model(args.model, args.ollama_models)
            src = official.model_blob
            default = "models/" + args.model.replace(":", "-").replace("/", "-")
        else:
            src = Path(args.gguf)
            default = "models/" + src.stem
        dst = Path(args.out or default + "-output.gguf")
        if official is not None:
            test_mf, control_mf = modelfile_paths(dst)
            for path in (test_mf, control_mf):
                if path.exists():
                    raise AddOutputError(f"le Modelfile existe déjà : {path}")
            # Rendus avant l'écriture du GGUF : un gabarit non transcriptible n'écrit rien.
            texts = [render_modelfile(dst, official), render_modelfile(src, official)]
        res = add_output_weight(src, dst, args.ollama_models)
        if official is not None:
            for path, text in zip((test_mf, control_mf), texts, strict=True):
                path.write_text(text, "utf-8")
                modelfiles.append(path)
    except (OSError, ValueError, GGUFError, OllamaStoreError, ReportError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
    print(f"{res['output']} écrit : {TARGET} {res['dims']} Q8_0 ({res['mode']}), ", end="")
    print(f"+{res['added_bytes']} octets, source {res['source_type']}")
    if res["tie_keys"]:
        print(f"Attention : métadonnées de liaison non modifiées : {', '.join(res['tie_keys'])}")
    for path in modelfiles:
        print(f"{path} écrit")
    return 0
