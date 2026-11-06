"""Report d'une édition HF dans le GGUF Ollama officiel (SPEC CAP-1, CAP-2, CAP-3).

Étapes, dans cet ordre ; aucune sortie GGUF tant qu'une vérification échoue :
1. sha256 des octets bf16 bruts des tenseurs édités contre les valeurs publiées ;
2. contrôle de fidélité : requantifier des tenseurs originaux (les tenseurs de même nature non
   édités du checkpoint, et les originaux des tenseurs visés si un checkpoint de base est donné)
   doit redonner exactement leurs octets dans le GGUF officiel ;
3. copie du GGUF officiel, puis réécriture des seules plages d'octets des tenseurs visés ;
4. Modelfile repris du manifeste officiel et diff vérifiable (JSON).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from slm_weight_editing.gguf_file import GGUFFile, TensorInfo, read_gguf
from slm_weight_editing.lm_head import find_embed
from slm_weight_editing.ollama_store import OllamaModel, load_model
from slm_weight_editing.q8_0 import (
    BLOCK_BYTES,
    QK8_0,
    bf16_bytes_to_f32,
    quantize_q8_0,
    split_blocks,
)
from slm_weight_editing.safetensors_file import SafeTensor, open_checkpoint

# Édition publiée du MIT (reference-mit.md) : base, dépôt et sha256 des octets bf16 bruts.
WSP_SOURCE_MODEL = "qwen3.5:0.8b"
WSP_TAG = "qwen3.5-wsp:0.8b"
WSP_BASE_REPO = "Qwen/Qwen3.5-0.8B"
WSP_BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
WSP_EDIT_REPO = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
WSP_EDITED_SHA256 = {
    "model.language_model.layers.15.mlp.down_proj.weight": (
        "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53"
    ),
    "model.language_model.layers.19.mlp.down_proj.weight": (
        "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2"
    ),
}

DIFF_FORMAT = "slm-weight-editing/diff-v1"
SUPPORTED_GGML_TYPES = ("Q8_0",)

# Correspondances vérifiées sur le GGUF officiel de qwen3.5:0.8b ; les autres noms sont refusés.
# blk.N.ffn_down.weight : Q8_0, dims ggml (3584, 1024) pour une forme HF (1024, 3584).
# token_embd.weight : Q8_0, dims ggml (1024, 248320) pour HF (248320, 1024) ; embeddings liés
# (pas de output.weight). Les blocs de 32 courent le long de la dimension contiguë (une ligne HF).
# Troisième champ : tenseur témoin quand il n'est pas visé. token_embd ne l'est pas (jamais
# vérifié reproductible octet pour octet, ~1 Go en f32) : visé, son original est son seul témoin.
_HF_TO_GGUF = (
    (
        re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.mlp\.down_proj\.weight$"),
        "blk.{}.ffn_down.weight",
        True,
    ),
    (
        re.compile(r"^model\.(?:language_model\.)?embed_tokens\.weight$"),
        "token_embd.weight",
        False,
    ),
    # lm_head détaché (CAP-10) : le GGUF officiel n'a pas d'output.weight, il est ajouté en copie
    # Q8_0 de token_embd (gguf_add_output). Accepté seulement si le diff porte le détachement.
    (re.compile(r"^lm_head\.weight$"), "output.weight", False),
)
TIED_EMBEDDING = "token_embd.weight"
OUTPUT_WEIGHT = "output.weight"
PROTOCOL_GAP = (
    "lm_head détaché des embeddings : écart de protocole avec l'édition du MIT (embeddings liés)"
)
# Blocs Q8_0 quantifiés par tranche : pic mémoire borné (token_embd : 254 M valeurs).
_QUANT_CHUNK_BLOCKS = 1 << 18


class ReportError(Exception):
    """Report interrompu ; `exit_code` est le code de sortie de la commande."""

    exit_code = 1


class ShaMismatchError(ReportError):
    exit_code = 3


class FidelityError(ReportError):
    exit_code = 2

    def __init__(self, message: str, report: dict) -> None:
        super().__init__(message)
        self.report = report


def gguf_name(hf_name: str) -> str | None:
    """Nom GGUF d'un tenseur HF, ou None s'il n'a pas de correspondance vérifiée."""
    for pattern, template, _ in _HF_TO_GGUF:
        m = pattern.match(hf_name)
        if m:
            return template.format(*m.groups())
    return None


def is_witness(hf_name: str) -> bool:
    """Tenseur non visé qui sert de témoin de fidélité (down_proj seulement, comme au lot 0)."""
    return any(p.match(hf_name) and witness for p, _, witness in _HF_TO_GGUF)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def _check_pair(st: SafeTensor, info: TensorInfo) -> None:
    if st.dtype != "BF16":
        raise ReportError(f"{st.name} : dtype {st.dtype}, BF16 attendu")
    if info.type_name not in SUPPORTED_GGML_TYPES:
        raise ReportError(f"{info.name} : type {info.type_name} non pris en charge (Q8_0 seul)")
    # ggml range ses dimensions de la plus contiguë à la moins contiguë : l'inverse de HF.
    if tuple(reversed(info.dims)) != st.shape:
        raise ReportError(
            f"{st.name} {list(st.shape)} et {info.name} {list(info.dims)} (ordre ggml) : "
            "formes incompatibles sans transposition"
        )
    if st.end - st.start != 2 * math.prod(st.shape):
        raise ReportError(f"{st.name} : {st.end - st.start} octets pour la forme {list(st.shape)}")


def quantize_tensor(st: SafeTensor, intermediate: str) -> bytes:
    """Q8_0 des octets bf16, par tranches de blocs (chaque bloc se quantifie seul)."""
    raw = st.read_bytes()
    step = 2 * QK8_0 * _QUANT_CHUNK_BLOCKS
    return b"".join(
        quantize_q8_0(bf16_bytes_to_f32(raw[i : i + step]), intermediate)
        for i in range(0, len(raw), step)
    )


def _check_tied(checkpoint: str | Path, edited: dict[str, SafeTensor], gguf: GGUFFile) -> None:
    """token_embd sert aussi de sortie (pas de output.weight) : le checkpoint doit être lié."""
    if "output.weight" in gguf.tensors:
        raise ReportError(f"{TIED_EMBEDDING} visé mais le GGUF a un output.weight distinct")
    heads = [n for n in edited if n == "lm_head.weight" or n.endswith(".lm_head.weight")]
    if heads:
        raise ReportError(f"embed_tokens visé mais le checkpoint a un {heads[0]} distinct")
    config = Path(checkpoint) / "config.json"
    if not config.is_file():
        return
    try:
        data = json.loads(config.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise ReportError(f"{config} illisible : {exc}") from exc
    sections = [data, data.get("text_config")] if isinstance(data, dict) else []
    if any(isinstance(c, dict) and c.get("tie_word_embeddings") is False for c in sections):
        raise ReportError(f"embed_tokens visé mais {config} déclare tie_word_embeddings: false")


def _row_bytes(st: SafeTensor) -> int:
    """Octets Q8_0 d'une ligne HF (les blocs de 32 courent le long de la ligne)."""
    return st.shape[1] // QK8_0 * BLOCK_BYTES


def _lm_head_rows(lm_head: SafeTensor, embed: SafeTensor, intermediate: str) -> list:
    """Lignes de `lm_head` qui diffèrent de l'`embed_tokens` du checkpoint : (indice, Q8_0)."""
    if len(lm_head.shape) != 2 or lm_head.shape[1] % QK8_0:
        raise ReportError(f"{lm_head.name} : forme {list(lm_head.shape)} non quantifiable en Q8_0")
    rows, cols = lm_head.shape
    mine = np.frombuffer(lm_head.read_bytes(), np.uint8).reshape(rows, 2 * cols)
    theirs = np.frombuffer(embed.read_bytes(), np.uint8).reshape(rows, 2 * cols)
    changed = np.flatnonzero((mine != theirs).any(axis=1)).tolist()
    return [(r, quantize_q8_0(bf16_bytes_to_f32(mine[r].tobytes()), intermediate)) for r in changed]


def q8_0_gap(official: bytes, recomputed: bytes, ne0: int, limit: int = 5) -> dict:
    """Écart entre deux tenseurs Q8_0 : blocs, échelles, quanta et octets différents."""
    d_o, q_o = split_blocks(official)
    d_r, q_r = split_blocks(recomputed)
    bad_scale = d_o.view("<u2") != d_r.view("<u2")
    bad_quanta = q_o != q_r
    bad = bad_scale | bad_quanta.any(axis=1)
    byte_diff = np.frombuffer(official, np.uint8) != np.frombuffer(recomputed, np.uint8)
    first = []
    for i in np.flatnonzero(bad)[:limit].tolist():
        first.append(
            {
                "block": i,
                "row": i * QK8_0 // ne0,
                "col": i * QK8_0 % ne0,
                "scale_official": float(d_o[i]),
                "scale_recomputed": float(d_r[i]),
                "quanta_differing": int(bad_quanta[i].sum()),
            }
        )
    return {
        "blocks_total": int(bad.size),
        "blocks_differing": int(bad.sum()),
        "scales_differing": int(bad_scale.sum()),
        "quanta_differing": int(bad_quanta.sum()),
        "bytes_differing": int(byte_diff.sum()),
        "first_blocks": first,
    }


def fidelity_check(controls: list[tuple], gguf: GGUFFile, intermediate: str) -> dict:
    """Requantifie chaque tenseur témoin et compare aux octets du GGUF officiel.

    Un témoin est (origine, tenseur) ou (origine, tenseur, nom GGUF rapporté) : l'`embed_tokens`
    de la base, copié en `output.weight`, se compare à `token_embd` mais se rapporte sous le nom
    de la sortie.
    """
    results = []
    for origin, st, *shown in controls:
        info = gguf.tensor(gguf_name(st.name))
        _check_pair(st, info)
        gap = q8_0_gap(
            gguf.read_tensor_bytes(info.name), quantize_tensor(st, intermediate), info.dims[0]
        )
        entry = {
            "origin": origin,
            "hf_name": st.name,
            "gguf_name": shown[0] if shown else info.name,
        }
        entry |= {"ok": gap["blocks_differing"] == 0 and gap["bytes_differing"] == 0}
        if not entry["ok"]:
            entry["gap"] = gap
        results.append(entry)
    return {
        "intermediate": intermediate,
        "controls": len(results),
        "passed": sum(r["ok"] for r in results),
        "ok": bool(results) and all(r["ok"] for r in results),
        "tensors": results,
    }


def _fmt_param(value) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _quoted_block(keyword: str, text: str) -> str:
    if '"""' in text:
        raise ReportError(f'{keyword} contient """ : non transcriptible dans un Modelfile')
    return f'{keyword} """{text}"""'


def render_modelfile(gguf_path: Path, official: OllamaModel) -> str:
    """Modelfile : FROM le GGUF édité, puis renderer, parser, paramètres et textes de l'officiel."""
    lines = [f"FROM {gguf_path.resolve().as_posix()}", ""]
    if official.template is not None:
        lines.append(_quoted_block("TEMPLATE", official.template))
    if official.system is not None:
        lines.append(_quoted_block("SYSTEM", official.system))
    if official.renderer:
        lines.append(f"RENDERER {official.renderer}")
    if official.parser:
        lines.append(f"PARSER {official.parser}")
    for key, value in official.params.items():
        for item in value if isinstance(value, list) else [value]:
            lines.append(f"PARAMETER {key} {_fmt_param(item)}")
    if official.license is not None:
        lines.append(_quoted_block("LICENSE", official.license))
    return "\n".join(lines) + "\n"


@dataclass
class ReportResult:
    gguf: Path
    modelfile: Path
    diff: Path
    fidelity: Path


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8", newline="\n")


def public_text(text: str) -> str:
    """Remplace le dossier personnel par `~` : diff.json est destiné à un dépôt public."""
    home = Path.home()
    prefixes = {str(home), home.as_posix(), str(home.resolve()), home.resolve().as_posix()}
    for prefix in sorted(prefixes, key=len, reverse=True):
        text = text.replace(prefix, "~")
    return text


def report_gguf(
    checkpoint: str | Path,
    out_dir: str | Path,
    *,
    model: str = WSP_SOURCE_MODEL,
    tag: str = WSP_TAG,
    store: str | Path | None = None,
    base_checkpoint: str | Path | None = None,
    expected_sha256: dict[str, str] | None = None,
    intermediate: str = "f32",
    command: str | None = None,
    provenance: dict | None = None,
    lm_head_detached: bool = False,
) -> ReportResult:
    """`lm_head_detached` : le diff de séance porte un détachement de `lm_head` (CAP-10) ; le
    GGUF reçoit alors un `output.weight` Q8_0 (copie de `token_embd`, lignes éditées requantifiées).

    `provenance` ({"base": {...}, "edit": {...}}) décrit l'édition dans diff.json.

    Sans elle, l'édition est celle du MIT si `expected_sha256` est omis (valeurs par défaut) ;
    avec des sha256 fournis sans provenance (ex. une séance), base et dépôt restent inconnus
    (null) : le diff ne prétend jamais que l'édition est celle du MIT.
    """
    out = Path(out_dir)
    stem = tag.replace(":", "-").replace("/", "-")
    paths = ReportResult(
        out / f"{stem}.gguf", out / "Modelfile", out / "diff.json", out / "fidelite.json"
    )
    written: set[Path] = set()
    try:
        return _report(
            checkpoint,
            paths,
            written,
            model=model,
            tag=tag,
            store=store,
            base_checkpoint=base_checkpoint,
            expected_sha256=expected_sha256,
            intermediate=intermediate,
            command=command,
            provenance=provenance,
            lm_head_detached=lm_head_detached,
        )
    except BaseException:
        # Un échec ne laisse aucune sortie d'une séance précédente : `ollama create` ne doit pas
        # charger une édition périmée. Seul fidelite.json de cette séance reste, pour le détail.
        stale = [paths.gguf, paths.gguf.with_name(paths.gguf.name + ".partial")]
        stale += [paths.modelfile, paths.diff, paths.fidelity]
        for path in stale:
            if path not in written or path != paths.fidelity:
                path.unlink(missing_ok=True)
        raise


def _report(
    checkpoint: str | Path,
    paths: ReportResult,
    written: set[Path],
    *,
    model: str,
    tag: str,
    store: str | Path | None,
    base_checkpoint: str | Path | None,
    expected_sha256: dict[str, str] | None,
    intermediate: str,
    command: str | None,
    provenance: dict | None,
    lm_head_detached: bool,
) -> ReportResult:
    expected = WSP_EDITED_SHA256 if expected_sha256 is None else expected_sha256
    if provenance is None:
        if expected_sha256 is None:
            provenance = {
                "base": {"repo": WSP_BASE_REPO, "revision": WSP_BASE_REVISION},
                "edit": {"repo": WSP_EDIT_REPO},
            }
        else:
            provenance = {"base": {"repo": None, "revision": None}, "edit": {"repo": None}}
    official = load_model(model, store)
    gguf = read_gguf(official.model_blob)
    # Blob officiel vérifié d'abord : corrompu, il ferait échouer à tort le contrôle de fidélité.
    source_sha = sha256_file(official.model_blob)
    if f"sha256:{source_sha}" != official.model_digest:
        raise ReportError(f"blob {official.model_digest} corrompu (sha256 {source_sha})")
    edited = open_checkpoint(checkpoint)
    base = open_checkpoint(base_checkpoint) if base_checkpoint else None

    # 1. Tenseurs visés : présence et sha256 publié d'abord, puis correspondance et forme.
    for hf_name, want in expected.items():
        if hf_name not in edited:
            raise ReportError(f"{hf_name} absent du checkpoint {checkpoint}")
        got = sha256_bytes(edited[hf_name].read_bytes())
        if got != want:
            raise ShaMismatchError(f"{hf_name} : sha256 {got}, attendu {want} ; report refusé")
    lm_heads = [n for n in expected if gguf_name(n) == OUTPUT_WEIGHT]
    if lm_heads and not lm_head_detached:
        raise ReportError(
            f"{lm_heads[0]} visé sans détachement déclaré par le diff de séance : report refusé"
        )
    if lm_head_detached and len(lm_heads) != 1:
        raise ReportError(
            "le diff déclare un détachement de lm_head mais ne porte pas lm_head.weight"
        )
    embed_name = find_embed(edited) if lm_head_detached else None
    targets: list[tuple[SafeTensor, TensorInfo]] = []
    for hf_name in expected:
        if hf_name in lm_heads:
            _check_pair(edited[hf_name], gguf.tensor(TIED_EMBEDDING))
            if edited[hf_name].shape != edited[embed_name].shape:
                raise ReportError(f"{hf_name} et {embed_name} : formes différentes")
            continue
        name = gguf_name(hf_name)
        if name is None:
            raise ReportError(f"{hf_name} : pas de correspondance GGUF vérifiée")
        st, info = edited[hf_name], gguf.tensor(name)
        _check_pair(st, info)
        targets.append((st, info))
    if lm_head_detached:
        if OUTPUT_WEIGHT in gguf.tensors:
            raise ReportError(f"lm_head détaché mais le GGUF officiel a déjà un {OUTPUT_WEIGHT}")
    elif any(info.name == TIED_EMBEDDING for _, info in targets):
        _check_tied(checkpoint, edited, gguf)
    unwitnessed = [info.name for st, info in targets if not is_witness(st.name)]
    unwitnessed += [OUTPUT_WEIGHT] * len(lm_heads)
    if unwitnessed and base is None:
        raise ReportError(
            f"{', '.join(unwitnessed)} : aucun témoin de même nature ; --base-checkpoint requis "
            "(l'original du tenseur visé sert de témoin)"
        )

    # 2. Contrôle de fidélité, avant toute écriture de GGUF.
    controls = [
        ("checkpoint édité, tenseur non visé", st)
        for name, st in sorted(edited.items())
        if name not in expected and is_witness(name) and gguf_name(name) in gguf.tensors
    ]
    if base is not None:
        for hf_name in expected:
            if hf_name in lm_heads:
                # La base n'a pas de lm_head : son original est l'embed_tokens, copié en sortie.
                source = find_embed(base)
                origin = "checkpoint de base, embed_tokens d'origine (copie de sortie)"
                controls.append((origin, base[source], OUTPUT_WEIGHT))
                continue
            if hf_name not in base:
                raise ReportError(f"{hf_name} absent du checkpoint de base {base_checkpoint}")
            controls.append(("checkpoint de base, original du tenseur visé", base[hf_name]))
    fidelity = fidelity_check(controls, gguf, intermediate)
    out = paths.gguf.parent
    out.mkdir(parents=True, exist_ok=True)
    _write_json(paths.fidelity, fidelity)
    written.add(paths.fidelity)
    if not fidelity["ok"]:
        failed = [t["gguf_name"] for t in fidelity["tensors"] if not t["ok"]]
        reason = "aucun tenseur témoin" if not controls else f"écart sur {', '.join(failed)}"
        raise FidelityError(
            f"contrôle de fidélité en échec ({reason}) ; aucun GGUF écrit", fidelity
        )

    # 3. Requantification des tenseurs visés et réécriture des seules plages concernées.
    entries, patches, embed_after = [], [], None
    for st, info in targets:
        before = gguf.read_tensor_bytes(info.name)
        after = quantize_tensor(st, intermediate)
        if len(after) != info.n_bytes:
            raise ReportError(
                f"{info.name} : {len(after)} octets requantifiés, {info.n_bytes} attendus"
            )
        patches.append((info.name, 0, after))
        embed_after = after if info.name == TIED_EMBEDDING else embed_after
        hf_before = sha256_bytes(base[st.name].read_bytes()) if base is not None else None
        entries.append(
            {
                "hf_name": st.name,
                "gguf_name": info.name,
                "ggml_type": info.type_name,
                "hf_shape": list(st.shape),
                "gguf_dims": list(info.dims),
                "gguf_offset": info.offset,
                "gguf_n_bytes": info.n_bytes,
                "hf_bf16_sha256_before": hf_before,
                "hf_bf16_sha256_after": sha256_bytes(st.read_bytes()),
                "gguf_sha256_before": sha256_bytes(before),
                "gguf_sha256_after": sha256_bytes(after),
                "gguf_blocks_changed": q8_0_gap(before, after, info.dims[0])["blocks_differing"],
            }
        )
    lm_rows: list[tuple[int, bytes]] = []
    if lm_heads:
        lm_head = edited[lm_heads[0]]
        lm_rows = _lm_head_rows(lm_head, edited[embed_name], intermediate)
        before = gguf.read_tensor_bytes(TIED_EMBEDDING)
        # output.weight démarre en copie de token_embd (édité s'il l'est), puis les lignes de
        # lm_head qui diffèrent de l'embed édité sont requantifiées.
        if embed_after is not None:
            patches.append((OUTPUT_WEIGHT, 0, embed_after))
        patches += [(OUTPUT_WEIGHT, row * _row_bytes(lm_head), data) for row, data in lm_rows]
        reference = embed_after if embed_after is not None else before
        info = gguf.tensor(TIED_EMBEDDING)
        entries.append(
            {
                "hf_name": lm_head.name,
                "gguf_name": OUTPUT_WEIGHT,
                "ggml_type": info.type_name,
                "hf_shape": list(lm_head.shape),
                "gguf_dims": list(info.dims),
                "gguf_offset": None,
                "gguf_n_bytes": info.n_bytes,
                "hf_bf16_sha256_before": (
                    sha256_bytes(base[find_embed(base)].read_bytes()) if base is not None else None
                ),
                "hf_bf16_sha256_after": sha256_bytes(lm_head.read_bytes()),
                "gguf_sha256_before": sha256_bytes(before),
                "gguf_sha256_after": None,
                "gguf_blocks_changed": sum(
                    q8_0_gap(
                        reference[r * _row_bytes(lm_head) : (r + 1) * _row_bytes(lm_head)],
                        data,
                        info.dims[0],
                    )["blocks_differing"]
                    for r, data in lm_rows
                ),
                "rows_edited": [r for r, _ in lm_rows],
            }
        )
    gguf_path, modelfile = paths.gguf, paths.modelfile
    # Modelfile rendu avant l'écriture du GGUF : un texte non transcriptible n'écrit rien.
    modelfile_text = render_modelfile(gguf_path, official)
    partial = gguf_path.with_name(gguf_path.name + ".partial")
    out_gguf = gguf
    if lm_heads:
        from slm_weight_editing.gguf_add_output import add_output_weight

        partial.unlink(missing_ok=True)
        add_output_weight(official.model_blob, partial, store)
        out_gguf = read_gguf(partial)
    else:
        shutil.copyfile(official.model_blob, partial)
    with partial.open("r+b") as f:
        for name, offset, data in patches:
            f.seek(out_gguf.tensor(name).offset + offset)
            f.write(data)
    if lm_heads:
        output_bytes = out_gguf.read_tensor_bytes(OUTPUT_WEIGHT)
        entries[-1]["gguf_sha256_after"] = sha256_bytes(output_bytes)
    for entry in entries:
        entry["gguf_offset"] = out_gguf.tensor(entry["gguf_name"]).offset
    os.replace(partial, gguf_path)

    # 4. Modelfile et diff vérifiable (chemins personnels masqués : le diff se publie).
    modelfile.write_text(modelfile_text, "utf-8", newline="\n")
    diff = {
        "format": DIFF_FORMAT,
        "tag": tag,
        "source_model": model,
        "base": provenance["base"],
        "edit": provenance["edit"],
        "intermediate": intermediate,
        "tensors": entries,
        "gguf": {
            "source_digest": official.model_digest,
            "source_sha256": source_sha,
            "output": gguf_path.name,
            "output_sha256": sha256_file(gguf_path),
            "alignment": out_gguf.alignment,
            "data_offset": out_gguf.data_offset,
        },
        "lm_head_detached": bool(lm_heads),
        "protocol_gap": PROTOCOL_GAP if lm_heads else None,
        "fidelity": {k: fidelity[k] for k in ("intermediate", "controls", "passed", "ok")}
        | {"target_originals_checked": base is not None},
        "modelfile": {"renderer": official.renderer, "parser": official.parser},
        "apply": public_text(command) if command is not None else None,
        "ollama_create": public_text(f"ollama create {tag} -f {modelfile.as_posix()}"),
    }
    _write_json(paths.diff, diff)
    return paths
