"""Commande `probe` (CAP-1) : forward forcé du modèle de `work/`, sans aucun jeton généré.

L'éditeur obtient les logits et des activations sur des textes fixes. Chaque appel décompte les
jetons passés en forward d'un budget de probe (ligne `open` du journal), distinct du budget de
lots. Les captures sont écrites dans `<espace>/probes/<n>/` (`manifest.json` et un `.npy`
float32 par capture, lisible en `mmap_mode`) ; elles ne sont pas publiées au gel, seuls leurs
sha256 figurent au journal. Format : `_bmad-output/specs/spec-harnais-v2/probe-captures.md`.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

from slm_weight_editing import harness
from slm_weight_editing.harness import HarnessError

PROBES_DIR = "probes"
MANIFEST_FILE = "manifest.json"
MAX_TEXTS = 64
MAX_TEXT_TOKENS = 8192
MAX_OUTPUT_BYTES = 2 * 1000**3
TOP_K = 20
CAPTURES = ("resid", "mlp_act", "attn_out", "final_norm")
DEFAULT_PROBE_BUDGET = 2_000_000
DEFAULT_PROBE_DISK_MAX = 20 * 1000**3
_ATTN_NAMES = ("self_attn", "linear_attn", "attn", "token_mixer")


def _int_list(value: str | None, option: str) -> list[int] | None:
    if value is None:
        return None
    try:
        items = [int(x) for x in value.split(",")]
    except ValueError:
        raise HarnessError(
            f"{option} {value!r} refusé : entiers séparés par des virgules"
        ) from None
    if any(i < 0 for i in items):
        raise HarnessError(f"{option} {value!r} refusé : entiers positifs ou nuls seulement")
    return sorted(set(items))


def parse_captures(value: str | None) -> list[str]:
    if not value:
        raise HarnessError(f"--capture obligatoire : un sous-ensemble de {','.join(CAPTURES)}")
    items = value.split(",")
    bad = [i for i in items if i not in CAPTURES]
    if bad:
        raise HarnessError(f"--capture {bad[0]!r} refusé : {', '.join(CAPTURES)} seulement")
    return [c for c in CAPTURES if c in items]


def parse_positions(value: str) -> tuple[str, int, int | None]:
    """`all`, `last` ou une plage `A:B` (B exclu) -> (genre, début, fin)."""
    if value in ("all", "last"):
        return value, 0, None
    start, sep, end = value.partition(":")
    if sep and start.isdigit() and end.isdigit() and int(start) < int(end):
        return "range", int(start), int(end)
    raise HarnessError(f"--positions {value!r} refusé : all, last ou une plage A:B (A < B)")


def select_positions(spec: tuple[str, int, int | None], n_tokens: int) -> list[int]:
    kind, start, end = spec
    if kind == "all":
        return list(range(n_tokens))
    if kind == "last":
        return [n_tokens - 1]
    if start >= n_tokens:
        raise HarnessError(f"--positions {start}:{end} refusé : le texte n'a que {n_tokens} jetons")
    return list(range(start, min(end, n_tokens)))


@dataclasses.dataclass(frozen=True)
class Dims:
    hidden: int
    intermediate: int
    layers: int
    vocab: int


def read_dims(work: Path) -> Dims:
    """Dimensions lues dans config.json (aucun chargement du modèle)."""
    try:
        cfg = json.loads((work / "config.json").read_text("utf-8"))
        cfg = cfg.get("text_config", cfg)
        hidden = int(cfg["hidden_size"])
        return Dims(
            hidden,
            int(cfg.get("intermediate_size", hidden)),
            int(cfg["num_hidden_layers"]),
            int(cfg["vocab_size"]),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HarnessError(f"config.json de work/ illisible pour probe ({exc})", 1) from exc


def estimate_output_bytes(
    dims: Dims,
    rows: int,
    captures: list[str],
    n_layers: int,
    n_token_ids: int,
    full_vocab_rows: int,
) -> int:
    """Octets de captures estimés avant calcul (marge de 1 % comprise)."""
    per_row = TOP_K * 12 + n_token_ids * 4
    if "resid" in captures:
        per_row += n_layers * 2 * dims.hidden * 4
    if "mlp_act" in captures:
        per_row += n_layers * dims.intermediate * 4
    if "attn_out" in captures:
        per_row += n_layers * dims.hidden * 4
    if "final_norm" in captures:
        per_row += dims.hidden * 4
    total = rows * per_row + full_vocab_rows * dims.vocab * 4
    return int(total * 1.01) + 1


def read_texts(path: Path) -> list[str]:
    """Fichier JSONL : une ligne `{"text": "..."}` par texte (au plus 64)."""
    try:
        raw = path.read_text("utf-8")
    except OSError as exc:
        raise HarnessError(f"--texts illisible : {path} ({exc})", 1) from exc
    texts = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            ok = isinstance(row, dict) and set(row) == {"text"} and isinstance(row["text"], str)
        except ValueError:
            ok = False
        if not ok or not row["text"]:
            raise HarnessError(
                f"--texts {path.name} ligne {number} refusée : "
                'attendu {"text": "texte non vide"} seulement'
            )
        texts.append(row["text"])
        if len(texts) > MAX_TEXTS:
            raise HarnessError(f"plafond de {MAX_TEXTS} textes par appel : plus demandés")
    if not texts:
        raise HarnessError(f"--texts {path.name} ne contient aucun texte")
    return texts


def _probe_limits(journal: list[dict]) -> tuple[int, int]:
    opened = next((e for e in journal if e.get("event") == "open"), None)
    if opened is None:
        raise HarnessError("journal sans ligne open", 1)
    budget, disk = opened.get("probe_budget"), opened.get("probe_disk_max")
    if type(budget) is not int or type(disk) is not int:
        raise HarnessError(
            "probe refusé : cette séance a été ouverte sans budget de probe "
            "(options --probe-budget et --probe-disk-max de session-workspace)"
        )
    return budget, disk


def _layer_modules(model):
    base = model.model
    base = getattr(base, "language_model", base)
    return list(base.layers), base.norm


def _attn_module(layer):
    for name in _ATTN_NAMES:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise HarnessError("attn_out : bloc sans module d'attention reconnu", 1)


def _first(out):
    return out[0] if isinstance(out, tuple) else out


def _forward_all(
    work: Path,
    id_lists: list[list[int]],
    row_sets: list[list[int]],
    full_vocab: list[int],
    captures: list[str],
    layers: list[int],
    token_ids: list[int],
    device: str,
    out_dir: Path,
) -> dict:
    """Un forward par texte, captures par crochets, écriture directe en `.npy` (memmap)."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(work, local_files_only=True, dtype="auto")
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    blocks, norm = _layer_modules(model)
    if len(blocks) != read_dims(work).layers:
        raise HarnessError("nombre de blocs du modèle différent de config.json", 1)
    seen: dict[str, torch.Tensor] = {}
    hooks = []

    def keep(name):
        def store(_module, _args, out=None):
            seen[name] = _first(out if out is not None else _args[0])

        return store

    def keep_pre(name):
        def store(_module, args, kwargs):
            seen[name] = args[0] if args else kwargs["hidden_states"]

        return store

    for index in layers:
        block = blocks[index]
        if "resid" in captures:
            hooks.append(
                block.register_forward_pre_hook(keep_pre(f"resid_pre_l{index}"), with_kwargs=True)
            )
            hooks.append(block.register_forward_hook(keep(f"resid_l{index}")))
        if "mlp_act" in captures:
            hooks.append(
                block.mlp.down_proj.register_forward_pre_hook(
                    lambda m, a, n=f"mlp_act_l{index}": seen.__setitem__(n, a[0])
                )
            )
        if "attn_out" in captures:
            hooks.append(_attn_module(block).register_forward_hook(keep(f"attn_out_l{index}")))
    if "final_norm" in captures:
        hooks.append(norm.register_forward_hook(keep("final_norm")))

    rows = sum(len(r) for r in row_sets)
    dims = read_dims(work)
    shapes: dict[str, tuple[int, ...]] = {}
    maps: dict[str, np.ndarray] = {}

    def memmap(name: str, shape: tuple[int, ...], dtype) -> np.ndarray:
        shapes[name] = shape
        maps[name] = np.lib.format.open_memmap(
            out_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        return maps[name]

    top_ids = memmap("top_logit_ids", (rows, min(TOP_K, dims.vocab)), np.int64)
    top_vals = memmap("top_logit_values", (rows, min(TOP_K, dims.vocab)), np.float32)
    tok_logits = memmap("token_logits", (rows, len(token_ids)), np.float32) if token_ids else None
    fv = (
        memmap("full_vocab_logits", (len(id_lists) * len(full_vocab), dims.vocab), np.float32)
        if full_vocab
        else None
    )
    # Les noms de captures dépendent de la sortie réelle des crochets : créés au premier texte.
    dtype_name = None
    offset = 0
    fv_offset = 0
    try:
        with torch.inference_mode():
            for ids, selected in zip(id_lists, row_sets, strict=True):
                seen.clear()
                tensor = torch.tensor([ids], dtype=torch.long, device=device)
                logits = model(input_ids=tensor).logits[0].float()
                dtype_name = str(model.dtype)
                idx = torch.tensor(selected, dtype=torch.long, device=device)
                top = torch.topk(logits[idx], top_ids.shape[1], dim=-1)
                top_ids[offset : offset + len(selected)] = top.indices.cpu().numpy()
                top_vals[offset : offset + len(selected)] = top.values.cpu().numpy()
                if tok_logits is not None:
                    chosen = logits[idx][:, token_ids]
                    tok_logits[offset : offset + len(selected)] = chosen.cpu().numpy()
                if fv is not None:
                    for position in full_vocab:
                        fv[fv_offset] = logits[position].cpu().numpy()
                        fv_offset += 1
                for name, value in seen.items():
                    array = value[0, idx].float().cpu().numpy()
                    if name not in maps:
                        memmap(name, (rows, array.shape[-1]), np.float32)
                    maps[name][offset : offset + len(selected)] = array
                offset += len(selected)
    finally:
        for hook in hooks:
            hook.remove()
        for array in maps.values():
            array.flush()
        maps.clear()
        del top_ids, top_vals, tok_logits, fv
    return {"shapes": shapes, "model_dtype": dtype_name}


def run_probe(
    session_id: str,
    texts_path: str | Path,
    *,
    capture: list[str],
    layers: list[int] | None = None,
    positions: str = "all",
    token_ids: list[int] | None = None,
    full_vocab_positions: list[int] | None = None,
    sessions_root: str | Path = harness.DEFAULT_SESSIONS_ROOT,
    probes_root: str | Path | None = None,
    device: str = "cpu",
    argv: dict | None = None,
) -> dict:
    """Un appel probe : contrôles, réservation, forward, captures, journal. Rien n'est généré."""
    spec = parse_positions(positions)
    token_ids = token_ids or []
    full_vocab = full_vocab_positions or []
    directory = harness.session_dir(session_id, sessions_root)
    harness.read_state(directory)
    with harness._session_lock(directory):
        state = harness.read_state(directory)
        if state.frozen:
            raise HarnessError("séance gelée : plus aucune probe")
        journal = harness.read_journal(directory)
        budget, disk_max = _probe_limits(journal)
        work = directory / harness.WORK_DIR
        dims = read_dims(work)
        raw_texts = read_texts(Path(texts_path))
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(work, local_files_only=True)
        id_lists = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in raw_texts]
        for i, ids in enumerate(id_lists):
            if not ids:
                raise HarnessError(f"texte {i} sans aucun jeton")
            if len(ids) > MAX_TEXT_TOKENS:
                raise HarnessError(
                    f"plafond de {MAX_TEXT_TOKENS} jetons par texte : le texte {i} en a {len(ids)}"
                )
        chosen = list(range(dims.layers)) if layers is None else layers
        if layers is not None and any(layer >= dims.layers for layer in layers):
            raise HarnessError(
                f"--layers refusé : le modèle a {dims.layers} couches (0 à {dims.layers - 1})"
            )
        if any(t >= dims.vocab for t in token_ids):
            raise HarnessError(f"--token-ids refusé : le vocabulaire a {dims.vocab} jetons")
        row_sets = [select_positions(spec, len(ids)) for ids in id_lists]
        for position in full_vocab:
            short = [len(ids) for ids in id_lists if position >= len(ids)]
            if short:
                raise HarnessError(
                    f"--full-vocab-positions {position} refusé : "
                    f"un texte n'a que {min(short)} jetons"
                )
        tokens = sum(len(ids) for ids in id_lists)
        rows = sum(len(r) for r in row_sets)
        estimate = estimate_output_bytes(
            dims, rows, capture, len(chosen), len(token_ids), len(full_vocab) * len(id_lists)
        )
        if estimate > MAX_OUTPUT_BYTES:
            raise HarnessError(
                f"plafond de sortie de {MAX_OUTPUT_BYTES} octets : {estimate} octets estimés"
            )
        used = harness.used_probe_tokens(directory)
        if used + tokens > budget:
            raise HarnessError(
                f"budget de probe dépassé : {used} jetons utilisés + {tokens} demandés > {budget}"
            )
        reserved_bytes = sum(
            e.get("estimated_bytes", 0) for e in journal if e.get("event") == "probe_reserve"
        )
        if reserved_bytes + estimate > disk_max:
            raise HarnessError(
                f"plafond disque de probe de {disk_max} octets : {reserved_bytes} déjà réservés "
                f"+ {estimate} estimés"
            )
        actual_device = harness.resolve_device(device)
        if actual_device == "cuda":
            harness.check_cuda_memory(work)
        number = sum(1 for e in journal if e.get("event") == "probe_reserve") + 1
        base = Path(probes_root) if probes_root is not None else directory / PROBES_DIR
        out_dir = base / str(number)
        if out_dir.exists():
            raise HarnessError(f"{out_dir} existe déjà : numéro de probe {number} déjà pris")
        hashes = harness.tensor_hashes(work)
        files = harness.work_file_hashes(work)
        # Réservation avant le calcul, comme un lot : un plantage consomme les jetons.
        started = harness._now()
        harness._append_journal(
            directory,
            {
                "event": "probe_reserve",
                "probe": number,
                "tokens": tokens,
                "estimated_bytes": estimate,
                "budget": budget,
                "device": actual_device,
                "at": started,
            },
        )
        out_dir.mkdir(parents=True)
        result = _run_forward(
            work, id_lists, row_sets, full_vocab, capture, chosen, token_ids, actual_device, out_dir
        )
        offsets = [sum(len(r) for r in row_sets[:i]) for i in range(len(row_sets))]
        texts_meta = [
            {"token_ids": ids, "positions": rows_, "offset": offset}
            for ids, rows_, offset in zip(id_lists, row_sets, offsets, strict=True)
        ]
        manifest = {
            "probe": number,
            "session_id": session_id,
            "tokens": tokens,
            "texts": texts_meta,
            "full_vocab_positions": full_vocab,
            "full_vocab_rows": [
                {"text": i, "position": position}
                for i in range(len(id_lists))
                for position in full_vocab
            ],
            "token_ids": token_ids,
            "layers": chosen,
            "capture": capture,
            "arguments": argv or {},
            "tensors_sha256": hashes,
            "files_sha256": files,
            "model_dtype": result["model_dtype"],
            "arrays": {
                name: {"shape": list(shape), "dtype": "float32" if "ids" not in name else "int64"}
                for name, shape in result["shapes"].items()
            },
        }
        manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        (out_dir / MANIFEST_FILE).write_bytes(manifest_bytes)
        entry = {
            "event": "probe",
            "probe": number,
            "tokens": tokens,
            "budget": budget,
            "remaining": budget - used - tokens,
            "arguments": argv or {},
            "tensors_sha256": hashes,
            "files_sha256": files,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "output_files": {
                p.name: harness._sha256_file(p) for p in sorted(out_dir.glob("*.npy"))
            },
            "device": harness.device_info(actual_device),
            "started_at": started,
            "finished_at": harness._now(),
        }
        harness._append_journal(directory, entry)
        return entry


def _run_forward(*args) -> dict:
    import gc

    try:
        return _forward_all(*args)
    finally:
        gc.collect()


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "probe",
        help="forward forcé sur des textes fixes : logits et activations, sans génération",
        description=(
            f"Au plus {MAX_TEXTS} textes de {MAX_TEXT_TOKENS} jetons, sortie estimée à "
            f"{MAX_OUTPUT_BYTES // 1000**3} Go au plus ; décompté du budget de probe (jetons "
            "passés en forward). Écrit probes/<n>/ dans l'espace de séance."
        ),
    )
    p.add_argument("--session", required=True)
    p.add_argument("--sessions-root", help="dossier des séances (celui de workspace.json)")
    p.add_argument("--texts", required=True, help='fichier JSONL de lignes {"text": "..."}')
    p.add_argument("--capture", required=True, help=f"parmi {','.join(CAPTURES)} (virgules)")
    p.add_argument("--layers", help="couches (virgules ; défaut : toutes)")
    p.add_argument("--positions", default="all", help="all (défaut), last ou une plage A:B")
    p.add_argument("--token-ids", help="jetons dont on veut la logit à chaque position")
    p.add_argument("--full-vocab-positions", help="positions au vocabulaire entier (virgules)")
    p.add_argument("--device", choices=harness.DEVICES, default="auto")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        capture = parse_captures(args.capture)
        layers = _int_list(args.layers, "--layers")
        token_ids = _int_list(args.token_ids, "--token-ids")
        full_vocab = _int_list(args.full_vocab_positions, "--full-vocab-positions")
        found = harness.find_workspace()
        root = harness.bind_workspace(args.session, args.sessions_root)
        texts = Path(args.texts)
        if not texts.is_file():
            raise HarnessError(f"--texts illisible : {texts} (fichier absent)", 1)
        probes_root = None
        if found is not None:
            ws = found[0]
            resolved = (texts if texts.is_absolute() else Path.cwd() / texts).resolve()
            if ws not in resolved.parents:
                raise HarnessError(f"--texts refusé : fichier hors de {ws}")
            probes_root = ws / PROBES_DIR
        entry = run_probe(
            args.session,
            texts,
            capture=capture,
            layers=layers,
            positions=args.positions,
            token_ids=token_ids,
            full_vocab_positions=full_vocab,
            sessions_root=root,
            probes_root=probes_root,
            device=args.device,
            argv={
                "texts": args.texts,
                "texts_sha256": harness._sha256_file(texts),
                "capture": capture,
                "layers": layers,
                "positions": args.positions,
                "token_ids": token_ids,
                "full_vocab_positions": full_vocab,
            },
        )
        print(
            f"Probe {entry['probe']} : {entry['tokens']} jetons, reste {entry['remaining']} "
            f"jetons de probe ; captures dans probes/{entry['probe']}/"
        )
    except HarnessError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        print(harness.EDITOR_WAY, file=sys.stderr)
        return exc.exit_code
    except OSError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    return 0
