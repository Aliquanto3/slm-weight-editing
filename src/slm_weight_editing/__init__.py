"""Édition directe des poids de petits modèles de langage par un LLM, sans entraînement."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

__version__ = "0.1.0"


def _parser() -> argparse.ArgumentParser:
    from slm_weight_editing import harness
    from slm_weight_editing.q8_0 import INTERMEDIATES
    from slm_weight_editing.report_gguf import WSP_SOURCE_MODEL, WSP_TAG

    parser = argparse.ArgumentParser(prog="slm-weight-editing")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser(
        "report-gguf",
        help="reporte l'édition du MIT (RikiyaT) dans le GGUF Ollama officiel",
        description=(
            "Vérifie les sha256 des tenseurs édités, contrôle la fidélité de la requantification, "
            "puis écrit le GGUF édité, son Modelfile et un diff vérifiable. Ne lance pas Ollama."
        ),
    )
    rep.add_argument("--checkpoint", required=True, help="dossier ou fichier safetensors édité")
    rep.add_argument(
        "--base-checkpoint",
        required=True,
        help="checkpoint de base : contrôle de fidélité des originaux visés (CAP-1), sha256 avant",
    )
    rep.add_argument("--model", default=WSP_SOURCE_MODEL, help="modèle Ollama officiel")
    rep.add_argument(
        "--tag", help=f"tag du modèle édité (défaut {WSP_TAG} ; obligatoire avec --edit-diff)"
    )
    rep.add_argument(
        "--edit-diff",
        help="diff.json d'une séance gelée : ses sha256 après remplacent ceux de l'édition du MIT",
    )
    rep.add_argument(
        "--ollama-models", help="magasin d'Ollama (sinon OLLAMA_MODELS, sinon ~/.ollama/models)"
    )
    rep.add_argument("--out", help="dossier de sortie (défaut : models/<tag>)")
    rep.add_argument(
        "--intermediate",
        choices=INTERMEDIATES,
        default="f32",
        help="passage bf16 -> fXX avant Q8_0 (défaut f32, comme llama.cpp)",
    )
    from slm_weight_editing import compare_edits, freeze

    compare_edits.add_parser(sub)
    freeze.add_parser(sub)
    harness.add_parser(sub)
    from slm_weight_editing import audit_transcript, gguf_add_output, script_runner, workspace

    audit_transcript.add_parser(sub)
    gguf_add_output.add_parser(sub)
    script_runner.add_parser(sub)
    workspace.add_parser(sub)
    return parser


def default_out_dir(tag: str) -> str:
    return "models/" + tag.replace(":", "-").replace("/", "-")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = _parser().parse_args(argv)
    if getattr(args, "func", None):
        return args.func(args)
    from slm_weight_editing.gguf_file import GGUFError
    from slm_weight_editing.ollama_store import OllamaStoreError
    from slm_weight_editing.report_gguf import (
        WSP_TAG,
        FidelityError,
        ReportError,
        report_gguf,
    )
    from slm_weight_editing.safetensors_file import SafetensorsError

    expected, provenance = None, None
    if args.edit_diff:
        from slm_weight_editing.freeze import DiffError, load_diff, relevant_files
        from slm_weight_editing.report_gguf import sha256_file

        if not args.tag:
            print("Erreur : --edit-diff demande --tag (pas de tag du MIT)", file=sys.stderr)
            return 1
        try:
            meta, _ = load_diff(args.edit_diff)
            expected = {t["name"]: t["sha256_after"] for t in meta["tensors"]}
        except (DiffError, KeyError, TypeError) as exc:
            print(f"Erreur : diff invalide ({exc})", file=sys.stderr)
            return 1
        if not expected:
            print("Erreur : le diff ne modifie aucun tenseur", file=sys.stderr)
            return 1
        other_files = relevant_files(meta["files_changed"])
        if other_files:
            names = ", ".join(str(f.get("path")) for f in other_files[:3])
            print(
                f"Erreur : la séance a aussi modifié des fichiers hors tenseurs ({names}) ; "
                "ni apply-diff ni report-gguf ne les rejouent, le GGUF ne serait donc pas "
                "l'édition gelée : report refusé",
                file=sys.stderr,
            )
            return 1
        provenance = {
            "base": {"repo": None, "revision": meta.get("base_revision")},
            "edit": {
                "repo": None,
                "session_id": meta.get("session_id"),
                "diff_sha256": sha256_file(Path(args.edit_diff)),
                "journal_sha256": meta.get("journal", {}).get("sha256"),
            },
        }
    args.tag = args.tag or WSP_TAG
    out = args.out or default_out_dir(args.tag)
    try:
        result = report_gguf(
            args.checkpoint,
            out,
            model=args.model,
            tag=args.tag,
            store=args.ollama_models,
            base_checkpoint=args.base_checkpoint,
            expected_sha256=expected,
            provenance=provenance,
            intermediate=args.intermediate,
            command="uv run slm-weight-editing " + shlex.join(argv),
        )
    except FidelityError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        for t in exc.report["tensors"]:
            if not t["ok"]:
                g = t["gap"]
                print(
                    f"  {t['gguf_name']} ({t['origin']}) : {g['blocks_differing']}/"
                    f"{g['blocks_total']} blocs, {g['scales_differing']} échelles, "
                    f"{g['bytes_differing']} octets différents",
                    file=sys.stderr,
                )
        print(f"  détail : {out}/fidelite.json", file=sys.stderr)
        return exc.exit_code
    except ReportError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return exc.exit_code
    except (GGUFError, OllamaStoreError, SafetensorsError, OSError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    print(f"GGUF édité : {result.gguf}")
    print(f"Modelfile : {result.modelfile}")
    print(f"Diff : {result.diff}")
    print(f"À lancer par Anaël : ollama create {args.tag} -f {result.modelfile.as_posix()}")
    return 0
