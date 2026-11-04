"""Édition directe des poids de petits modèles de langage par un LLM, sans entraînement."""

from __future__ import annotations

import argparse
import shlex
import sys

__version__ = "0.0.0"


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
    rep.add_argument("--tag", default=WSP_TAG, help="tag du modèle édité")
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
    from slm_weight_editing.compare_edits import add_parser

    add_parser(sub)
    harness.add_parser(sub)
    from slm_weight_editing import audit_transcript

    audit_transcript.add_parser(sub)
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
    from slm_weight_editing.report_gguf import FidelityError, ReportError, report_gguf
    from slm_weight_editing.safetensors_file import SafetensorsError

    out = args.out or default_out_dir(args.tag)
    try:
        result = report_gguf(
            args.checkpoint,
            out,
            model=args.model,
            tag=args.tag,
            store=args.ollama_models,
            base_checkpoint=args.base_checkpoint,
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
