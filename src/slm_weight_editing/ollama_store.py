"""Lecture seule du magasin local d'Ollama : manifeste, configuration, couches et blob GGUF.

Aucun appel à `ollama` : on lit les fichiers sous `<magasin>/manifests` et `<magasin>/blobs`.
Le renderer et le parser sont dans le blob de configuration (`config.digest`), les paramètres,
la licence, le gabarit et le message système dans des couches typées par `mediaType`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

REGISTRY = "registry.ollama.ai"
MEDIA_PREFIX = "application/vnd.ollama.image."


class OllamaStoreError(ValueError):
    """Manifeste ou blob introuvable dans le magasin d'Ollama."""


def default_store(explicit: str | Path | None = None) -> Path:
    """Magasin : option explicite, sinon `OLLAMA_MODELS`, sinon `~/.ollama/models`."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return Path(env)
    return Path.home() / ".ollama" / "models"


def manifest_path(store: Path, model: str) -> Path:
    """`qwen3.5:0.8b` -> manifests/registry.ollama.ai/library/qwen3.5/0.8b."""
    name, _, tag = model.partition(":")
    tag = tag or "latest"
    parts = name.split("/")
    if len(parts) == 1:
        parts = [REGISTRY, "library", *parts]
    elif len(parts) == 2:
        parts = [REGISTRY, *parts]
    return store.joinpath("manifests", *parts, tag)


def blob_path(store: Path, digest: str) -> Path:
    return store / "blobs" / digest.replace(":", "-")


@dataclass
class OllamaModel:
    model: str
    store: Path
    manifest: Path
    model_blob: Path
    model_digest: str
    config: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    license: str | None = None
    template: str | None = None
    system: str | None = None

    @property
    def renderer(self) -> str | None:
        return self.config.get("renderer")

    @property
    def parser(self) -> str | None:
        return self.config.get("parser")


def _read_blob(store: Path, digest: str) -> bytes:
    path = blob_path(store, digest)
    if not path.is_file():
        raise OllamaStoreError(f"blob {digest} absent du magasin {store}")
    return path.read_bytes()


def load_model(model: str, store: str | Path | None = None) -> OllamaModel:
    root = default_store(store)
    manifest = manifest_path(root, model)
    if not manifest.is_file():
        raise OllamaStoreError(f"manifeste de {model} introuvable : {manifest}")
    data = json.loads(manifest.read_text("utf-8"))
    config = json.loads(_read_blob(root, data["config"]["digest"]))
    layers = {layer["mediaType"].removeprefix(MEDIA_PREFIX): layer for layer in data["layers"]}
    if "model" not in layers:
        raise OllamaStoreError(f"{model} : pas de couche de modèle dans le manifeste")
    digest = layers["model"]["digest"]
    model_blob = blob_path(root, digest)
    if not model_blob.is_file():
        raise OllamaStoreError(f"blob GGUF {digest} absent du magasin {root}")

    def text(kind: str) -> str | None:
        return _read_blob(root, layers[kind]["digest"]).decode("utf-8") if kind in layers else None

    params = json.loads(text("params")) if "params" in layers else {}
    return OllamaModel(
        model=model,
        store=root,
        manifest=manifest,
        model_blob=model_blob,
        model_digest=digest,
        config=config,
        params=params,
        license=text("license"),
        template=text("template"),
        system=text("system"),
    )
