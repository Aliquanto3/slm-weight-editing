"""Lecture des octets bruts d'un checkpoint safetensors (fichier ou dossier), sans torch.

Format : 8 octets (u64 petit-boutiste) de longueur d'en-tête, en-tête JSON, puis les données ;
les `data_offsets` de chaque tenseur sont relatifs au début des données.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path


class SafetensorsError(ValueError):
    """Checkpoint illisible ou tenseur introuvable."""


@dataclass(frozen=True)
class SafeTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    path: Path
    start: int  # absolu dans le fichier
    end: int

    def read_bytes(self) -> bytes:
        with self.path.open("rb") as f:
            f.seek(self.start)
            data = f.read(self.end - self.start)
        if len(data) != self.end - self.start:
            raise SafetensorsError(f"{self.name} : fichier tronqué")
        return data


def _read_header(path: Path) -> dict[str, SafeTensor]:
    with path.open("rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise SafetensorsError(f"{path} : en-tête safetensors absent")
        (n,) = struct.unpack("<Q", head)
        header = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        out[name] = SafeTensor(
            name, entry["dtype"], tuple(entry["shape"]), path, base + start, base + end
        )
    return out


def open_checkpoint(path: str | Path) -> dict[str, SafeTensor]:
    """Index des tenseurs d'un fichier `.safetensors` ou de tous ceux d'un dossier."""
    path = Path(path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.safetensors"))
    else:
        raise SafetensorsError(f"checkpoint introuvable : {path}")
    if not files:
        raise SafetensorsError(f"aucun fichier .safetensors dans {path}")
    index: dict[str, SafeTensor] = {}
    for file in files:
        for name, tensor in _read_header(file).items():
            if name in index:
                raise SafetensorsError(f"tenseur {name} présent dans deux fichiers")
            index[name] = tensor
    return index
