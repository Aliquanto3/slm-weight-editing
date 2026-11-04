"""Lecture minimale d'un fichier GGUF (v2 et v3, petit-boutiste) : métadonnées et tenseurs.

Le lecteur ne décode que ce qu'il faut pour situer les octets de chaque tenseur ; la réécriture
copie le fichier d'origine et ne remplace que des plages d'octets, pour que tout le reste
(métadonnées, ordre, alignement, autres tenseurs) reste identique octet pour octet.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass, field
from pathlib import Path

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# Types ggml utiles ici : (nom, éléments par bloc, octets par bloc).
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    8: ("Q8_0", 32, 34),
    30: ("BF16", 1, 2),
}

# Types des valeurs de métadonnées : taille fixe en octets (chaînes et tableaux à part).
_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_SCALAR_FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q"}
_SCALAR_FORMATS |= {11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9


class GGUFError(ValueError):
    """Fichier GGUF illisible ou tenseur introuvable."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: tuple[int, ...]  # ordre ggml : dims[0] est la dimension contiguë (ne0)
    ggml_type: int
    offset: int  # absolu, depuis le début du fichier

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.ggml_type, (f"type {self.ggml_type}", 0, 0))[0]

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def n_bytes(self) -> int:
        if self.ggml_type not in GGML_TYPES:
            raise GGUFError(f"{self.name} : type ggml {self.ggml_type} non pris en charge")
        _, block, size = GGML_TYPES[self.ggml_type]
        if self.dims[0] % block:
            raise GGUFError(f"{self.name} : ne0={self.dims[0]} non multiple du bloc {block}")
        return self.n_elements // block * size


@dataclass
class GGUFFile:
    path: Path
    version: int
    alignment: int
    data_offset: int
    metadata: dict[str, object] = field(default_factory=dict)
    tensors: dict[str, TensorInfo] = field(default_factory=dict)

    def tensor(self, name: str) -> TensorInfo:
        if name not in self.tensors:
            raise GGUFError(f"tenseur {name} absent de {self.path}")
        return self.tensors[name]

    def read_tensor_bytes(self, name: str) -> bytes:
        info = self.tensor(name)
        with self.path.open("rb") as f:
            f.seek(info.offset)
            data = f.read(info.n_bytes)
        if len(data) != info.n_bytes:
            raise GGUFError(f"{name} : fichier tronqué")
        return data


class _Cursor:
    def __init__(self, buf: mmap.mmap) -> None:
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise GGUFError("en-tête GGUF tronqué")
        out = self.buf[self.pos : self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt: str):
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.take(size))[0]

    def string(self) -> str:
        return self.take(self.unpack("Q")).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            itype, count = self.unpack("I"), self.unpack("Q")
            if itype in _SCALAR_SIZES:  # tableau scalaire : sauté, seule sa taille compte
                self.take(_SCALAR_SIZES[itype] * count)
                return f"<tableau de {count}>"
            for _ in range(count):
                self.value(itype)
            return f"<tableau de {count}>"
        if vtype in _SCALAR_FORMATS:
            return self.unpack(_SCALAR_FORMATS[vtype])
        raise GGUFError(f"type de métadonnée inconnu : {vtype}")


def read_gguf(path: str | Path) -> GGUFFile:
    """Lit l'en-tête d'un GGUF, sans charger les tenseurs."""
    path = Path(path)
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        cur = _Cursor(buf)
        if cur.take(4) != GGUF_MAGIC:
            raise GGUFError(f"{path} n'est pas un fichier GGUF")
        version = cur.unpack("I")
        if version not in (2, 3):
            raise GGUFError(f"version GGUF {version} non prise en charge")
        n_tensors, n_kv = cur.unpack("Q"), cur.unpack("Q")
        metadata: dict[str, object] = {}
        for _ in range(n_kv):
            key = cur.string()
            metadata[key] = cur.value(cur.unpack("I"))
        raw: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(n_tensors):
            name = cur.string()
            n_dims = cur.unpack("I")
            dims = tuple(cur.unpack("Q") for _ in range(n_dims))
            raw.append((name, dims, cur.unpack("I"), cur.unpack("Q")))
        alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT))
        data_offset = -(-cur.pos // alignment) * alignment
        size = len(buf)
    tensors = {}
    for name, dims, ggml_type, rel in raw:
        info = TensorInfo(name, dims, ggml_type, data_offset + rel)
        if ggml_type in GGML_TYPES and info.offset + info.n_bytes > size:
            raise GGUFError(f"{name} déborde de la fin du fichier")
        tensors[name] = info
    return GGUFFile(path, version, alignment, data_offset, metadata, tensors)
