"""Quantification Q8_0 à la manière de ggml (`quantize_row_q8_0_ref`), en numpy.

Bloc de 32 valeurs : échelle d = amax / 127 (calculée en float32, stockée en float16), puis
q = roundf(x * (1 / d)), arrondi du demi loin de zéro. Un bloc nul donne d = 0 et q = 0.
Disposition d'un bloc : 2 octets d'échelle float16, puis 32 octets int8.
"""

from __future__ import annotations

import numpy as np

QK8_0 = 32
BLOCK_BYTES = 2 + QK8_0
_BLOCK_DTYPE = np.dtype([("d", "<f2"), ("qs", "i1", (QK8_0,))])

# Passage intermédiaire avant quantification : float32 direct (llama.cpp) ou float16 (un export
# F16 puis quantifié). Le contrôle de fidélité dit si le passage choisi redonne les octets de
# l'officiel ; sur des poids réalistes, les deux passages donnent presque toujours les mêmes
# octets, et le contrôle ne les départage donc pas.
INTERMEDIATES = ("f32", "f16")


def bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    """Convertit des octets bfloat16 bruts (petit-boutiste) en float32 exacts."""
    if len(raw) % 2:
        raise ValueError("longueur impaire pour des octets bfloat16")
    u16 = np.frombuffer(raw, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _roundf(x: np.ndarray) -> np.ndarray:
    """roundf de C : demi loin de zéro (np.round arrondit au pair).

    Pas de floor(|x| + 0.5) : pour 0.49999997, la somme s'arrondit à 1 en float32.
    """
    a = np.abs(x)
    whole = np.floor(a)
    return np.sign(x) * (whole + np.floor(2 * (a - whole)))


def quantize_q8_0(values: np.ndarray, intermediate: str = "f32") -> bytes:
    """Quantifie un tableau float32 (taille multiple de 32, ordre mémoire ggml) en Q8_0."""
    if intermediate not in INTERMEDIATES:
        raise ValueError(f"passage intermédiaire inconnu : {intermediate}")
    x = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if x.size % QK8_0:
        raise ValueError(f"{x.size} valeurs : pas un multiple de {QK8_0}")
    if intermediate == "f16":
        x = x.astype(np.float16).astype(np.float32)
    blocks = x.reshape(-1, QK8_0)
    amax = np.abs(blocks).max(axis=1)
    d = amax / np.float32(127)
    with np.errstate(divide="ignore"):
        inv = np.where(d == 0, np.float32(0), np.float32(1) / d).astype(np.float32)
    q = _roundf(blocks * inv[:, None])
    out = np.empty(blocks.shape[0], dtype=_BLOCK_DTYPE)
    out["d"] = d.astype(np.float16)
    out["qs"] = q.astype(np.int8)
    return out.tobytes()


def split_blocks(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Sépare des octets Q8_0 en (échelles float16, quanta int8 de forme (n, 32))."""
    if len(raw) % BLOCK_BYTES:
        raise ValueError("longueur non multiple d'un bloc Q8_0")
    blocks = np.frombuffer(raw, dtype=_BLOCK_DTYPE)
    return blocks["d"], blocks["qs"]


def dequantize_q8_0(raw: bytes) -> np.ndarray:
    d, qs = split_blocks(raw)
    return (d.astype(np.float32)[:, None] * qs.astype(np.float32)).reshape(-1)
