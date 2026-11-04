import struct

import gguf
import numpy as np
import pytest

from slm_weight_editing.q8_0 import (
    bf16_bytes_to_f32,
    dequantize_q8_0,
    quantize_q8_0,
    split_blocks,
)


def block(scale_bits: int, quanta: list[int]) -> bytes:
    return struct.pack("<H", scale_bits) + bytes(np.array(quanta, np.int8).tobytes())


def test_bf16_conversion_is_exact():
    raw = struct.pack("<4H", 0x3F80, 0xBF80, 0x0000, 0x4049)
    assert bf16_bytes_to_f32(raw).tolist() == [1.0, -1.0, 0.0, 3.140625]


def test_bf16_odd_length_is_refused():
    with pytest.raises(ValueError, match="impaire"):
        bf16_bytes_to_f32(b"\x00")


def test_zero_block_gives_zero_scale_and_quanta():
    assert quantize_q8_0(np.zeros(32, np.float32)) == bytes(34)


def test_half_integers_round_away_from_zero():
    # amax = 127 donne d = 1 : q = roundf(x) exactement, le demi loin de zéro.
    x = np.zeros(32, np.float32)
    x[:7] = [127.0, 0.5, -0.5, 1.5, -2.5, 2.5, 0.49999997]
    want = [127, 1, -1, 2, -3, 3, 0] + [0] * 25
    assert quantize_q8_0(x) == block(0x3C00, want)  # 0x3C00 : 1.0 en float16


def test_scale_is_amax_over_127_in_float16():
    x = np.full(32, 0.01, np.float32)
    x[3] = -0.254
    d, qs = split_blocks(quantize_q8_0(x))
    assert d[0] == np.float16(np.float32(0.254) / np.float32(127))
    assert qs[0, 3] == -127
    assert qs[0, 0] == 5  # 0.01 / 0.002 = 5


@pytest.mark.parametrize("scale", [1e-6, 0.03, 1.0, 300.0])
def test_matches_gguf_py_reference(scale):
    rng = np.random.default_rng(1)
    x = bf16_bytes_to_f32(
        (rng.normal(0, scale, 4096).astype(np.float32).view(np.uint32) >> 16)
        .astype("<u2")
        .tobytes()
    )
    x[:32] = 0.0
    ref = gguf.quants.quantize(x.reshape(64, 64), gguf.GGMLQuantizationType.Q8_0)
    assert quantize_q8_0(x) == ref.tobytes()


def test_roundtrip_requantizes_to_the_same_bytes():
    rng = np.random.default_rng(2)
    raw = quantize_q8_0(rng.normal(0, 0.1, 256).astype(np.float32))
    values = dequantize_q8_0(raw)
    assert values.shape == (256,)
    assert quantize_q8_0(values) == raw


def test_dequantization_error_is_bounded_by_half_a_step_plus_scale_rounding():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.1, 128).astype(np.float32)
    raw = quantize_q8_0(x)
    d = split_blocks(raw)[0].astype(np.float32)
    err = np.abs(dequantize_q8_0(raw) - x).reshape(-1, 32).max(axis=1)
    # d/2 d'arrondi, plus 127 fois l'arrondi float16 de l'échelle (au plus 2^-11 en relatif).
    assert np.all(err <= d * (0.5 + 127 * 2.0**-11))


def test_f16_intermediate_can_change_the_bytes():
    # Valeurs sous-normales en float16 : le passage par f16 les arrondit, pas le passage f32.
    rng = np.random.default_rng(4)
    x = rng.normal(0, 2e-6, 1024).astype(np.float32)
    assert quantize_q8_0(x, "f32") != quantize_q8_0(x, "f16")
    y = np.array([1.0, -0.5] * 16, np.float32)  # exactes en float16 : mêmes octets
    assert quantize_q8_0(y, "f32") == quantize_q8_0(y, "f16")


def test_size_not_multiple_of_32_is_refused():
    with pytest.raises(ValueError, match="multiple de 32"):
        quantize_q8_0(np.ones(33, np.float32))


def test_unknown_intermediate_is_refused():
    with pytest.raises(ValueError, match="inconnu"):
        quantize_q8_0(np.ones(32, np.float32), "bf16")
