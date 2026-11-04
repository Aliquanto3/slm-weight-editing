import json
import math

import numpy as np
import pytest
from conftest import to_bf16, write_safetensors

import slm_weight_editing
from slm_weight_editing.compare_edits import (
    CompareError,
    compare_deltas,
    newcombe_diff,
    principal_directions,
    read_summary,
    wilson,
)

W1, W2, BIAS = "l.w1", "l.w2", "l.bias"


def base_tensors():
    rng = np.random.default_rng(0)
    return {
        W1: rng.normal(size=(4, 6)).astype(np.float32),
        W2: rng.normal(size=(3, 5)).astype(np.float32),
        BIAS: rng.normal(size=(4,)).astype(np.float32),
    }


def edited(base, changes):
    out = {k: v.copy() for k, v in base.items()}
    for (name, idx), val in changes.items():
        out[name][idx] += val
    return out


def make(tmp_path, a_changes, b_changes):
    base = base_tensors()
    paths = {}
    for key, tensors in (
        ("orig", base),
        ("a", edited(base, a_changes)),
        ("b", edited(base, b_changes)),
    ):
        paths[key] = write_safetensors(tmp_path / f"{key}.safetensors", tensors, dtype="F32")
    return paths


def test_cosine_and_counts(tmp_path):
    # A et B touchent W1 : même motif rang 1 (u v^T), B avec le signe opposé sur une moitié.
    u, v = np.array([1.0, 2, 0, 0]), np.array([1.0, 0, 1, 0, 0, 0])
    base = base_tensors()
    a_w1 = base[W1] + np.outer(u, v).astype(np.float32)
    b_w1 = base[W1] - 3 * np.outer(u, v).astype(np.float32)
    a = dict(base, **{W1: a_w1, W2: edited(base, {(W2, (0, 0)): 1.0})[W2]})
    b = dict(base, **{W1: b_w1, BIAS: edited(base, {(BIAS, (1,)): 2.0})[BIAS]})
    paths = {}
    for key, tensors in (("o", base), ("a", a), ("b", b)):
        paths[key] = write_safetensors(tmp_path / f"{key}.safetensors", tensors, dtype="F32")
    rc = slm_weight_editing.main(
        [
            "compare-edits",
            "--original", str(paths["o"]),
            "--edit-a", str(paths["a"]),
            "--edit-b", str(paths["b"]),
            "--label-a", "Claude",
            "--label-b", "MIT",
            "--out", str(tmp_path / "out" / "rapport.md"),
        ]
    )  # fmt: skip
    assert rc == 0
    res = json.loads((tmp_path / "out" / "rapport.json").read_text(encoding="utf-8"))
    w = res["weights"]
    assert w["total_weights"] == 24 + 15 + 4
    assert w["a"]["tensors"] == {W1: 4, W2: 1}
    assert w["b"]["tensors"] == {W1: 4, BIAS: 1}
    assert w["a"]["modified_weights"] == 5
    assert w["a"]["model_share"] == pytest.approx(5 / 43)
    assert w["only_a"] == [W2] and w["only_b"] == [BIAS]
    (common,) = w["common"]
    assert common["name"] == W1
    assert common["flat_cosine"] == pytest.approx(-1.0, abs=1e-6)
    # Mêmes directions principales, signe arbitraire : valeur absolue = 1.
    assert common["left_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert common["right_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert common["note"] is None
    md = (tmp_path / "out" / "rapport.md").read_text(encoding="utf-8")
    assert "| Claude | 2 | 5 |" in md and "| MIT | 2 | 5 |" in md
    assert "Tenseurs propres à Claude : 1." in md
    assert f"- {BIAS} : 1 poids" in md
    assert "## Banc" not in md  # sans summary.json : poids seulement
    assert str(tmp_path) not in md


def test_orthogonal_deltas(tmp_path):
    paths = make(tmp_path, {(W1, (0, 0)): 1.0}, {(W1, (1, 1)): 1.0})
    from slm_weight_editing.compare_edits import compare

    res, _ = compare(paths["orig"], paths["a"], paths["b"])
    (c,) = res["weights"]["common"]
    assert c["flat_cosine"] == pytest.approx(0.0, abs=1e-9)
    assert c["left_cosine"] == pytest.approx(0.0, abs=1e-9)
    assert c["right_cosine"] == pytest.approx(0.0, abs=1e-9)


def test_zero_delta_flagged_without_division():
    z = np.zeros((3, 3))
    d = np.eye(3)
    res = compare_deltas({"t": z}, {"t": d}, 9)
    (c,) = res["common"]
    assert c["flat_cosine"] is None and c["left_cosine"] is None
    assert "rang 0 pour A" in c["note"]
    assert principal_directions(z) is None
    both = compare_deltas({"t": z}, {"t": z}, 9)["common"][0]
    assert "A et B" in both["note"]


def test_bf16_and_f16_deltas(tmp_path):
    base = {"t": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
    ed = {"t": np.array([[1.5, 2.0], [3.0, 4.0]], dtype=np.float32)}
    p = {}
    for key, tensors in (("o", base), ("a", ed)):
        p[key] = write_safetensors(
            tmp_path / f"{key}.safetensors", {n: to_bf16(a) for n, a in tensors.items()}
        )
    from slm_weight_editing.compare_edits import compare_checkpoints

    res = compare_checkpoints(p["o"], p["a"], p["a"])
    assert res["a"]["tensors"] == {"t": 1}
    # F16
    for key, tensors in (("o", base), ("a", ed)):
        p[key] = write_safetensors(
            tmp_path / f"{key}16.safetensors",
            {n: a.astype("<f2") for n, a in tensors.items()},
            dtype="F16",
        )
    assert compare_checkpoints(p["o"], p["a"], p["a"])["common"][0]["flat_cosine"] == 1.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda t: {k: v for k, v in t.items() if k != BIAS}, "noms de tenseurs"),
        (lambda t: {**t, W1: np.zeros((6, 4), np.float32)}, "forme"),
        (lambda t: {**t, W1: t[W1].astype("<f2")}, "dtype"),
    ],
)
def test_incompatible_checkpoints(tmp_path, capsys, mutate, message):
    base = base_tensors()
    o = write_safetensors(tmp_path / "o.safetensors", base, dtype="F32")
    good = write_safetensors(tmp_path / "g.safetensors", base, dtype="F32")
    bad_t = mutate(base)
    if message == "dtype":
        # W1 en F16, le reste en F32 : écrit à la main via deux fichiers.
        bad = write_safetensors(tmp_path / "bad.safetensors", {W1: bad_t[W1]}, dtype="F16")
        rest = write_safetensors(
            tmp_path / "rest.safetensors", {k: v for k, v in base.items() if k != W1}, dtype="F32"
        )
        bad = tmp_path / "dir"
        bad.mkdir()
        (tmp_path / "bad.safetensors").replace(bad / "a.safetensors")
        rest.replace(bad / "b.safetensors")
    else:
        bad = write_safetensors(tmp_path / "bad.safetensors", bad_t, dtype="F32")
    rc = slm_weight_editing.main(
        ["compare-edits", "--original", str(o), "--edit-a", str(good), "--edit-b", str(bad)]
        + ["--out", str(tmp_path / "r.md")]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("Erreur : édition B") and message in err
    assert not (tmp_path / "r.md").exists()


def test_wilson_and_newcombe_values():
    p, lo, hi = wilson(5, 10)
    assert (p, round(lo, 4), round(hi, 4)) == (0.5, 0.2366, 0.7634)
    one, two = wilson(81, 263), wilson(48, 256)
    d, lo, hi = newcombe_diff(one, two)
    assert d == pytest.approx(81 / 263 - 48 / 256)
    assert lo == pytest.approx(d - math.hypot(one[0] - one[1], two[2] - two[0]))
    assert hi == pytest.approx(d + math.hypot(one[2] - one[0], two[0] - two[1]))
    assert 0 < lo < d < hi < 0.25
    assert wilson(0, 10)[1] == 0.0
    with pytest.raises(CompareError):
        wilson(0, 0)


def summary(models, extra_text=True):
    """summary.json qb.report/1 synthétique, avec des champs texte à ne pas reprendre."""

    def entry(tag, mean, lo, hi, graded, trunc):
        e = {
            "model": tag, "mean": mean, "n": 100, "ci_low": lo, "ci_high": hi,
            "graded": graded, "truncated": trunc,
        }  # fmt: skip
        if extra_text:
            e["question"] = "QUESTION-SECRETE"
            e["answer"] = "REPONSE-SECRETE"
        return e

    def scope(usage, idx):
        return {
            "usage": usage,
            "models": [
                entry(tag, m + 0.0, m - 0.05, m + 0.05, 100, int(100 * t))
                for tag, (m, t) in models.items()
            ][: len(models)],
            "note": "NOTE-SECRETE",
        }

    s = {
        "schema": "qb.report/1",
        "campaign": "C:/Users/prive/campagne",
        "tracks": [
            {"track": "chat", "scopes": [scope(None, 0), scope("redaction", 1)]},
            {"track": "rag", "scopes": [scope(None, 2)]},
        ],
    }
    return s


def test_read_summary_selects_model_and_keys(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps(summary({"a:1": (0.5, 0.1), "b:1": (0.6, 0.3)})), encoding="utf-8")
    got = read_summary(path, "b:1")
    assert set(got) == {("chat", "global"), ("chat", "redaction"), ("rag", "global")}
    assert got[("chat", "global")]["mean"] == 0.6 and got[("chat", "global")]["truncated"] == 30
    with pytest.raises(CompareError, match="plusieurs modèles"):
        read_summary(path, None)
    with pytest.raises(CompareError, match="absent"):
        read_summary(path, "zzz")
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema": "autre"}', encoding="utf-8")
    with pytest.raises(CompareError, match="qb.report/1"):
        read_summary(bad, None)


def test_bench_section_values_and_no_leak(tmp_path):
    paths = make(tmp_path, {(W1, (0, 0)): 1.0}, {(W1, (0, 0)): 2.0})
    s1 = tmp_path / "s1.json"
    s2 = tmp_path / "s2.json"
    s1.write_text(json.dumps(summary({"edit-a": (0.5, 0.10)})), encoding="utf-8")
    s2.write_text(json.dumps(summary({"edit-b": (0.4, 0.30)})), encoding="utf-8")
    out = tmp_path / "rapport.md"
    rc = slm_weight_editing.main(
        ["compare-edits", "--original", str(paths["orig"]), "--edit-a", str(paths["a"])]
        + ["--edit-b", str(paths["b"]), "--summary", str(s1), "--summary", str(s2)]
        + ["--out", str(out)]
    )
    assert rc == 0
    md = out.read_text(encoding="utf-8")
    js = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    for text in (md, json.dumps(js)):
        for secret in ("SECRETE", "prive", "campagne", str(tmp_path)):
            assert secret not in text
    rows = {(r["track"], r["category"]): r for r in js["bench"]}
    assert set(rows) == {("chat", "global"), ("chat", "redaction"), ("rag", "global")}
    sc = rows[("chat", "global")]["score"]
    assert sc["diff"] == pytest.approx(0.1)
    assert sc["ci_low"] == pytest.approx(0.1 - math.hypot(0.05, 0.05))
    assert sc["ci_high"] == pytest.approx(0.1 + math.hypot(0.05, 0.05))
    tr = rows[("chat", "global")]["truncation"]
    assert tr["a"] == 0.10 and tr["b"] == 0.30
    assert tr["diff"] == pytest.approx(-0.2)
    assert tr["ci_low"] < -0.2 < tr["ci_high"] < 0
    assert "## Banc" in md and "| chat | redaction |" in md
    assert "-0.200" not in md and "-20.0 pt" in md


def test_single_summary_with_two_tags(tmp_path):
    paths = make(tmp_path, {(W1, (0, 0)): 1.0}, {(W1, (0, 0)): 2.0})
    s = tmp_path / "s.json"
    s.write_text(json.dumps(summary({"x": (0.5, 0.1), "y": (0.7, 0.2)})), encoding="utf-8")
    from slm_weight_editing.compare_edits import compare

    res, _ = compare(paths["orig"], paths["a"], paths["b"], summaries=[s], tags=("x", "y"))
    r = next(r for r in res["bench"] if (r["track"], r["category"]) == ("rag", "global"))
    assert r["score"]["diff"] == pytest.approx(-0.2)
    with pytest.raises(CompareError, match="au plus deux"):
        compare(paths["orig"], paths["a"], paths["b"], summaries=[s, s, s])
