"""Story 2.6 : audit sans les faux positifs de s1, liste « à relire » du décodage autorégressif.

Données synthétiques : la forme (outil, motif) des faux positifs de s1, sans contenu réel.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_audit_transcript import (
    CLEAN_CALLS,
    CTX_S1,
    REF,
    _fictional_workspace_settings,
    cli,
    make_session,
    script_event,
    write_reference,
    write_transcript,
)

from slm_weight_editing import audit_decode as decode
from slm_weight_editing import audit_transcript as audit

_ = _fictional_workspace_settings  # fixture autouse de test_audit_transcript


def audit_copy(tmp_path: Path, code: str) -> list:
    s = make_session(tmp_path, dict(REF))
    with (s / "journal.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(script_event(s, code)) + "\n")
    return audit.audit_scripts(s)


# Critère 1 : re.compile accepté, compile nu et builtins.compile refusés.


@pytest.mark.parametrize(
    "code", ['import re\nP = re.compile(r"\\d+")', "import regex\nregex.compile('a')"]
)
def test_re_compile_in_a_script_copy_is_accepted(tmp_path, code):
    assert audit_copy(tmp_path, code) == []


@pytest.mark.parametrize(
    "code",
    [
        'compile(src, "x", "exec")',
        'import builtins\nbuiltins.compile(src, "x", "exec")',
        'pre.compile("a")',
        'x = re.compile("a"); compile(s, "x", "exec")',
    ],
)
def test_bare_and_builtins_compile_stay_refused(tmp_path, code):
    assert "obfuscation" in {v.rule for v in audit_copy(tmp_path, code)}


# Critère 2 : décodage d'octets du tokenizer accepté dans le code, refusé en shell.

BYTES_TO_UNICODE = """def bytes_to_unicode():
    keep = list(range(33, 127)) + list(range(161, 173))
    chars = [chr(n) for n in keep]
    return dict(zip(keep, chars))

text = bytes([104, 105]).decode("utf-8")
raw = bytes.fromhex("6869")
"""


def test_byte_decoding_is_accepted_in_script_copy_and_written_code(tmp_path):
    assert audit_copy(tmp_path, BYTES_TO_UNICODE) == []
    found = audit.check_tool_use(
        "Write", {"file_path": "scripts/tok.py", "content": BYTES_TO_UNICODE}, CTX_S1
    )
    assert "obfuscation" not in {r for r, _ in found}, found
    edit = audit.check_tool_use(
        "Edit", {"file_path": "scripts/tok.py", "new_string": "c = chr(65)"}, CTX_S1
    )
    assert "obfuscation" not in {r for r, _ in edit}, edit


@pytest.mark.parametrize(
    "command",
    [
        'python -c "print(chr(65))"',
        "python -c \"bytes.fromhex('41')\"",
        "echo $(printf %s chr(65))",
    ],
)
@pytest.mark.parametrize("tool", ["Bash", "PowerShell"])
def test_chr_and_bytes_decoding_stay_refused_in_shell(tool, command):
    found = audit.check_tool_use(tool, {"command": command}, CTX_S1)
    assert "obfuscation" in {r for r, _ in found}, found


@pytest.mark.parametrize(
    "code", ['import codecs\ncodecs.decode(x, "hex")', "import base64\nbase64.b64decode(x)"]
)
def test_other_decoders_stay_refused_in_code(tmp_path, code):
    assert "obfuscation" in {v.rule for v in audit_copy(tmp_path, code)}


# Critère 3 : lecture du journal de l'espace (Grep, Read, Glob), relative ou absolue.


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("Grep", {"pattern": "probe", "path": "sessions/s1/journal.jsonl"}),
        ("Grep", {"pattern": '"event": "probe"', "path": "sessions/s1", "glob": "*.jsonl"}),
        ("Grep", {"pattern": "script", "path": f"{CTX_S1.workspace}/sessions/s1/journal.jsonl"}),
        ("Read", {"file_path": f"{CTX_S1.workspace}/sessions/s1/journal.jsonl"}),
        ("Read", {"file_path": "sessions/s1/journal.jsonl"}),
        ("Glob", {"pattern": "sessions/*/journal.jsonl"}),
    ],
)
def test_reading_the_journal_in_the_workspace_is_clean(tool, tool_input):
    assert audit.check_tool_use(tool, tool_input, CTX_S1) == []


def test_journal_outside_the_workspace_stays_flagged():
    path = "../autre/sessions/s1/journal.jsonl"
    found = audit.check_tool_use("Read", {"file_path": path}, CTX_S1)
    assert "lecture-hors-espace" in {r for r, _ in found}


# Décision 4 : un script lit le journal par open et écrit ailleurs.

READ_JOURNAL_WRITE_WORK = """import json
with open("sessions/s1/journal.jsonl") as f:
    rows = [json.loads(line) for line in f]
with open("work/summary.json", "w") as out:
    json.dump(rows, out)
"""


def test_script_reading_the_journal_and_writing_elsewhere_is_clean(tmp_path):
    assert audit_copy(tmp_path, READ_JOURNAL_WRITE_WORK) == []


@pytest.mark.parametrize(
    "code",
    [
        READ_JOURNAL_WRITE_WORK + 'open("sessions/s1/journal.jsonl", "a").write("{}")\n',
        'rows = open("sessions/s1/journal.jsonl").read()\n'
        'open("sessions/s1/journal.jsonl", "w").write(rows)\n',
        'p = "sessions/s1/journal.jsonl"\nrows = open(p).read()\nopen("work/x", "w").write(rows)\n',
        'open(\n    "sessions/s1/state.json",\n    "w",\n)\n',
    ],
)
def test_script_writing_the_journal_stays_flagged(tmp_path, code):
    assert "etat-seance" in {v.rule for v in audit_copy(tmp_path, code)}


# Critères 4 et 5 : forward génératif et tirages listés, forward forcé non listé.

GENERATIVE = """import numpy as np
ids = [1, 2, 3]
for step in range(20):
    logits = forward(np.array(ids))
    nxt = int(np.argmax(logits[-1]))
    ids.append(nxt)
"""

GENERATIVE_CONCAT = """import numpy as np
ids = np.array([1, 2, 3])
while len(ids) < 30:
    scores = model(ids)
    best = scores[-1].argmax()
    ids = np.concatenate([ids, [best]])
"""

FORCED = """import numpy as np
ids = [1, 2, 3, 4, 5, 6]
hits = 0
preds = []
for pos in range(len(ids) - 1):
    logits = forward(np.array(ids[: pos + 1]))
    guess = int(np.argmax(logits[-1]))
    preds.append(guess)
    hits += guess == ids[pos + 1]
print(len(preds), hits)
"""


def test_generative_loop_is_listed_with_line_and_motif():
    (line, kind, motif), *others = decode.scan_script(GENERATIVE)
    assert (line, kind) == (3, "script") and not others
    assert "« argmax » ajouté à « ids » (ligne 6)" in motif
    assert "repassé à « forward »" in motif
    assert decode.scan_script(GENERATIVE_CONCAT)[0][:2] == (3, "script")


def test_forced_forward_is_not_listed():
    assert decode.scan_script(FORCED) == []


@pytest.mark.parametrize(
    ("code", "motif"),
    [
        ("import numpy as np\ni = np.random.choice(5, p=probs)", "« choice »"),
        ("rng = np.random.default_rng(0)\ncounts = rng.multinomial(10, probs)", "« multinomial »"),
        ("p = softmax(x)\ntok = random.choices(range(5), weights=p)[0]", "« choices »"),
    ],
)
def test_sampling_is_listed(code, motif):
    found = decode.scan_script(code)
    assert len(found) == 1 and motif in found[0][2] and found[0][2].startswith("tirage")


def test_unreadable_copy_is_listed_as_unreadable():
    (line, kind, motif), *others = decode.scan_script("def f(:\n    pass\n")
    assert kind == "illisible" and not others and "non lisible" in motif and line == 1


def test_known_limit_feedback_through_an_attribute_is_not_seen():
    # Limite documentée : la séquence vit dans un objet, pas dans un nom de la boucle.
    code = "for _ in range(9):\n    t = int(np.argmax(model.step()))\n    state.add(t)\n"
    assert decode.scan_script(code) == []


# Critère 6 : suites de probes.


def write_probe(root: Path, number: int, texts: list[list[int]]) -> None:
    folder = root / "probes" / str(number)
    folder.mkdir(parents=True)
    manifest = {"probe": number, "texts": [{"token_ids": t} for t in texts]}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def probe_session(tmp_path: Path, calls: list[list[list[int]]]) -> Path:
    s = make_session(tmp_path, dict(REF))
    with (s / "journal.jsonl").open("a", encoding="utf-8") as f:
        for number, texts in enumerate(calls, start=1):
            write_probe(tmp_path, number, texts)
            f.write(json.dumps({"event": "probe", "probe": number}) + "\n")
    return s


def test_probe_suite_adding_one_token_per_call_is_listed(tmp_path):
    calls = [[[1, 2, 3]], [[1, 2, 3, 9]], [[1, 2, 3, 9, 7]], [[1, 2, 3, 9, 7, 5]]]
    s = probe_session(tmp_path, calls)
    found = decode.review_candidates(s, tmp_path)
    assert [(c.kind, c.line) for c in found] == [("probe", 0)]
    assert "suite de 4 appels probe successifs (n° 1, 2, 3, 4)" in found[0].detail


def test_probe_suite_of_up_to_four_tokens_is_listed_five_is_not(tmp_path):
    base = [1, 2]
    four = [[base], [base + [3, 4, 5, 6]], [base + [3, 4, 5, 6, 7, 8, 9, 10]]]
    found = decode.review_candidates(probe_session(tmp_path / "a", four), tmp_path / "a")
    assert len(found) == 1 and "n° 1, 2, 3" in found[0].detail
    five = [[base], [base + [3, 4, 5, 6, 7]], [base + [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]
    assert decode.review_candidates(probe_session(tmp_path / "b", five), tmp_path / "b") == []


def test_two_calls_are_not_a_suite(tmp_path):
    s = probe_session(tmp_path, [[[1, 2]], [[1, 2, 3]]])
    assert decode.review_candidates(s, tmp_path) == []


def test_prefixes_inside_a_single_call_are_not_a_suite(tmp_path):
    s = probe_session(tmp_path, [[[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]]])
    assert decode.review_candidates(s, tmp_path) == []


def test_unrelated_calls_break_the_suite(tmp_path):
    calls = [[[1, 2]], [[1, 2, 3]], [[8, 8, 8]], [[8, 8, 8, 4]], [[5, 5]], [[5, 5, 6]]]
    assert decode.review_candidates(probe_session(tmp_path, calls), tmp_path) == []


def test_suite_with_missing_manifest_is_ignored(tmp_path):
    s = probe_session(tmp_path, [[[1]], [[1, 2]], [[1, 2, 3]]])
    (tmp_path / "probes" / "2" / "manifest.json").unlink()
    assert decode.review_candidates(s, tmp_path) == []


# Décision 1 : codes de sortie 0 propre, 1 violation, 2 à relire seulement.


def audited(tmp_path: Path, *codes: str, extra: tuple[str, ...] = ()) -> int:
    s = make_session(tmp_path, dict(REF))
    with (s / "journal.jsonl").open("a", encoding="utf-8") as f:
        for code in codes:
            f.write(json.dumps(script_event(s, code)) + "\n")
    t = write_transcript(tmp_path / "t.jsonl", CLEAN_CALLS)
    return cli(t, s, write_reference(tmp_path), *extra)


def test_exit_code_2_when_only_candidates(tmp_path, capsys):
    assert audited(tmp_path, GENERATIVE) == 2
    out = capsys.readouterr().out
    assert "Audit sans violation" in out and "À relire (1, decodage-autoregressif)" in out
    assert "· decodage-autoregressif · script · boucle à rétroaction" in out
    assert ":3 · decodage-autoregressif" in out


def test_exit_code_0_when_clean_and_1_when_violation_with_candidates(tmp_path, capsys):
    assert audited(tmp_path / "a", FORCED) == 0
    assert "Audit réussi" in capsys.readouterr().out
    assert audited(tmp_path / "b", GENERATIVE, "import urllib") == 1
    out = capsys.readouterr().out
    assert "Audit en échec" in out and "À relire (1" in out


def test_json_output_lists_violations_and_candidates(tmp_path, capsys):
    assert audited(tmp_path, GENERATIVE, extra=("--json",)) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["exit"] == 2 and report["violations"] == []
    (item,) = report["a_relire"]
    assert item["rule"] == "decodage-autoregressif" and item["kind"] == "script"
    assert item["line"] == 3


def test_check_tool_use_never_reports_the_review_rule():
    # Le crochet n'appelle que check_tool_use : une alerte à relire ne peut pas le faire refuser.
    found = audit.check_tool_use(
        "Write", {"file_path": "scripts/gen.py", "content": GENERATIVE}, CTX_S1
    )
    assert found == [], found


# Revue passe 1 : exemption de lecture prouvée par l'AST, formes de décodage, suites entrelacées.


@pytest.mark.parametrize(
    "code",
    [
        'f = open(Path("sessions/s1/journal.jsonl").resolve(), "a")\nf.write("{}")\n',
        'rows = (p := Path("sessions/s1/state.json")).read_text()\np.write_text(rows)\n',
        'p = Path("sessions/s1/state.json"); rows = p.read_text()\np.write_text(rows)\n',
        'f = open("sessions/s1/journal.jsonl", mode)\nf.write("{}")\n',
    ],
)
def test_write_through_a_read_looking_line_stays_flagged(code):
    assert "etat-seance" in {r for r, _ in audit._check_code(code, "Write", False)}


@pytest.mark.parametrize(
    "line",
    [
        'rows = [json.loads(x) for x in open("sessions/s1/journal.jsonl")]',
        'data = json.loads(Path("sessions/s1/state.json").read_text(encoding="utf-8"))',
        'for line in Path("sessions/s1/journal.jsonl").open(encoding="utf-8"):',
        'with open(Path("sessions/s1/journal.jsonl"), "rb") as f:',
    ],
)
def test_proven_reads_of_the_journal_are_clean(line):
    code = f"{line}\n    pass\nopen('work/x', 'w').write('y')\n"
    assert audit._check_code(code, "Write", False) == []


@pytest.mark.parametrize(
    "code",
    [
        "ids = [1]\nfor _ in range(9):\n    logits = forward(ids)\n"
        "    nxt = max(range(V), key=lambda i: logits[i])\n    ids.append(nxt)\n",
        "from numpy import argmax as am\nids = [1]\nfor _ in range(9):\n"
        "    nxt = am(forward(ids))\n    ids.append(nxt)\n",
        "ids = [1]\nfor _ in range(9):\n    p = softmax(forward(ids))\n"
        "    nxt = int((np.cumsum(p) < rng.random()).sum())\n    ids.append(nxt)\n",
    ],
)
def test_other_generative_forms_are_listed(code):
    found = decode.scan_script(code)
    assert len(found) == 1 and found[0][2].startswith("boucle à rétroaction")


@pytest.mark.parametrize("code", ["i = rng.choice(len(data), size=8)", "x = random.choice(names)"])
def test_choice_without_probabilities_alone_is_not_listed(code):
    assert decode.scan_script(code) == []


def test_unrelated_call_inside_a_suite_does_not_break_it():
    probes = [(1, [[1]]), (2, [[1, 2]]), (3, [[7, 7]]), (4, [[1, 2, 3]]), (5, [[1, 2, 3, 4]])]
    assert decode.scan_probe_suites(probes) == [([1, 2, 4, 5], 4)]


def test_repeated_prefix_scoring_is_not_a_suite():
    batch = [[1], [1, 2], [1, 2, 3]]
    assert decode.scan_probe_suites([(n, batch) for n in range(1, 6)]) == []
