"""Serializer 1.1.0 conformance tests.

Pins the exact executable output of `polycodegpt.tokenization.serializer` to the
required formats, checks marker placement at the token-ID level, and exercises
the reversible escaping / deserialization API:

    Python      : <|lang:python|><|docstring|>{docstring}<|code|>{code}<|eos|>
    JavaScript  : <|lang:javascript|><|docstring|>{docstring}<|code|>{code}<|eos|>
    Documentation: <|lang:en|><|text|>{rst}<|eos|>

Fixtures are hand-written; none of them are tokenizer training inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polycodegpt.tokenization.serializer import (  # noqa: E402
    SERIALIZER_VERSION,
    SPECIAL_TOKENS,
    escape_content,
    load_doc_inventory,
    parse_serialized,
    serialize_code_record,
    serialize_documentation,
    serialize_javascript_record,
    serialize_python_record,
    unescape_content,
)

TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "artifacts" / "polycodegpt_tokenizer_v1_1.json"
DOC_INVENTORY = PROJECT_ROOT / "data" / "manifests" / "python_docs_training_corpus_inventory.csv"

PY_RECORD = {
    "language": "python",
    "source_dataset": "CodeSearchNet Python",
    "docstring": "Return the sum of a and b.",
    "code": "def add(a, b):\n    return a + b\n",
}
JS_RECORD = {
    "language": "javascript",
    "source_dataset": "CodeSearchNet JavaScript",
    "docstring": "Return the sum of a and b.",
    "code": "const add = (a, b) => a + b;\n",
}
RST = "Title\n=====\n\n.. note::\n   Keep `code`, directives and   spacing.\n"


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    assert TOKENIZER_PATH.is_file(), f"missing tokenizer artifact: {TOKENIZER_PATH}"
    return Tokenizer.from_file(str(TOKENIZER_PATH))


# ------------------------------------------------------- exact formats ------
def test_python_serialization_exact():
    expected = (
        "<|lang:python|><|docstring|>Return the sum of a and b."
        "<|code|>def add(a, b):\n    return a + b\n<|eos|>"
    )
    assert serialize_python_record(PY_RECORD) == expected
    assert serialize_code_record(PY_RECORD) == expected


def test_javascript_serialization_exact():
    expected = (
        "<|lang:javascript|><|docstring|>Return the sum of a and b."
        "<|code|>const add = (a, b) => a + b;\n<|eos|>"
    )
    assert serialize_javascript_record(JS_RECORD) == expected
    assert serialize_code_record(JS_RECORD) == expected


def test_documentation_serialization_exact():
    assert serialize_documentation(RST) == f"<|lang:en|><|text|>{RST}<|eos|>"


def test_empty_docstring_keeps_both_markers():
    out = serialize_code_record({**PY_RECORD, "docstring": ""})
    assert out == "<|lang:python|><|docstring|><|code|>def add(a, b):\n    return a + b\n<|eos|>"


def test_docstring_equal_to_code_is_not_deduplicated():
    out = serialize_code_record({**PY_RECORD, "docstring": "x", "code": "x"})
    assert out == "<|lang:python|><|docstring|>x<|code|>x<|eos|>"


# ------------------------------------------------------------ markers -------
def test_every_code_marker_present_once():
    for record, lang in ((PY_RECORD, "<|lang:python|>"), (JS_RECORD, "<|lang:javascript|>")):
        out = serialize_code_record(record)
        for marker in (lang, "<|docstring|>", "<|code|>", "<|eos|>"):
            assert out.count(marker) == 1, f"{marker} must appear exactly once in {out[:80]!r}"
        assert out.startswith(lang + "<|docstring|>")
        assert out.endswith("<|eos|>")


def test_marker_ordering_is_fixed():
    for record, lang in ((PY_RECORD, "<|lang:python|>"), (JS_RECORD, "<|lang:javascript|>")):
        out = serialize_code_record(record)
        positions = [out.index(m) for m in (lang, "<|docstring|>", "<|code|>", "<|eos|>")]
        assert positions == sorted(positions), positions


def test_reserved_language_markers_never_emitted():
    for text in (serialize_code_record(PY_RECORD), serialize_code_record(JS_RECORD), serialize_documentation(RST)):
        assert "<|lang:fr|>" not in text
        assert "<|lang:ar|>" not in text
        assert "<|pad|>" not in text


def test_special_token_table_is_fixed():
    assert SERIALIZER_VERSION == "1.1.0"
    assert SPECIAL_TOKENS == (
        "<|pad|>",
        "<|unk|>",
        "<|eos|>",
        "<|code|>",
        "<|docstring|>",
        "<|text|>",
        "<|lang:python|>",
        "<|lang:javascript|>",
        "<|lang:en|>",
        "<|lang:fr|>",
        "<|lang:ar|>",
    )
    assert len(set(SPECIAL_TOKENS)) == 11


def test_inventory_document_is_serialized_verbatim():
    docs = load_doc_inventory(DOC_INVENTORY, PROJECT_ROOT)
    assert len(docs) == 500
    doc = docs[0]
    raw = Path(doc["local_path"]).read_text(encoding="utf-8")
    out = serialize_documentation(raw)
    assert out == f"<|lang:en|><|text|>{escape_content(raw)}<|eos|>"
    assert parse_serialized(out)["text"] == raw
    assert all("/Doc/" in d["rel_path"] for d in docs)
    assert not any(d["rel_path"].endswith("Doc/license.rst") for d in docs)


# ------------------------------------------------------------ escaping ------
@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain",
        "back\\slash",
        "trailing\\",
        "\\\\double\\\\",
        "<|eos|>",
        "a<|code|>b",
        "\\<|eos|>",
        "<\\|eos|>",
        "<|lang:python|><|eos|>",
        "a || b | c",
        "<<|>",
        "<|",
        "|>",
    ],
)
def test_backslash_and_marker_escaping_is_reversible(text):
    escaped = escape_content(text)
    assert unescape_content(escaped) == text
    assert not any(token in escaped for token in SPECIAL_TOKENS), f"control token leaked: {escaped!r}"


def test_escaping_is_injective():
    inputs = ["\\", "\\\\", "<|eos|>", "<\\|eos|>", "a|b", ""]
    escaped = [escape_content(x) for x in inputs]
    assert len(set(escaped)) == len(inputs)


def test_literal_control_token_is_escaped_in_stream():
    record = {**PY_RECORD, "docstring": "says <|eos|> literally", "code": "x = '<|code|>'\n"}
    out = serialize_code_record(record)
    # exactly the four structural markers, nothing injected by content
    assert out.count("<|eos|>") == 1
    assert out.count("<|code|>") == 1
    assert out.count("<|docstring|>") == 1
    assert parse_serialized(out)["docstring"] == record["docstring"]
    assert parse_serialized(out)["code"] == record["code"]


def test_escaped_content_does_not_leak_control_token_ids(tok):
    vocab = tok.get_vocab()
    record = {
        **PY_RECORD,
        "docstring": "mention <|eos|> and <|lang:fr|> inside text",
        "code": "s = '<|eos|> <|pad|>'\n",
    }
    ids = tok.encode(serialize_code_record(record)).ids
    assert ids[0] == vocab["<|lang:python|>"]
    assert ids[1] == vocab["<|docstring|>"]
    assert ids[-1] == vocab["<|eos|>"]
    assert ids.count(vocab["<|eos|>"]) == 1
    assert ids.count(vocab["<|code|>"]) == 1
    assert ids.count(vocab["<|pad|>"]) == 0
    assert ids.count(vocab["<|lang:fr|>"]) == 0
    # and the stream still decodes back to the escaped text, then to the original
    text = tok.decode(ids, skip_special_tokens=False)
    parsed = parse_serialized(text)
    assert parsed["docstring"] == record["docstring"]
    assert parsed["code"] == record["code"]


# ------------------------------------------------------- deserialization ----
def test_parse_serialized_recovers_code_records():
    for record, lang in ((PY_RECORD, "python"), (JS_RECORD, "javascript")):
        parsed = parse_serialized(serialize_code_record(record))
        assert parsed == {
            "kind": "code",
            "language": lang,
            "docstring": record["docstring"],
            "code": record["code"],
        }


def test_parse_serialized_recovers_documentation():
    parsed = parse_serialized(serialize_documentation(RST))
    assert parsed == {"kind": "documentation", "natural_language": "en", "text": RST}


@pytest.mark.parametrize(
    "bad",
    [
        "<|lang:python|>no markers",
        "<|lang:python|><|docstring|>a<|eos|>",
        "<|lang:en|>text<|eos|>",
        "plain text",
    ],
)
def test_parse_serialized_rejects_malformed_streams(bad):
    with pytest.raises(ValueError):
        parse_serialized(bad)


# --------------------------------------- unicode through the serializer -----
@pytest.mark.parametrize(
    "payload",
    [
        "Le café est prêt — déjà vu, œuvre, naïve.",
        "البرمجة بالعربية: دالة تُرجع المجموع.",
        "🎉 emoji + ﬁ ligature + 中文 + e\u0301",
    ],
)
def test_unicode_payload_roundtrips(tok, payload):
    record = {**PY_RECORD, "docstring": payload, "code": f"# {payload}\npass\n"}
    text = serialize_code_record(record)
    assert tok.decode(tok.encode(text).ids, skip_special_tokens=False) == text
    assert parse_serialized(text)["code"] == record["code"]


# ------------------------------------------------ long input / no truncation
def test_long_input_not_truncated(tok):
    long_code = ("def f(x):\n    return x + 1  # padding\n" * 4000) + "\\ escape at end\n"
    record = {**PY_RECORD, "docstring": "long", "code": long_code}
    text = serialize_code_record(record)
    assert text.endswith("<|eos|>")
    parsed = parse_serialized(text)
    assert parsed["code"] == long_code
    assert len(parsed["code"]) == len(long_code)
    ids = tok.encode(text).ids
    assert tok.decode(ids, skip_special_tokens=False) == text


# ------------------------------------------------- save / reload parity -----
def test_save_reload_consistency(tok):
    reloaded = Tokenizer.from_str(tok.to_str())
    assert reloaded.get_vocab() == tok.get_vocab()
    for text in (serialize_code_record(PY_RECORD), serialize_code_record(JS_RECORD), serialize_documentation(RST)):
        assert reloaded.encode(text).ids == tok.encode(text).ids
        assert reloaded.decode(tok.encode(text).ids, skip_special_tokens=False) == text

    on_disk = Tokenizer.from_file(str(TOKENIZER_PATH))
    assert on_disk.get_vocab() == tok.get_vocab()
    assert on_disk.encode(serialize_code_record(PY_RECORD)).ids == tok.encode(serialize_code_record(PY_RECORD)).ids


def test_special_token_ids_unique(tok):
    vocab = tok.get_vocab()
    ids = [vocab[t] for t in SPECIAL_TOKENS]
    assert len(set(ids)) == 11
    assert ids == list(range(11))
    for token in SPECIAL_TOKENS:
        assert tok.encode(token).ids == [vocab[token]]


def test_unknown_token_configuration(tok):
    vocab = tok.get_vocab()
    assert vocab["<|unk|>"] == 1
    assert tok.get_added_tokens_decoder()[1].content == "<|unk|>"
    assert tok.get_added_tokens_decoder()[1].special is True
    # byte-level alphabet makes unk unreachable for any str input
    for sample in ("", "plain", "الحروف", "é", "🎉", "\x00\x01\x7f", "\\", "<|eos|>"):
        assert vocab["<|unk|>"] not in tok.encode(sample).ids
