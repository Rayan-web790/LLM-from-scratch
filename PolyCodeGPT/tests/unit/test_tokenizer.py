"""Unit tests for the trained PolyCodeGPT byte-level BPE tokenizer.

Covers (A) special-token registration, (B) unique IDs, (C) Python code,
(D) JavaScript code, (E) indentation/whitespace, (F) punctuation and operators,
(G) English, (H) French accents, (I) Arabic, (J) emoji/arbitrary Unicode,
(K) round-trip reconstruction, (L) empty input, (M) long input.

All inputs are hand-written fixtures defined below. None of them appear in the
tokenizer training corpus, and nothing here reads valid.jsonl or test.jsonl.
Byte-level encodability of French and Arabic proves only that the vocabulary
can represent those bytes — it is not evidence of language-model proficiency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polycodegpt.tokenization.serializer import (  # noqa: E402
    SPECIAL_TOKENS,
    serialize_code_record,
    serialize_document,
)

TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "artifacts" / "polycodegpt_tokenizer_v1_1.json"

PYTHON_CODE = '''def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number."""
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
'''

JS_CODE = """const memo = new Map();
function fib(n) {
  if (n < 2) return n;
  if (memo.has(n)) return memo.get(n);
  const value = fib(n - 1) + fib(n - 2);
  memo.set(n, value);
  return value;
}
export default fib;
"""

INDENTED = "class A:\n    def method(self):\n        if self.x:\n            return\t1\n        return 0\n"
OPERATORS = "x = (a + b) * c ** 2 // 3 % 4; y => y !== null && !z || (q ?? 0); a <<= 1; s = `t${v}`;\n"
ENGLISH = "The tokenizer preserves whitespace, punctuation, and casing exactly as written.\n"
FRENCH = "Le renard brun saute par-dessus le chien paresseux. Ça coûte 12 € — naïve, déjà vu, œuvre.\n"
ARABIC = "البرمجة بلغة بايثون ممتعة وسهلة التعلم، والتوثيق متاح بالكامل.\n"
EMOJI = "🚀 Начало — 開始 — 🙂‍↔️ e\u0301 ﬁ ½ ✓ \u200b end\n"
LONG_UNIT = ("def f(x):\n    return x + 1  # padding line\n" * 4000)


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    assert TOKENIZER_PATH.is_file(), (
        f"tokenizer artifact missing: {TOKENIZER_PATH}. Run scripts/train_tokenizer.py first."
    )
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def encode_text(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text).ids


def roundtrip(tok: Tokenizer, text: str) -> str:
    """Decode without dropping special tokens, so the result is comparable."""
    return tok.decode(encode_text(tok, text), skip_special_tokens=False)


# ---------------------------------------------------------------- A / B -----
def test_a_special_tokens_registered(tok):
    vocab = tok.get_vocab()
    added = tok.get_added_tokens_decoder()
    assert len(SPECIAL_TOKENS) == 11
    for token in SPECIAL_TOKENS:
        assert token in vocab, f"{token} missing from vocabulary"
        assert vocab[token] in added, f"{token} not registered as an added/special token"
    assert tok.encode("<|lang:python|>").tokens == ["<|lang:python|>"]


def test_b_special_token_ids_unique(tok):
    vocab = tok.get_vocab()
    ids = [vocab[t] for t in SPECIAL_TOKENS]
    assert len(set(ids)) == 11, f"duplicate special token IDs: {ids}"
    assert ids == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], f"unexpected special token order: {ids}"
    # each special token must encode to exactly its own single ID
    for token in SPECIAL_TOKENS:
        assert tok.encode(token).ids == [vocab[token]]


# ------------------------------------------------------------ C / D / E -----
def test_c_python_code_roundtrip(tok):
    ids = encode_text(tok, PYTHON_CODE)
    assert len(ids) > 0
    assert roundtrip(tok, PYTHON_CODE) == PYTHON_CODE
    # Python keywords should not explode into single characters at this vocab size
    assert len(ids) < len(PYTHON_CODE)


def test_d_javascript_code_roundtrip(tok):
    ids = encode_text(tok, JS_CODE)
    assert len(ids) > 0
    assert roundtrip(tok, JS_CODE) == JS_CODE
    assert len(ids) < len(JS_CODE)


def test_e_indentation_and_whitespace_preserved(tok):
    assert roundtrip(tok, INDENTED) == INDENTED
    for fragment in ("    def method", "        if self.x", "            return\t1"):
        assert roundtrip(tok, fragment) == fragment
    # leading/trailing runs are kept verbatim
    spaced = "   a   b\t\tc\n\n"
    assert roundtrip(tok, spaced) == spaced
    tokens = tok.encode("    return x").tokens
    assert tokens and tokens[0].startswith("\u0120")  # space marker, not stripped


# ------------------------------------------------------------------- F ------
def test_f_punctuation_and_operators(tok):
    assert roundtrip(tok, OPERATORS) == OPERATORS
    for fragment in ("->", "=>", "!==", "**", "//", "??", "`t${v}`", "@decorator", "#comment"):
        assert roundtrip(tok, fragment) == fragment


# ------------------------------------------------------------ G / H / I -----
def test_g_english_text(tok):
    assert roundtrip(tok, ENGLISH) == ENGLISH
    assert roundtrip(tok, serialize_document(ENGLISH)).startswith("<|lang:en|><|text|>")


def test_h_french_accents(tok):
    assert roundtrip(tok, FRENCH) == FRENCH
    # accented characters must be representable without falling back to <|unk|>
    assert tok.get_vocab()["<|unk|>"] not in encode_text(tok, FRENCH)


def test_i_arabic_text(tok):
    assert roundtrip(tok, ARABIC) == ARABIC
    assert tok.get_vocab()["<|unk|>"] not in encode_text(tok, ARABIC)


# ------------------------------------------------------------------- J ------
def test_j_emoji_and_arbitrary_unicode(tok):
    assert roundtrip(tok, EMOJI) == EMOJI
    assert tok.get_vocab()["<|unk|>"] not in encode_text(tok, EMOJI)
    for fragment in ("中文测试", "日本語のテキスト", "🙂", "e\u0301", "\U0001f468\u200d\U0001f469\u200d\U0001f467"):
        assert roundtrip(tok, fragment) == fragment


# ------------------------------------------------------------------- K ------
def test_k_roundtrip_reconstruction(tok):
    py = serialize_code_record(
        {
            "language": "python",
            "source_dataset": "CodeSearchNet Python",
            "docstring": "Add two numbers.",
            "code": "def add(a, b):\n    return a + b\n",
        }
    )
    js = serialize_code_record(
        {
            "language": "javascript",
            "source_dataset": "CodeSearchNet JavaScript",
            "docstring": "Add two numbers.",
            "code": "function add(a, b) {\n  return a + b;\n}\n",
        }
    )
    for text in (py, js, serialize_document("# Doc\n\nText with `code`.\n")):
        assert roundtrip(tok, text) == text
    assert encode_text(tok, py)[0] == tok.get_vocab()["<|lang:python|>"]
    assert encode_text(tok, js)[0] == tok.get_vocab()["<|lang:javascript|>"]


# ------------------------------------------------------------ L / M ---------
def test_l_empty_input(tok):
    enc = tok.encode("")
    assert enc.ids == []
    assert tok.decode(enc.ids) == ""
    assert tok.decode(enc.ids, skip_special_tokens=False) == ""


def test_m_long_input(tok):
    ids = encode_text(tok, LONG_UNIT)
    assert len(ids) > 1000
    assert roundtrip(tok, LONG_UNIT) == LONG_UNIT
    assert len(ids) < len(LONG_UNIT)  # compression, not byte-per-token
