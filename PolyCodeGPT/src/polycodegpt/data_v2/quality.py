"""Static quality gates and deterministic duplicate fingerprints."""

from __future__ import annotations

import hashlib
import re
import zlib

from .constants import (
    CODE_MAX_CHARS,
    CODE_MAX_LINES,
    CODE_MIN_CHARS,
    CODE_MIN_LINES,
    GENERATED_MARKERS,
)
from .language import path_rejection, syntax_ok

_WS_RE = re.compile(r"\s+")
_PY_COMMENT_RE = re.compile(r"(?m)#.*$")
_JS_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
_JS_LINE_COMMENT_RE = re.compile(r"(?m)//.*$")
_TOKEN_RE = re.compile(r"[A-Za-z_$][\w$]*|\d+|[^\s\w]")
_MINHASH_PERMUTATIONS = 16
_NEAR_BANDS = 4
_NEAR_ROWS = 4
_NEAR_THRESHOLD = 0.8
_SHINGLE_CAP = 64


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _strip_comments(language: str, text: str) -> str:
    if language == "python":
        return _PY_COMMENT_RE.sub("", text)
    if language == "javascript":
        return _JS_LINE_COMMENT_RE.sub("", _JS_BLOCK_COMMENT_RE.sub("", text))
    return text


def normalized_key(language: str, text: str) -> str:
    """Whitespace/comment-insensitive key; exact text remains separately keyed."""
    return sha256_text(_WS_RE.sub("", _strip_comments(language, canonicalize(text))))


def minified_like(code: str) -> bool:
    lines = code.splitlines()
    if not lines:
        return True
    longest = max(len(line) for line in lines)
    average = sum(len(line) for line in lines) / len(lines)
    if longest > 1200:
        return True
    if len(lines) >= 8 and average > 260:
        return True
    # Dense one-line payloads are usually generated bundles.
    if len(lines) <= 3 and longest > 700 and code.count(";") >= 12:
        return True
    return False


def generated_like(code: str) -> bool:
    head = code[:5000].lower()
    return any(marker in head for marker in GENERATED_MARKERS)


def code_rejection(language: str, code: str, path: str) -> str | None:
    """Return a stable rejection code, or None if the candidate passes."""
    if not isinstance(code, str):
        return "non_string_content"
    code = canonicalize(code)
    length = len(code)
    lines = code.count("\n") + 1
    if length < CODE_MIN_CHARS:
        return "too_short"
    if length > CODE_MAX_CHARS:
        return "too_long"
    if lines < CODE_MIN_LINES:
        return "too_few_lines"
    if lines > CODE_MAX_LINES:
        return "too_many_lines"
    if "\ufffd" in code:
        return "decoding_replacement"
    if path_rejection(path):
        return path_rejection(path) or "bad_path"
    if generated_like(code):
        return "generated_marker"
    if minified_like(code):
        return "minified"
    ok, reason = syntax_ok(language, code)
    if not ok:
        return f"syntax_invalid:{reason}"
    return None


def documentation_rejection(text: str, path: str) -> str | None:
    if not isinstance(text, str):
        return "non_string_content"
    text = canonicalize(text)
    if len(text) < 400:
        return "too_short"
    if len(text) > 20_000:
        return "too_long"
    if "\ufffd" in text:
        return "decoding_replacement"
    if path_rejection(path):
        return path_rejection(path) or "bad_path"
    return None


def _shingles(text: str) -> list[int]:
    tokens = _TOKEN_RE.findall(canonicalize(text).lower())
    if len(tokens) < 5:
        return []
    shingles = [" ".join(tokens[i : i + 5]) for i in range(len(tokens) - 4)]
    if len(shingles) > _SHINGLE_CAP:
        step = len(shingles) / _SHINGLE_CAP
        shingles = [shingles[int(i * step)] for i in range(_SHINGLE_CAP)]
    return [zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF for s in shingles]


def minhash_signature(text: str) -> tuple[int, ...]:
    """16-permutation MinHash over 5-token shingles.

    Base shingle hashes use CRC32 (fast, deterministic). Permutations are
    deterministic integer mixes; this is a near-duplicate index, not a
    cryptographic fingerprint.
    """
    hashes = _shingles(text)
    if not hashes:
        return tuple([0] * _MINHASH_PERMUTATIONS)
    signature = []
    for perm in range(_MINHASH_PERMUTATIONS):
        seed = (perm + 1) * 0x9E3779B1
        signature.append(min(((base ^ seed) * 0x85EBCA6B) & 0xFFFFFFFF for base in hashes))
    return tuple(signature)


def near_bands(signature: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [
        tuple(signature[i * _NEAR_ROWS : (i + 1) * _NEAR_ROWS])
        for i in range(_NEAR_BANDS)
    ]


def signature_similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or not b:
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / len(a)


def is_near_duplicate(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return signature_similarity(a, b) >= _NEAR_THRESHOLD


def code_fingerprints(language: str, code: str) -> tuple[str, str, tuple[int, ...]]:
    exact = sha256_text(canonicalize(code))
    norm = normalized_key(language, code)
    sig = minhash_signature(code)
    return exact, norm, sig


def documentation_fingerprints(text: str) -> tuple[str, str, tuple[int, ...]]:
    exact = sha256_text(canonicalize(text))
    norm = normalized_key("en", text)
    sig = minhash_signature(text)
    return exact, norm, sig
