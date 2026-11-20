"""Exploratory, heuristic record classifier used by the Phase 10E/10F audits.

This module is deliberately not a semantic annotator. Every category below is
produced by a fixed, mechanical rule over the serialized record's ``code`` and
``docstring`` strings. Labels are multi-label: one record can and often does
carry several labels, and the keyword categories are *not* mutually exclusive.

Category rules (exact):

``concise_standalone``
    ``len(code.splitlines()) <= 12`` and the stripped code starts with ``def ``
    (Python) or ``function `` (JavaScript).

``algorithmic``
    Case-insensitive regex ``\\b(sort|search|factorial|fibonacci|prime|gcd|
    binary|bubble|merge)\\b`` matches in ``code + docstring``.

``framework_application``
    Case-insensitive regex ``\\b(react|angular|vue|express|webpack|babel|grunt|
    gulp|jquery|bootstrap|element)\\b`` matches in ``code + docstring``.

``compiler_tooling``
    Case-insensitive regex ``\\b(typescript|compiler|babel|eslint|webpack|
    parser|syntax|ast)\\b`` matches in ``code + docstring``.

``api_wrapper``
    Case-insensitive regex ``\\b(fetch|axios|request|api|http|endpoint)\\b``
    matches in ``code + docstring``.

``incomplete_or_no_closing_brace`` (JavaScript only)
    Stripped code does not end with ``}``.

``incomplete_or_no_colon`` (Python only)
    Stripped code does not end with ``:`` or ``)``.

``comment_like_description``
    Stripped docstring starts with ``#``, ``//``, ``/*`` or ``*``, or contains
    fewer than four whitespace-separated words.

``meaningful_description``
    The complement of ``comment_like_description``: docstring is non-empty,
    does not start with a comment marker, and has at least four words.

Known limitations of ``meaningful_description``: it is a length/prefix test, not
a readability or usefulness test. Structured JSDoc blocks (``@param``,
``@return``, ``{type}`` tags), Sphinx ``:param:`` blocks, and terse imperative
phrases ("Populate the resolutions array") all pass the rule whenever they have
four or more words, so the category over-counts "natural language prose" in
JavaScript. The rule also cannot detect a docstring that is a copy of the
function signature, a bare URL, or generated boilerplate. Keyword categories are
likewise weak evidence: a match only proves the token appears somewhere in the
record (often in an unrelated identifier or comment), never that the record is a
genuine algorithmic/framework/compiler example.
"""

from __future__ import annotations

import re

CATEGORY_NAMES = (
    "concise_standalone",
    "algorithmic",
    "framework_application",
    "compiler_tooling",
    "api_wrapper",
    "incomplete_or_no_closing_brace",
    "incomplete_or_no_colon",
    "comment_like_description",
    "meaningful_description",
)

_ALGORITHMIC = re.compile(r"\b(sort|search|factorial|fibonacci|prime|gcd|binary|bubble|merge)\b")
_FRAMEWORK = re.compile(r"\b(react|angular|vue|express|webpack|babel|grunt|gulp|jquery|bootstrap|element)\b")
_TOOLING = re.compile(r"\b(typescript|compiler|babel|eslint|webpack|parser|syntax|ast)\b")
_API = re.compile(r"\b(fetch|axios|request|api|http|endpoint)\b")


def classify(lang: str, record: dict) -> set[str]:
    """Return the set of heuristic labels for one corpus record."""
    code = record.get("code", "") or ""
    doc = record.get("docstring", "") or ""
    cats: set[str] = set()

    if len(code.splitlines()) <= 12 and (
        code.strip().startswith("def ") or code.strip().startswith("function ")
    ):
        cats.add("concise_standalone")

    haystack = (code + doc).lower()
    if _ALGORITHMIC.search(haystack):
        cats.add("algorithmic")
    if _FRAMEWORK.search(haystack):
        cats.add("framework_application")
    if _TOOLING.search(haystack):
        cats.add("compiler_tooling")
    if _API.search(haystack):
        cats.add("api_wrapper")

    if lang == "javascript" and not code.strip().endswith("}"):
        cats.add("incomplete_or_no_closing_brace")
    if lang == "python" and not code.strip().endswith(":") and not code.strip().endswith(")"):
        cats.add("incomplete_or_no_colon")

    doc_stripped = doc.strip()
    if doc_stripped.startswith(("#", "//", "/*", "*")) or len(doc_stripped.split()) < 4:
        cats.add("comment_like_description")
    else:
        cats.add("meaningful_description")

    return cats
