"""Phase 12A Part B/C — deterministic eligibility rules for the JS audit.

Every threshold used to accept or reject a record lives here so the filtering
rules quoted in the reports are the rules the pipeline actually runs. Nothing in
this module touches the corpus; it is pure classification over features that the
audit pipeline has already extracted.

Rule identifiers (``REASONS``) are stable and appear verbatim in the filtering
manifest, so every exclusion is auditable.
"""

from __future__ import annotations

import hashlib
import re

#: Descriptions are bucketed by shape first, then by usefulness.
DESCRIPTION_CATEGORIES = (
    "structured_jsdoc",
    "jsdoc_block",
    "plain_text",
    "line_comment",
    "none",
)

#: Minimum size for a description to count as meaningful natural language.
MIN_DESCRIPTION_WORDS = 4
MIN_DESCRIPTION_CHARS = 20
MAX_DESCRIPTION_CHARS = 1200
MIN_ALPHA_RATIO = 0.50

#: Code must be big enough to teach something and small enough that the whole
#: serialized record (docstring + code + markers) fits one 512-token window.
MIN_CODE_TOKENS = 12
MAX_CODE_TOKENS = 500
MAX_SERIALIZED_TOKENS = 512

#: Structural kinds that can be supervised as standalone function completion.
ALLOWED_KINDS = ("function_decl", "function_assignment", "arrow_assignment")

#: Near-duplicate detection: 15 minhash values over 5-token shingles, banded in
#: three groups of five. Documented as approximate (see report).
SHINGLE_SIZE = 5
PERMUTATIONS = 15
BANDS = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14))

#: Reasons, most-blocking first. A record is recorded against the first reason it
#: hits, and every reason it hits is kept in the audit trail.
REASONS = (
    "parse_invalid",
    "not_standalone_function",
    "description_missing",
    "description_fragment",
    "description_boilerplate",
    "depends_on_ecosystem",
    "stateful_or_method",
    "code_too_short",
    "code_too_long",
    "name_mismatch",
    "duplicate",
    "near_duplicate",
)

REASON_TEXT = {
    "parse_invalid": "V8 rejected the snippet (syntax error)",
    "not_standalone_function": "not a single complete standalone function definition",
    "description_missing": "no description text",
    "description_fragment": "description is a comment fragment, not usable natural language",
    "description_boilerplate": "description is a licence/copyright preamble",
    "depends_on_ecosystem": "requires modules, imports/exports, or host globals",
    "stateful_or_method": "uses `this`/object state, so not a self-contained function",
    "code_too_short": f"fewer than {MIN_CODE_TOKENS} code tokens",
    "code_too_long": f"more than {MAX_SERIALIZED_TOKENS} serialized tokens",
    "name_mismatch": "declared function name disagrees with the corpus func_name",
    "duplicate": "identical normalized code already kept",
    "near_duplicate": "near-identical code already kept (minhash LSH)",
}

BOILERPLATE_PATTERNS = (
    re.compile(r"copyright", re.I),
    re.compile(r"all rights reserved", re.I),
    re.compile(r"licensed under", re.I),
    re.compile(r"spdx-license-identifier", re.I),
    re.compile(r"this file is part of", re.I),
    re.compile(r"@license\b", re.I),
)


def classify_description(text: str) -> tuple[str, str]:
    """Return ``(category, quality)`` for a record's description.

    ``category`` is the shape of the comment, ``quality`` is ``meaningful``,
    ``fragment``, ``boilerplate`` or ``missing``.
    """
    stripped = (text or "").strip()
    if not stripped:
        return "none", "missing"

    if stripped.startswith("/**"):
        category = "structured_jsdoc" if re.search(r"^\s*\*\s*@\w+", stripped, re.M) else "jsdoc_block"
        body = re.sub(r"^\s*\*\s?", "", stripped.strip("/*").strip(), flags=re.M)
    elif stripped.startswith("//") or stripped.startswith("#"):
        category = "line_comment"
        body = stripped.lstrip("/#").strip()
    else:
        category = "plain_text"
        body = stripped

    if any(pattern.search(stripped) for pattern in BOILERPLATE_PATTERNS):
        return category, "boilerplate"

    words = [w for w in re.split(r"\s+", body) if re.search(r"[A-Za-z]", w)]
    alpha_ratio = (
        sum(ch.isalpha() or ch.isspace() for ch in body) / len(body) if body else 0.0
    )
    if (
        category == "line_comment"
        or len(body) < MIN_DESCRIPTION_CHARS
        or len(body) > MAX_DESCRIPTION_CHARS
        or len(words) < MIN_DESCRIPTION_WORDS
        or alpha_ratio < MIN_ALPHA_RATIO
    ):
        return category, "fragment"
    return category, "meaningful"


def normalize_code(code: str) -> str:
    """Whitespace-insensitive form used for exact duplicate detection."""
    return re.sub(r"\s+", " ", (code or "").strip())


def normalize_name(name: str | None) -> str:
    """Reduce ``Foo.prototype.bar`` / ``exports.bar`` to a comparable leaf name."""
    if not name:
        return ""
    leaf = re.split(r"[.\[\]]", name.strip())[-1]
    return re.sub(r"[^A-Za-z0-9_$]", "", leaf).lower()


def exact_hash(code: str) -> str:
    return hashlib.sha256(normalize_code(code).encode("utf-8")).hexdigest()


def evaluate(features: dict) -> tuple[list[str], dict]:
    """Classify one record. Returns ``(reasons, flags)`` in canonical order.

    ``features`` carries the parser output plus the derived description and
    length measurements. Any non-empty ``reasons`` list means rejection.
    """
    reasons: set[str] = set()

    if features["parse"] == "invalid":
        reasons.add("parse_invalid")

    standalone = (
        features["kind"] in ALLOWED_KINDS
        and features["complete"]
        and features["body_kind"] in ("block", "expression")
    )
    if not standalone:
        reasons.add("not_standalone_function")

    quality = features["description_quality"]
    if quality == "missing":
        reasons.add("description_missing")
    elif quality == "boilerplate":
        reasons.add("description_boilerplate")
    elif quality == "fragment":
        reasons.add("description_fragment")

    if features["has_require"] or features["has_import_export"] or features["uses_host_global"]:
        reasons.add("depends_on_ecosystem")

    if features["uses_this"]:
        reasons.add("stateful_or_method")

    if features["code_tokens"] < MIN_CODE_TOKENS:
        reasons.add("code_too_short")
    if features["serialized_tokens"] > MAX_SERIALIZED_TOKENS or features["code_tokens"] > MAX_CODE_TOKENS:
        reasons.add("code_too_long")

    corpus_name = normalize_name(features.get("func_name"))
    declared = normalize_name(features.get("declared_name"))
    if corpus_name and declared and corpus_name != declared:
        reasons.add("name_mismatch")

    ordered = [reason for reason in REASONS if reason in reasons]
    flags = {
        "console_calls": features.get("console_calls", 0),
        "declared_name": features.get("declared_name"),
        "corpus_name": features.get("func_name"),
        "description_category": features["description_category"],
        "description_quality": quality,
        "reasons_all": ordered,
    }
    return ordered, flags
