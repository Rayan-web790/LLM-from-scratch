"""Deterministic repository-level train/validation assignment."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Iterable

from .constants import VALIDATION_PERCENT
from .language import repo_group


def partition_for(repo: str, content_kind: str, forced_train_repos: set[str] | None = None) -> str:
    """Assign a repository to train or validation.

    Documentation is train-only, matching the historical docs policy. Any
    repository that contributes documentation is forced to train, so its code
    cannot leak into validation. Other code repositories use a stable hash so
    no repository can appear in both splits.
    """
    repo = repo_group("", repo)
    if content_kind == "documentation":
        return "train"
    if not repo or repo == "unknown":
        return "train"
    if forced_train_repos and repo in forced_train_repos:
        return "train"
    digest = hashlib.sha256(f"polycodegpt-v2|{repo}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "validation" if bucket < VALIDATION_PERCENT else "train"


def split_report(records: Iterable[dict]) -> dict:
    repos: dict[str, set[str]] = {"train": set(), "validation": set()}
    records_by_split = Counter()
    tokens_by_split = Counter()
    for rec in records:
        part = rec["partition"]
        records_by_split[part] += 1
        tokens_by_split[part] += int(rec.get("token_count", 0))
        repos[part].add(rec["repo"])
    overlap = sorted(repos["train"] & repos["validation"])
    return {
        "validation_percent_policy": VALIDATION_PERCENT,
        "repository_overlap": overlap,
        "repository_overlap_count": len(overlap),
        "train_repositories": len(repos["train"]),
        "validation_repositories": len(repos["validation"]),
        "records": dict(records_by_split),
        "tokens": dict(tokens_by_split),
    }


def check_no_repo_overlap(records: Iterable[dict]) -> None:
    seen: dict[str, str] = {}
    for rec in records:
        repo = repo_group(rec["source_id"], rec["repo"])
        part = rec["partition"]
        if repo in seen and seen[repo] != part:
            raise RuntimeError(f"repository leakage: {repo} in {seen[repo]} and {part}")
        seen[repo] = part
