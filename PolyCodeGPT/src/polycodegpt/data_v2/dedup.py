"""Global exact, normalized, and MinHash/LSH near-duplicate removal."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

from .quality import is_near_duplicate, near_bands


@dataclass
class DedupStats:
    seen: int = 0
    exact: int = 0
    normalized: int = 0
    near: int = 0
    kept: int = 0
    by_group: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    def as_dict(self) -> dict:
        return {
            "seen": self.seen,
            "exact_duplicates": self.exact,
            "normalized_duplicates": self.normalized,
            "near_duplicates": self.near,
            "kept": self.kept,
            "by_group": {k: dict(v) for k, v in sorted(self.by_group.items())},
        }


def deduplicate(
    records: Iterable[dict],
    *,
    stats: DedupStats | None = None,
    reject: Callable[[dict, str, str], None] | None = None,
) -> Iterator[dict]:
    """Yield first occurrences, dropping later exact/normalized/near duplicates.

    ``records`` must carry ``exact_sha256``, ``normalized_sha256``, and
    ``near_signature`` plus a grouping key (language + content kind). Duplicate
    removal happens globally before the train/validation split.
    """
    stats = stats or DedupStats()
    exact_seen: dict[str, str] = {}
    norm_seen: dict[str, str] = {}
    buckets: dict[tuple[str, tuple[int, ...]], list[tuple[tuple[int, ...], str]]] = defaultdict(list)
    for rec in records:
        stats.seen += 1
        group = f"{rec.get('content_kind', 'code')}:{rec.get('language', '')}"
        gstats = stats.by_group[group]
        exact = rec.get("exact_sha256") or ""
        norm = rec.get("normalized_sha256") or ""
        sig = tuple(rec.get("near_signature") or ())
        if exact and exact in exact_seen:
            stats.exact += 1
            gstats["exact"] += 1
            if reject:
                reject(rec, "exact_duplicate", exact_seen[exact])
            continue
        if norm and norm in norm_seen:
            stats.normalized += 1
            gstats["normalized"] += 1
            if reject:
                reject(rec, "normalized_duplicate", norm_seen[norm])
            continue
        near_hit = None
        if sig and any(sig):
            for band in near_bands(sig):
                key = (group, band)
                for other_sig, other_id in buckets[key]:
                    if is_near_duplicate(sig, other_sig):
                        near_hit = other_id
                        break
                if near_hit:
                    break
        if near_hit:
            stats.near += 1
            gstats["near"] += 1
            if reject:
                reject(rec, "near_duplicate", near_hit)
            continue
        # Keep this record only after all three checks.
        if exact:
            exact_seen[exact] = rec["record_id"]
        if norm:
            norm_seen[norm] = rec["record_id"]
        if sig and any(sig):
            for band in near_bands(sig):
                buckets[(group, band)].append((sig, rec["record_id"]))
        stats.kept += 1
        gstats["kept"] += 1
        yield rec
