#!/usr/bin/env python
"""Safe inspection of CodeSearchNet *_licenses.pkl metadata (Phase 2B.2).

Two-stage guard before any deserialization:

  Stage 1 - static opcode scan with pickletools.genops(). Any opcode capable of
            importing, calling, instantiating, or resolving external state is a
            hard rejection (GLOBAL, STACK_GLOBAL, REDUCE, BUILD, OBJ, INST,
            NEWOBJ, NEWOBJ_EX, EXT1/2/4, PERSID, BINPERSID).
  Stage 2 - a restricted Unpickler whose find_class() unconditionally raises,
            so no global can be resolved even if stage 1 were bypassed.

Only after both stages pass is the payload loaded, then its schema is validated.

Standard library only. Read-only with respect to data/raw.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import pickletools
import sys
from collections import Counter

MAX_FILE_BYTES = 512 * 1024 * 1024

# Opcodes that can execute code, import modules, or resolve external references.
FORBIDDEN = {
    "GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "OBJ", "INST",
    "NEWOBJ", "NEWOBJ_EX", "EXT1", "EXT2", "EXT4",
    "PERSID", "BINPERSID",
}
# Also refuse protocol-0 INST/OBJ spellings and any module-level attribute load.
FORBIDDEN_PREFIXES = ("EXT", "PERSID")


class UnsafePickle(RuntimeError):
    pass


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that refuses every global lookup."""

    def find_class(self, module, name):  # noqa: D102
        raise UnsafePickle(f"refused global resolution: {module}.{name}")


def scan_opcodes(path: str) -> dict:
    counts = Counter()
    first_seen = {}
    with open(path, "rb") as fh:
        for opcode, arg, pos in pickletools.genops(fh):
            counts[opcode.name] += 1
            first_seen.setdefault(opcode.name, (pos, arg))
    bad = []
    for name in counts:
        if name in FORBIDDEN or name.startswith(FORBIDDEN_PREFIXES):
            bad.append((name, counts[name], first_seen[name]))
    return {"counts": counts, "forbidden": bad}


def safe_load(path: str):
    with open(path, "rb") as fh:
        return RestrictedUnpickler(fh, encoding="utf-8").load()


def analyse(path: str, label: str) -> dict:
    size = os.path.getsize(path)
    if size > MAX_FILE_BYTES:
        raise UnsafePickle(f"{path}: {size} bytes exceeds cap {MAX_FILE_BYTES}")

    with open(path, "rb") as fh:
        head = fh.read(2)
    proto = head[1] if head[:1] == b"\x80" else 0

    scan = scan_opcodes(path)
    print(f"[{label}] size={size} proto={proto} opcodes={len(scan['counts'])}")
    print(f"[{label}] top opcodes: "
          + ", ".join(f"{k}={v}" for k, v in scan["counts"].most_common(8)))
    if scan["forbidden"]:
        for name, n, (pos, arg) in scan["forbidden"]:
            print(f"[{label}] FORBIDDEN opcode {name} x{n} at byte {pos} "
                  f"(arg={arg!r})", file=sys.stderr)
        raise UnsafePickle(f"{label}: dangerous opcodes present")

    print(f"[{label}] opcode scan: clean (no GLOBAL/REDUCE/BUILD/... found)")
    data = safe_load(path)
    print(f"[{label}] restricted unpickle: OK, type={type(data).__name__}")

    if not isinstance(data, dict):
        raise UnsafePickle(f"{label}: expected dict, got {type(data)}")

    stats = {
        "label": label, "path": path, "sha256": hashlib.sha256(
            open(path, "rb").read()).hexdigest(),
        "file_bytes": size, "protocol": proto,
        "records": len(data),
        "key_types": Counter(), "value_types": Counter(),
        "key_shapes": Counter(), "tuple_arities": Counter(),
        "empty_content": 0, "nonempty_content": 0,
        "non_utf8_candidates": 0, "empty_path": 0,
        "bad_keys": [], "bad_values": [],
        "duplicate_keys": 0,
    }

    seen = set()
    for k, v in data.items():
        stats["key_types"][type(k).__name__] += 1
        stats["value_types"][type(v).__name__] += 1
        if k in seen:
            stats["duplicate_keys"] += 1
        seen.add(k)
        if isinstance(k, str):
            parts = k.count("/")
            stats["key_shapes"][parts] += 1
            if parts != 1:
                stats["bad_keys"].append(k)
        else:
            stats["bad_keys"].append(repr(k)[:80])

        # Observed shapes: list[(path, content)] (per fetch_licenses.py) or a
        # single (path, content) tuple (per resources/README.md).
        entries = []
        if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str):
            entries = [v]
            stats["tuple_arities"]["single-tuple"] += 1
        elif isinstance(v, list):
            stats["tuple_arities"][f"list[{len(v)}]"] += 1
            for e in v:
                if isinstance(e, tuple) and len(e) == 2:
                    entries.append(e)
                else:
                    stats["bad_values"].append(repr(e)[:80])
        else:
            stats["bad_values"].append(repr(v)[:80])

        for p, c in entries:
            if not isinstance(p, str):
                stats["bad_values"].append(repr(p)[:80]); continue
            if not p:
                stats["empty_path"] += 1
            if not isinstance(c, str):
                stats["bad_values"].append(repr(c)[:80]); continue
            if c.strip():
                stats["nonempty_content"] += 1
            else:
                stats["empty_content"] += 1
            try:
                c.encode("utf-8")
            except UnicodeEncodeError:
                stats["non_utf8_candidates"] += 1

    print(f"[{label}] records={stats['records']} key_types={dict(stats['key_types'])} "
          f"value_types={dict(stats['value_types'])}")
    print(f"[{label}] value shapes={dict(stats['value_arities'] if 'value_arities' in stats else stats['tuple_arities'])}")
    print(f"[{label}] empty_content={stats['empty_content']} "
          f"nonempty_content={stats['nonempty_content']} "
          f"bad_keys={len(stats['bad_keys'])} bad_values={len(stats['bad_values'])}")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", help="path=label pairs, e.g. a.pkl=python")
    args = ap.parse_args(argv)
    rc = 0
    for spec in args.files:
        path, _, label = spec.partition("=")
        label = label or os.path.basename(path)
        try:
            analyse(path, label)
        except UnsafePickle as exc:
            print(f"[{label}] REJECTED: {exc}", file=sys.stderr)
            rc = 4
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
