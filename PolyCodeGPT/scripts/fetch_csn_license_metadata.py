#!/usr/bin/env python
"""Selective retrieval of CodeSearchNet license metadata from a remote ZIP.

Phase 2B.2. Reads only the ZIP end-of-central-directory record, the central
directory, and the single *_licenses.pkl member. Never downloads the archive.

Safety rules enforced here:
  * HTTPS only.
  * Explicit byte ranges; every read is bounded.
  * A small, fixed number of ranges: ZIP tail, central directory, member local
    header, and the member payload itself.
  * Any HTTP != 206 is a hard failure -- there is no full-download fallback.
  * The retrieved member must match its declared name, uncompressed size and
    CRC-32 before it is written to disk.
  * Output goes to data/interim/, never data/raw/.

Standard library only.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import http.client
import os
import ssl
import struct
import sys
import urllib.parse
import zlib

# ---------------------------------------------------------------- limits ----
MAX_CENTRAL_DIRECTORY = 64 * 1024 * 1024   # bytes
MAX_MEMBER_COMPRESSED = 512 * 1024 * 1024  # bytes
MAX_MEMBER_UNCOMPRESSED = 512 * 1024 * 1024
TAIL_PROBE = 66_000                        # EOCD + comment + ZIP64 locator

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOCATOR_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CENTRAL_SIG = b"PK\x01\x02"
LOCAL_SIG = b"PK\x03\x04"


class RemoteZipError(RuntimeError):
    pass


# ------------------------------------------------------------ http layer ----
def _connection(url: str):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        raise RemoteZipError(f"refusing non-HTTPS URL: {url}")
    return http.client.HTTPSConnection(
        parts.hostname, parts.port or 443, timeout=120,
        context=ssl.create_default_context(),
    )


def head(url: str) -> dict:
    """HEAD request; returns status and the headers we care about."""
    conn = _connection(url)
    try:
        conn.request("HEAD", urllib.parse.urlsplit(url).path,
                     headers={"User-Agent": "PolyCodeGPT-dataset-audit/1.0"})
        resp = conn.getresponse()
        resp.read()
        keys = ("content-length", "last-modified", "etag", "accept-ranges",
                "content-type", "content-disposition")
        return {"status": resp.status, "reason": resp.reason,
                "headers": {k: resp.getheader(k) for k in keys}}
    finally:
        conn.close()


def get_range(url: str, start: int, end: int) -> bytes:
    """Fetch exactly bytes[start..end] (inclusive).  Requires HTTP 206."""
    if start < 0 or end < start:
        raise RemoteZipError(f"bad range {start}-{end}")
    conn = _connection(url)
    try:
        conn.request("GET", urllib.parse.urlsplit(url).path, headers={
            "User-Agent": "PolyCodeGPT-dataset-audit/1.0",
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
        })
        resp = conn.getresponse()
        if resp.status != 206:
            raise RemoteZipError(
                f"expected 206 Partial Content, got {resp.status} {resp.reason}; "
                "refusing to continue (no full-archive fallback)"
            )
        declared = resp.getheader("content-range") or ""
        data = resp.read(end - start + 1)
        if len(data) != end - start + 1:
            raise RemoteZipError(
                f"short read: asked {end - start + 1} bytes, got {len(data)} "
                f"(content-range={declared!r})"
            )
        return data
    finally:
        conn.close()


# ------------------------------------------------------- zip structures -----
def find_eocd(url: str, total_size: int) -> dict:
    """Read the ZIP tail and locate the end-of-central-directory record."""
    start = max(0, total_size - TAIL_PROBE)
    tail = get_range(url, start, total_size - 1)

    pos = tail.rfind(EOCD_SIG)
    if pos < 0 or len(tail) - pos < 22:
        raise RemoteZipError("EOCD record not found in tail")

    (_, disk, cd_disk, n_disk, n_total, cd_size, cd_offset,
     comment_len) = struct.unpack("<4sHHHHIIH", tail[pos:pos + 22])

    result = {
        "cd_size": cd_size,
        "cd_offset": cd_offset,
        "n_entries": n_total,
        "comment_len": comment_len,
        "zip64": False,
    }

    # ZIP64 promotion when any 32-bit field is saturated.
    if 0xFFFFFFFF in (cd_size, cd_offset) or n_total == 0xFFFF:
        loc = tail.rfind(EOCD64_LOCATOR_SIG, 0, pos)
        if loc < 0:
            raise RemoteZipError("ZIP64 locator not found")
        eocd64_offset = struct.unpack("<Q", tail[loc + 8:loc + 16])[0]
        blk = get_range(url, eocd64_offset, eocd64_offset + 56 - 1)
        if blk[:4] != EOCD64_SIG:
            raise RemoteZipError("bad ZIP64 EOCD signature")
        n_total = struct.unpack("<Q", blk[32:40])[0]
        cd_size = struct.unpack("<Q", blk[40:48])[0]
        cd_offset = struct.unpack("<Q", blk[48:56])[0]
        result.update(cd_size=cd_size, cd_offset=cd_offset,
                      n_entries=n_total, zip64=True)

    if cd_size > MAX_CENTRAL_DIRECTORY:
        raise RemoteZipError(
            f"central directory too large ({cd_size} bytes > "
            f"{MAX_CENTRAL_DIRECTORY})"
        )
    return result


def list_members(url: str, eocd: dict) -> list[dict]:
    """Parse the central directory into member records."""
    cd = get_range(url, eocd["cd_offset"], eocd["cd_offset"] + eocd["cd_size"] - 1)
    members, off = [], 0
    while off + 46 <= len(cd) and cd[off:off + 4] == CENTRAL_SIG:
        fields = struct.unpack("<4sHHHHHHIIIHHHHHII", cd[off:off + 46])
        (_, _ver_made, _ver_need, flags, method, _mtime, _mdate, crc,
         csize, usize, name_len, extra_len, comment_len, _disk, _iattr,
         _eattr, local_offset) = fields
        name = cd[off + 46:off + 46 + name_len].decode("utf-8", "replace")
        extra = cd[off + 46 + name_len:off + 46 + name_len + extra_len]

        # ZIP64 extended information extra field (id 0x0001).
        if 0xFFFFFFFF in (csize, usize, local_offset):
            p = 0
            while p + 4 <= len(extra):
                hid, hlen = struct.unpack("<HH", extra[p:p + 4])
                body = extra[p + 4:p + 4 + hlen]
                if hid == 0x0001:
                    q = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", body[q:q + 8])[0]; q += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", body[q:q + 8])[0]; q += 8
                    if local_offset == 0xFFFFFFFF:
                        local_offset = struct.unpack("<Q", body[q:q + 8])[0]; q += 8
                p += 4 + hlen

        members.append({
            "name": name, "flags": flags, "method": method, "crc": crc,
            "csize": csize, "usize": usize, "local_offset": local_offset,
        })
        off += 46 + name_len + extra_len + comment_len
    return members


def extract_member(url: str, m: dict) -> tuple[bytes, dict]:
    """Fetch one member's payload by range and validate it."""
    if m["flags"] & 0x1:
        raise RemoteZipError("member is encrypted; refusing")
    if m["csize"] > MAX_MEMBER_COMPRESSED or m["usize"] > MAX_MEMBER_UNCOMPRESSED:
        raise RemoteZipError(
            f"member too large (compressed={m['csize']}, "
            f"uncompressed={m['usize']})"
        )
    if m["method"] not in (0, 8):
        raise RemoteZipError(f"unsupported compression method {m['method']}")

    # Local header: 30 fixed bytes + name + extra.
    lh = get_range(url, m["local_offset"], m["local_offset"] + 30 - 1)
    if lh[:4] != LOCAL_SIG:
        raise RemoteZipError("bad local header signature")
    name_len, extra_len = struct.unpack("<HH", lh[26:30])
    local_name = get_range(url, m["local_offset"] + 30,
                           m["local_offset"] + 30 + name_len - 1)
    if local_name.decode("utf-8", "replace") != m["name"]:
        raise RemoteZipError(
            f"local name {local_name!r} != central name {m['name']!r}"
        )

    data_start = m["local_offset"] + 30 + name_len + extra_len
    raw = get_range(url, data_start, data_start + m["csize"] - 1)
    payload = raw if m["method"] == 0 else zlib.decompressobj(-zlib.MAX_WBITS).decompress(raw)

    if len(payload) != m["usize"]:
        raise RemoteZipError(
            f"uncompressed size {len(payload)} != declared {m['usize']}"
        )
    crc = binascii.crc32(payload) & 0xFFFFFFFF
    if crc != m["crc"]:
        raise RemoteZipError(f"CRC32 {crc:08x} != declared {m['crc']:08x}")
    return payload, {"compressed_bytes": len(raw), "uncompressed_bytes": len(payload),
                     "crc32": f"{crc:08x}"}


# ------------------------------------------------------------------ cli -----
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True, help="HTTPS URL of the remote ZIP")
    ap.add_argument("--member", help="exact member name to extract")
    ap.add_argument("--list", action="store_true", help="list members matching --grep")
    ap.add_argument("--grep", default="licens", help="substring filter for --list")
    ap.add_argument("--out", help="output file path (required with --member)")
    args = ap.parse_args(argv)

    info = head(args.url)
    if info["status"] != 200:
        print(f"HEAD failed: {info['status']} {info['reason']}", file=sys.stderr)
        return 2
    total = int(info["headers"]["content-length"])
    print(f"HEAD {info['status']} length={total} "
          f"last-modified={info['headers']['last-modified']} "
          f"accept-ranges={info['headers']['accept-ranges']}")

    eocd = find_eocd(args.url, total)
    print(f"central directory: offset={eocd['cd_offset']} size={eocd['cd_size']} "
          f"entries={eocd['n_entries']}")

    members = list_members(args.url, eocd)
    print(f"members parsed: {len(members)}")

    if args.list:
        for m in members:
            if args.grep.lower() in m["name"].lower():
                print(f"  {m['name']}  csize={m['csize']} usize={m['usize']} "
                      f"method={m['method']}")
        return 0

    if not args.member:
        ap.error("--member or --list is required")
    match = [m for m in members if m["name"] == args.member]
    if not match:
        print(f"member not found: {args.member}", file=sys.stderr)
        return 3

    payload, stats = extract_member(args.url, match[0])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as fh:
        fh.write(payload)
    print(f"extracted {args.member}: {stats} sha256={hashlib.sha256(payload).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
