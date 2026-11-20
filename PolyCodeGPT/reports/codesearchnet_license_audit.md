# CodeSearchNet — License Metadata Audit (Phase 2B.2)

> **Scope.** Licensing metadata only. No code examples, docstrings, or full archives
> were downloaded. No preprocessing, tokenizer or model work was performed.
>
> **Disclaimer.** This is a dataset-engineering classification exercise, **not legal
> advice**. Licence families are detected with heuristics; every binary decision below is
> documented so it can be re-checked. Consult qualified counsel before commercial use.

**Verdict:** `METADATA AUDIT COMPLETED WITH WARNINGS` — metadata was retrieved and
classified for both languages, but the officially documented S3 archives are no longer
publicly readable, so the DOI-registered Zenodo deposit was used instead and archive
parity with S3 `v2` could not be verified.

---

## 1. Source details

| Item | Value |
|---|---|
| Repository | https://github.com/github/CodeSearchNet |
| Status | **archived** (read-only); last push `2022-01-31T09:25:07Z`; created `2019-02-28` |
| Repository licence | MIT — applies to the **repository code only**, not to dataset examples |
| Challenge status | concluded (no new submissions) |
| Paper | arXiv:1909.09436 — *CodeSearchNet Challenge: Evaluating the State of Semantic Code Search* |
| Data documentation | `resources/README.md` |
| Documented archive pattern | `https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{python,java,go,php,javascript,ruby}.zip` |
| Python archive filename | `python.zip` |
| JavaScript archive filename | `javascript.zip` |
| `_licenses.pkl` location inside archive | **archive root**, i.e. `python_licenses.pkl` and `javascript_licenses.pkl` unzip directly into the data directory alongside the `python/`, `javascript/` folders |
| Official mirror referenced by project docs | Zenodo DOI `10.5281/zenodo.7908468` (record 7908468, six per-language zips, access right "open", deposit metadata licence `CC-BY-4.0`) |

### Documented JSONL schema (unchanged by this audit; verified from the official README)

`repo`, `path`, `func_name`, `original_string`, `language`, `code`, `code_tokens`,
`docstring`, `docstring_tokens`, `sha`, `partition`, `url`.

Provenance fields: `repo` (owner/name), `path` (file path in that repo), `url`
(GitHub permalink with line range), `sha` (commit — documented as currently unused).

### Documented pickle structure — and what it actually is

`resources/README.md` states the `_licenses.pkl` files are dictionaries mapping
`owner/name` → a **tuple** `(path, license content)`.

**Observed reality (verified):** the value is a **list of `(path, content)` tuples**, one
entry per licence-like file found in that repository. This matches
`function_parser/function_parser/fetch_licenses.py`, which builds
`licenses[nwo] = licenses.get(nwo, []) + [(path, content)]`.

Consequence: any consumer written against the README's "single tuple" description will
mis-parse these files. 1,284 Python repos and 2,077 JavaScript repos carry more than one
entry.

---

## 2. Archive metadata

### 2.1 Documented S3 URLs — NOT AVAILABLE

| Property | `python.zip` | `javascript.zip` |
|---|---|---|
| HEAD status | `301 Moved Permanently` → `403 Forbidden` | `301 Moved Permanently` → `403 Forbidden` |
| Redirect target | `code-search-net.s3.amazonaws.com` (`x-amz-bucket-region: us-west-2`) | same |
| Final body | `<Error><Code>AccessDenied</Code>…` | same |
| Content-Length | **unavailable** | **unavailable** |
| Last-Modified | **unavailable** | **unavailable** |
| ETag | **unavailable** | **unavailable** |
| Accept-Ranges | **unavailable** | **unavailable** |
| Content-Type | `application/xml` (error document) | `application/xml` (error document) |
| Bucket listing | `403 AccessDenied` | — |

The bucket is reachable and answers with genuine S3 error documents (valid
`x-amz-request-id`), so this is a **permission** failure, not a network failure. Public
access to `code-search-net` has been withdrawn; the URLs in the official README are dead.

Because the objects cannot be read at all, S3 byte-range support could not be measured —
it is neither confirmed nor refuted.

### 2.2 Zenodo deposit — AVAILABLE, range-capable

`https://zenodo.org/api/records/7908468/files/{python,javascript}.zip/content`

| Property | `python.zip` | `javascript.zip` |
|---|---|---|
| HEAD status | `200 OK` | `200 OK` |
| Content-Length | 940,909,997 | 1,664,713,350 |
| Last-Modified | `Sat, 08 Aug 2026 21:59:07 GMT` | same |
| ETag | not sent | not sent |
| Accept-Ranges on HEAD | not advertised (nginx) | not advertised |
| Content-Type | `application/octet-stream` | `application/octet-stream` |
| Range probe `bytes=0-0` | **`206 Partial Content`**, `Content-Range: bytes 0-0/940909997`, 1 byte | **`206`**, `Content-Range: bytes 0-0/1664713350` |
| Central directory | offset 940,907,361, size 2,614, 24 members | offset 1,664,711,604, size 1,648, 15 members |

`Accept-Ranges` is absent from HEAD but range requests are honoured, so capability was
established by probe rather than by header.

### 2.3 Archive-size discrepancy (documented vs actual)

| Claim | Value |
|---|---|
| Official README ("download data (~3.5GB)") | ~3.5 GB for all six languages |
| Project `DATASET_LICENSE_AUDIT` note | ~5.1 GB compressed (all languages) |
| **Measured, Zenodo, all six zips** | **6,123,192,768 bytes = 5.70 GiB (6.12 GB)** |
| **Measured, Zenodo, Python + JavaScript only** | **2,605,623,347 bytes = 2.43 GiB (2.61 GB)** |

Both documented estimates understate the real archives. The S3 objects may have differed,
but that cannot be checked while access is denied.

---

## 3. Retrieval method

`scripts/fetch_csn_license_metadata.py` (stdlib only) reads a remote ZIP without
downloading it:

1. `HEAD` → total size; refuse if not `200`.
2. Range-read the last 66,000 bytes → locate the EOCD record (ZIP64 handled).
3. Range-read the central directory only (bounded to 64 MiB).
4. Locate the exact member, then range-read its local header + compressed payload.
5. Validate: local name == central name, uncompressed size == declared size,
   CRC-32 == declared CRC.
6. Any non-`206` response aborts the run — there is no full-download fallback.

Bytes actually transferred per archive (extraction run):

| Archive | Ranges issued | Bytes read |
|---|---|---|
| `python.zip` | tail 66,000 + CD 2,614 + header/name 49 + member 32,666,588 | 32,735,251 (0.0035 % of 940,909,997) |
| `javascript.zip` | tail 66,000 + CD 1,648 + header/name 53 + member 32,315,567 | 32,383,268 (0.0019 % of 1,664,713,350) |

Output (`data/interim/`, never `data/raw/`):

| File | Bytes | SHA-256 |
|---|---|---|
| `python_licenses.pkl` | 134,172,919 | `404333e78827bd1743073251c65333239e625820d917bbcaf821dd0fd4433897` |
| `javascript_licenses.pkl` | 196,114,558 | `7ae57111b6d275f0d0b9a832d7c89003b5245f6fe9627658eb469f276d6181f1` |

Extracted members, per the ZIP central directory:

| Member | Compressed | Uncompressed | CRC-32 |
|---|---|---|---|
| `python_licenses.pkl` | 32,666,588 | 134,172,919 | `6980bb9f` ✅ |
| `javascript_licenses.pkl` | 32,315,567 | 196,114,558 | `145bda7a` ✅ |

No member other than the licence pickles was fetched; `python_dedupe_definitions_v2.pkl`
(2.9 GB uncompressed) and every `.jsonl.gz` shard were left untouched.

---

## 4. Pickle security checks

`scripts/inspect_csn_licenses.py` applies two independent guards **before** any
deserialization:

| Check | Result |
|---|---|
| Size cap (512 MiB) | passed (134 MB / 196 MB) |
| `pickletools.genops()` full opcode scan | passed |
| Protocol | 3 (binary, both files) |
| Distinct opcode types | 13 (both files) |
| `GLOBAL` / `STACK_GLOBAL` | 0 present |
| `REDUCE` / `BUILD` / `OBJ` / `INST` | 0 present |
| `NEWOBJ` / `NEWOBJ_EX` | 0 present |
| `EXT1` / `EXT2` / `EXT4` | 0 present |
| `PERSID` / `BINPERSID` | 0 present |
| Restricted `Unpickler.find_class` | installed; raises on any global |
| Unpickle result | `dict`, both files |

Opcode mix (Python file): `LONG_BINPUT`, `BINUNICODE`, `TUPLE2`, `EMPTY_LIST`,
`APPEND`, `MARK`, `APPENDS`, `BINPUT` and four more — i.e. plain containers and strings
only, with no callable or import machinery. Both files are structurally benign, and the
restricted unpickler means a malicious payload would still fail even if the static scan
had missed something.

---

## 5. Structure validation

| Property | Python | JavaScript |
|---|---|---|
| Top-level type | `dict` | `dict` |
| Records (repositories) | 17,059 | 50,174 |
| Key type | `str` ×17,059 | `str` ×50,174 |
| Key shape `owner/name` | 17,059 / 17,059 ✅ | 50,174 / 50,174 ✅ |
| Duplicate keys | 0 (dict semantics) | 0 |
| Value type | `list` ×17,059 | `list` ×50,174 |
| Licence-file records | 21,259 | 73,600 |
| Repos with >1 licence file | 1,284 | 2,077 |
| Empty licence content | 7 | 2 |
| Unparsable-invalid values | 0 | 0 |
| Records containing non-UTF-8 bytes (lone surrogates from `surrogateescape`) | 43 | 84 |

The non-UTF-8 records are prose licences with a stray binary byte. They are kept verbatim
and flagged in the inventory's `review_notes`; any consumer must open them with
`errors="surrogateescape"` or text will be silently mangled.

**Metadata completeness cannot be fully measured.** The pickles contain only repositories
where a licence file was found; the set of repositories with *no* licence file is not
represented, and recovering it would require the JSONL shards (out of scope). Therefore
`METADATA_MISSING` counts below are unknown, not zero.

---

## 6. Classification criteria

Applied per licence-file record, in this order:

1. Empty/whitespace content → `NO_LICENSE_FOUND`.
2. Family detection, resolved hierarchically (not additively):
   * GNU family — requires a real GNU grant clause; the name alone in a long MPL/EPL
     document (e.g. MPL-2.0 Exhibit B) is treated as a citation, not a licence body.
     One value: `AGPL` > `LGPL` > `GPL-3.0` > `GPL-2.0` > `GPL`.
   * `MPL-2.0`, `EPL`, `CC-BY-SA`.
   * Permissive set with subsumption (`BSD-3` before `BSD-2`, `ISC` before `0BSD`).
3. `.. include::`-only files → `INDIRECT`.
4. Category mapping:

| Category | Rule |
|---|---|
| `CLEAR_PERMISSIVE_CANDIDATE` | exactly one recognised permissive family, or several that are **all** in the permissive set (dual/multi-licensed, no copyleft) |
| `COPYLEFT_REVIEW_REQUIRED` | any GNU/MPL/EPL/CC-BY-SA family, or a permissive+copyleft mix |
| `PROPRIETARY_OR_RESTRICTIVE` | restrictive language ("all rights reserved", "proprietary", "confidential") **and** no permission grant |
| `NO_LICENSE_FOUND` | entry exists but content is empty |
| `AMBIGUOUS` | unrecognised family, short stub, pointer file, attribution-only licence (CC-BY), or several unrelated families |
| `METADATA_MISSING` | repository absent from the pickle — **not computable from this data** |

Repo-level category = **most restrictive** record in that repo (`CLEAR_PERMISSIVE` <
`COPYLEFT` < `PROPRIETARY` < `AMBIGUOUS` < `NO_LICENSE`). A repo is never permissive if
any one of its licence files is not.

---

## 7. Distribution — Python

Records 21,259 · repositories 17,059

| Category | Records | % |
|---|---|---|
| `CLEAR_PERMISSIVE_CANDIDATE` | 17,002 | 79.98% |
| `COPYLEFT_REVIEW_REQUIRED` | 3,576 | 16.82% |
| `AMBIGUOUS` | 642 | 3.02% |
| `PROPRIETARY_OR_RESTRICTIVE` | 32 | 0.15% |
| `NO_LICENSE_FOUND` | 7 | 0.03% |

| Family | Records | % |
|---|---|---|
| MIT | 10,675 | 50.21% |
| Apache-2.0 | 2,682 | 12.62% |
| BSD-3-Clause | 2,001 | 9.41% |
| AGPL | 1,759 | 8.27% |
| LGPL | 1,094 | 5.15% |
| BSD-2-Clause | 962 | 4.53% |
| ISC | 439 | 2.07% |
| UNKNOWN | 375 | 1.76% |
| CC-BY-SA | 311 | 1.46% |
| MPL-2.0 | 168 | 0.79% |
| INDIRECT | 149 | 0.70% |
| GPL-2.0 | 123 | 0.58% |
| OTHER | 111 | 0.52% |
| WTFPL | 55 | 0.26% |
| GPL-3.0 | 34 | 0.16% |
| OTHER-RESTRICTIVE | 32 | 0.15% |
| CC0-1.0 | 27 | 0.13% |
| EPL | 19 | 0.09% |
| Zlib | 15 | 0.07% |
| Unlicense | 13 | 0.06% |
| BSL-1.0 | 10 | 0.05% |
| PSF-2.0 | 9 | 0.04% |
| CC-BY-4.0 | 6 | 0.03% |
| `NONE-IN-METADATA` | 7 | 0.03% |
| multi-family, all permissive (109 files) | — | 0.51% of records |
| multi-family, includes copyleft/other (66 files) | — | 0.31% of records |

Repository level (17,059): `CLEAR_PERMISSIVE_CANDIDATE` 13,885 (81.4%),
`COPYLEFT_REVIEW_REQUIRED` 2,781 (16.3%), `AMBIGUOUS` 383 (2.2%),
`PROPRIETARY_OR_RESTRICTIVE` 9 (0.1%), `NO_LICENSE_FOUND` 1 (0.0%).

## 8. Distribution — JavaScript

Records 73,600 · repositories 50,174

| Category | Records | % |
|---|---|---|
| `CLEAR_PERMISSIVE_CANDIDATE` | 70,432 | 95.70% |
| `COPYLEFT_REVIEW_REQUIRED` | 2,368 | 3.22% |
| `AMBIGUOUS` | 749 | 1.02% |
| `PROPRIETARY_OR_RESTRICTIVE` | 49 | 0.07% |
| `NO_LICENSE_FOUND` | 2 | 0.00% |

| Family | Records | % |
|---|---|---|
| MIT | 58,914 | 80.05% |
| Apache-2.0 | 4,696 | 6.38% |
| ISC | 2,978 | 4.05% |
| BSD-2-Clause | 1,665 | 2.26% |
| BSD-3-Clause | 1,564 | 2.12% |
| AGPL | 1,126 | 1.53% |
| LGPL | 688 | 0.93% |
| UNKNOWN | 506 | 0.69% |
| MPL-2.0 | 311 | 0.42% |
| OTHER | 203 | 0.28% |
| WTFPL | 136 | 0.18% |
| CC0-1.0 | 98 | 0.13% |
| OTHER-RESTRICTIVE | 49 | 0.07% |
| Zlib | 40 | 0.05% |
| GPL-2.0 | 39 | 0.05% |
| Unlicense | 34 | 0.05% |
| CC-BY-4.0 | 28 | 0.04% |
| CC-BY-SA | 22 | 0.03% |
| EPL | 16 | 0.02% |
| GPL-3.0 | 9 | 0.01% |
| multi-family, all permissive (303 files) | — | 0.41% of records |
| multi-family, includes copyleft/other (165 files) | — | 0.22% of records |

Repository level (50,174): `CLEAR_PERMISSIVE_CANDIDATE` 47,848 (95.4%),
`COPYLEFT_REVIEW_REQUIRED` 1,853 (3.7%), `AMBIGUOUS` 443 (0.9%),
`PROPRIETARY_OR_RESTRICTIVE` 28 (0.1%), `NO_LICENSE_FOUND` 2 (0.0%).

---

## 9. Ambiguous and notable cases

| Case | Example | Why it is not auto-approved |
|---|---|---|
| Pointer file | `pypa/pipenv` `…/packaging/LICENSE`: "made available under the terms of *either* of the licenses found in LICENSE.APACHE or LICENSE.BSD" | The file names licences but contains no licence text |
| RST stub / include | `lehins/python-wepay` `docs/source/license.rst`: `.. include:: ../../LICENSE` | Real licence lives in another file (149 Python, 2 JavaScript records) |
| Debian copyright format | `PyAr/fades` `pkg/debian/copyright` | Structured multi-licence file; must be parsed, not pattern-matched |
| Heading only | `kashifrazzaqui/vyked` `docs/source/license.rst` = "License\n=======" (15 bytes) | No grant of any kind |
| Restrictive notice | `gorakhargosh/pathtools` `debian/copyright`: "Copyright … All rights reserved." | No permission grant at all |
| Third-party pointer | `sfgraves/starlink-pyhds` `hdf5/…/COPYING` | Refers to a full notice elsewhere; HDF5 terms unverified |
| Attribution licence | CC-BY-4.0 (6 Python, 28 JS records) | Permits use but imposes attribution; obligations unconfirmed |
| Mixed bundle | `MULTIPLE:BSD-3-Clause+LGPL+MIT+MPL-2.0` | Single file containing several unrelated licences |
| Non-UTF-8 text | 43 Python + 84 JS records | Stray binary byte; kept verbatim, flagged |

Note the AGPL share in Python (8.3% of records) — the largest copyleft block, and the one
most likely to matter for a permissively-licensed corpus.

---

## 10. Recommended filtering strategy

1. **Include only `CLEAR_PERMISSIVE_CANDIDATE` repositories.** Exclude every repo whose
   *any* licence file is copyleft, restrictive, ambiguous, or empty. Expected keep rate:
   ~81.4% of Python repos, ~95.4% of JavaScript repos.
2. **Filter at repository level, not file level.** A permissive `LICENSE` at the repo
   root does not neutralise a GPL `COPYING` in a vendored subdirectory.
3. **Do not treat `_licenses.pkl` presence as proof of licensing.** It records what files
   were found, not whether the found text is a valid grant.
4. **Treat `AMBIGUOUS` and `NO_LICENSE_FOUND` as excluded** until individually reviewed;
   they are 2.2% of Python and 0.9% of JavaScript repos — cheap to review manually.
5. **Re-check repo-level vs file-level attribution** before publishing derived data:
   CodeSearchNet examples are file-level, and a file may sit under a vendored
   third-party licence even when the repo root is MIT.
6. **Keep the licence text with the record.** The inventory carries the verbatim text and
   its SHA-256 so a future dispute can be resolved without re-downloading the archive.
7. **Preserve the Zenodo DOI and retrieval date** in any downstream manifest; the S3
   provenance cannot be re-established while the bucket is private.

---

## 11. Files created

| File | Size | Purpose |
|---|---|---|
| `data/interim/codesearchnet_license_metadata/python_licenses.pkl` | 134,172,919 B | Verbatim metadata member (not training data) |
| `data/interim/codesearchnet_license_metadata/javascript_licenses.pkl` | 196,114,558 B | Verbatim metadata member |
| `data/manifests/codesearchnet_license_inventory.csv` | 343,770,419 B | 94,859 rows × 10 columns, **full licence text included** |
| `data/manifests/codesearchnet_license_repo_summary.csv` | 4,906,642 B | 67,233 rows, one per repository, with worst-case category |
| `scripts/fetch_csn_license_metadata.py` | — | Bounded remote-ZIP reader |
| `scripts/inspect_csn_licenses.py` | — | Opcode scan + restricted unpickler |
| `scripts/build_csn_license_inventory.py` | — | Classifier and inventory builder |
| `reports/codesearchnet_license_audit.md` | — | This report |

⚠️ **The full-text inventory is 344 MB.** It is the evidence artefact, not a pipeline
input; it must not be committed to Git (`data/manifests/` is currently tracked). Either
git-ignore it and keep the 4.9 MB repo summary under version control, or store it
compressed outside the repository.

---

## 12. Limitations

1. **Primary archive unavailable.** `s3.amazonaws.com/code-search-net/…` returns
   `AccessDenied`; its `Content-Length`, `ETag`, `Last-Modified` and range behaviour are
   therefore unknown, and the task's primary source could not be measured.
2. **Mirror substitution.** Metadata came from the DOI-registered Zenodo deposit
   (`10.5281/zenodo.7908468`), not from the S3 objects the README names. The Zenodo zips
   contain the expected `v2` layout (`{lang}/final/jsonl/{train,valid,test}`), but
   **byte-level parity with S3 `v2` is unverified** and cannot be verified while the
   bucket is private. This is the main warning behind the verdict.
3. **Zenodo deposit is dated 2019-09-30** (files re-uploaded `2026-08-08`) and its
   metadata licence is `CC-BY-4.0`; that covers the deposit record, **not** the
   repository-level licences analysed here.
4. **`METADATA_MISSING` is unknown, not zero.** Repositories with no licence file are
   absent from the pickle by construction.
5. **Classification is heuristic.** Family detection is phrase-based with documented
   precedence; a licence rewritten in unusual wording lands in `AMBIGUOUS` rather than
   being guessed. ~1.0–3.0% of records are unresolved.
6. **No legal conclusion is drawn.** "Permissive" here means "recognised permissive
   family, no copyleft or restrictive language detected".
7. **Examples were never fetched**, so repository/file-level consistency between the
   metadata and the actual corpus rows is untested.

---

## 13. Next step (requires approval)

1. Decide whether the Zenodo deposit is an acceptable provenance substitute for the dead
   S3 URLs; if not, the CodeSearchNet path is blocked until access is restored.
2. Get approval to download the Python and JavaScript archives (2.61 GB combined) so the
   inventory can be matched against the real corpus rows (`repo`, `path`) and
   `METADATA_MISSING` can finally be computed.
3. Review the 383 Python + 443 JavaScript `AMBIGUOUS` repositories (and 2,781 + 1,853
   copyleft repositories) before any of them is considered.
4. Then, and only then, proceed to collection under the same integrity rules as Phase 2B.1.
