"""Inspect shard 14 to find large/slow-to-parse Python files."""
import sys
from pathlib import Path
import pyarrow.parquet as pq

sys.path.insert(0, "src")
sys.path.insert(0, ".")
from polycodegpt.data_v2.language import normalize_repo
from polycodegpt.data_v2.constants import LICENSE_ALLOWLIST

shard = Path("data/raw/v2/codeparrot_github_code/train-00013-of-01126.parquet")
pf = pq.ParquetFile(shard)

py_files = []
for rg in range(pf.num_row_groups):
    tbl = pf.read_row_group(rg, columns=["repo_name", "path", "size", "license", "content"])
    for rec in tbl.to_pylist():
        p = rec.get("path") or ""
        if not p.endswith(".py"):
            continue
        content = rec.get("content") or ""
        repo = normalize_repo(rec.get("repo_name") or "")
        lic = (rec.get("license") or "").strip().lower()
        allowed = lic in LICENSE_ALLOWLIST
        nlines = len(content.splitlines())
        py_files.append((len(content), allowed, repo, p, nlines))

py_files.sort(reverse=True)
print("Top 20 Python files in shard 14 by content length:")
print(f"{'length':>10}  {'allowed':>7}  {'lines':>6}  path")
for sz, allowed, repo, path, nlines in py_files[:20]:
    print(f"{sz:>10}  {str(allowed):>7}  {nlines:>6}  {repo}/{path}")
print()
print("Total Python files:", len(py_files))
print("Allowed license:", sum(1 for _, a, *_ in py_files if a))
print("Max size:", py_files[0][0] if py_files else 0)
print("Files > 100KB:", sum(1 for sz, *_ in py_files if sz > 100_000))
print("Files > 500KB:", sum(1 for sz, *_ in py_files if sz > 500_000))
print("Files > 1MB:  ", sum(1 for sz, *_ in py_files if sz > 1_000_000))
