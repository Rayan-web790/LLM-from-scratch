import json
import ast
from pathlib import Path
from collections import Counter

from tokenizers import Tokenizer
from tree_sitter import Language, Parser
import tree_sitter_javascript as tsjs

p = Path("data/interim/scenario_c_audit/extracted_candidates.jsonl")
tok = Tokenizer.from_file(
    "tokenizer/artifacts/v2/polycodegpt_tokenizer_v2.json"
)
js = Parser(Language(tsjs.language()))

counts = Counter()
lengths = Counter()
errors = []
longest = []

required = [
    "record_id", "source_id", "source_snapshot", "repo", "path",
    "license_id", "license_evidence", "source_url",
    "origin_partition", "parent_sha256", "code_sha256", "text", "code"
]

seen_ids = set()

with p.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        r = json.loads(line)
        lang = r["language"]
        counts[f"{lang}_records"] += 1

        missing = [k for k in required if not r.get(k)]
        if missing:
            errors.append((line_no, "missing fields", missing))

        if r["record_id"] in seen_ids:
            errors.append((line_no, "duplicate record_id", r["record_id"]))
        seen_ids.add(r["record_id"])

        actual = len(tok.encode(r["text"]).ids)
        if actual != r["token_count"]:
            errors.append((line_no, "token mismatch", actual, r["token_count"]))

        if actual <= 512:
            lengths[f"{lang}_within_512"] += 1
        else:
            lengths[f"{lang}_over_512"] += 1
            longest.append((actual, lang, r["repo"], r["path"], r["symbol"]))

        if "<|bos|>" not in r["text"] or "<|eos|>" not in r["text"]:
            errors.append((line_no, "missing BOS/EOS"))

        code = r["code"]
        if lang == "python":
            try:
                ast.parse(code)
            except SyntaxError as e:
                errors.append((line_no, "Python syntax", str(e)))
        elif lang == "javascript":
            if js.parse(code.encode("utf-8")).root_node.has_error:
                errors.append((line_no, "JavaScript syntax"))

print("========== SCENARIO C INTEGRITY ==========")
print("Records:", sum(counts.values()))
print("Language counts:", dict(counts))
print("Token fit:", dict(lengths))
print("Unique record IDs:", len(seen_ids))
print("Integrity errors:", len(errors))

for error in errors[:15]:
    print("ERROR:", error)

print("\nLongest examples:")
for item in sorted(longest, reverse=True)[:10]:
    print(item)

print("\nINTEGRITY CHECK COMPLETE")
