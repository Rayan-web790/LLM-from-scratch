import ast
import hashlib
import json
import re
import sys
import tokenize
import io
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from tokenizers import Tokenizer
from tree_sitter import Language, Parser
import tree_sitter_javascript as tsjs

from polycodegpt.data_v2.schema import serialize_code

INPUT = Path("data/interim/scenario_c_audit/parent_sample.jsonl")
OUT = Path("data/interim/scenario_c_audit")
TOKENIZER_PATH = Path("tokenizer/artifacts/v2/polycodegpt_tokenizer_v2.json")

CANDIDATES = OUT / "extracted_candidates.jsonl"
REPORT = OUT / "extraction_report.json"

MAX_TOKENS = 512
FUNCTION_TYPES = {
    "function_declaration",
    "generator_function_declaration",
    "function_expression",
    "arrow_function",
    "method_definition",
}

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
js_parser = Parser(Language(tsjs.language()))

stats = Counter()
by_language = defaultdict(Counter)
by_type = defaultdict(Counter)
token_lengths = defaultdict(list)
dependency_counts = defaultdict(Counter)
examples = defaultdict(list)
seen_hashes = set()
duplicates = 0
provenance_missing = 0


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def js_raw(data, node):
    if node is None:
        return ""
    return data[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def js_field(node, name):
    return node.child_by_field_name(name) if node else None


def js_walk(node):
    """Iterative depth-first traversal preserving recursive preorder."""
    stack = [node]

    while stack:
        current = stack.pop()
        yield current

        children = current.children
        stack.extend(reversed(children))


def has_function_ancestor(node):
    parent = node.parent
    while parent:
        if parent.type in FUNCTION_TYPES:
            return True
        parent = parent.parent
    return False


def js_owner(node, data):
    parent = node.parent

    while parent and parent.type == "parenthesized_expression":
        parent = parent.parent

    if parent and parent.type == "variable_declarator":
        name_node = js_field(parent, "name")
        declaration = parent.parent
        kind = ""

        if declaration:
            for child in declaration.children:
                if child.type in {"const", "let", "var"}:
                    kind = js_raw(data, child)
                    break

        name = js_raw(data, name_node)
        if name and name_node.type == "identifier":
            return "variable_binding", name, parent, kind

    if parent and parent.type == "assignment_expression":
        left = js_field(parent, "left")
        name = js_raw(data, left)
        if name:
            return "assignment_binding", name, parent, ""

    if parent and parent.type == "pair":
        key = js_field(parent, "key")
        name = js_raw(data, key)
        if name:
            return "object_property", name, parent, ""

    return None


def python_protected_string_lines(source):
    protected = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.STRING:
                if tok.start[0] != tok.end[0]:
                    protected.update(range(tok.start[0] + 1, tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError):
        pass
    return protected


def dedent_ast_block(
    source, start_line, end_line, indent, lines=None, protected=None
):
    if lines is None:
        lines = source.splitlines()
    if protected is None:
        protected = python_protected_string_lines(source)
    prefix = " " * indent
    result = []

    for line_number in range(start_line, end_line + 1):
        line = lines[line_number - 1]

        if line_number in protected:
            result.append(line)
        elif not line.strip():
            result.append("")
        elif line.startswith(prefix):
            result.append(line[len(prefix):])
        else:
            result.append(line)

    return "\n".join(result).rstrip() + "\n"


def python_signature(code):
    try:
        lines = code.splitlines()
        for i, line in enumerate(lines):
            if line.lstrip().startswith(("def ", "async def ")):
                collected = [line]
                if line.count("(") > line.count(")") or not line.rstrip().endswith(":"):
                    for following in lines[i + 1:]:
                        collected.append(following)
                        if following.rstrip().endswith(":"):
                            break
                return "\n".join(collected).strip()
    except Exception:
        pass
    return ""


def python_docstring(node):
    try:
        return ast.get_docstring(node, clean=False) or ""
    except Exception:
        return ""


def python_candidates(record):
    source = record.get("raw_content") or ""
    if not source.strip():
        return []

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        stats["python_parse_failures"] += 1
        return []

    lines = source.splitlines()
    protected = python_protected_string_lines(source)
    results = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes = [node]
            owner_context = ""
            owner_name = None
            category = "top_level_function"
        elif isinstance(node, ast.ClassDef):
            nodes = [
                child for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            owner_context = f"class {node.name}:"
            owner_name = node.name
            category = "class_method"
        else:
            continue

        for fn in nodes:
            decorators = getattr(fn, "decorator_list", [])
            start_line = min(
                [fn.lineno] + [d.lineno for d in decorators]
            )
            end_line = fn.end_lineno

            if not end_line:
                continue

            code = dedent_ast_block(
                source,
                start_line,
                end_line,
                fn.col_offset,
                lines=lines,
                protected=protected,
            )

            if not code.strip():
                continue

            name = fn.name
            signature = python_signature(code)
            docstring = python_docstring(fn)

            try:
                ast.parse(code)
                valid = True
            except SyntaxError:
                valid = False

            results.append({
                "language": "python",
                "category": category,
                "symbol": name,
                "owner": owner_name,
                "context": owner_context,
                "imports": "",
                "docstring": docstring,
                "signature": signature,
                "code": code,
                "syntax_valid": valid,
                "start_line": start_line,
                "end_line": end_line,
                "reconstruction": code,
            })

    return results


def js_candidates(record):
    source = record.get("raw_content") or ""
    if not source.strip():
        return []

    data = source.encode("utf-8")
    root = js_parser.parse(data).root_node

    if root.has_error:
        stats["javascript_parse_failures"] += 1
        return []

    results = []

    for node in js_walk(root):
        if node.type not in FUNCTION_TYPES:
            continue
        if has_function_ancestor(node):
            continue

        snippet = js_raw(data, node)
        if not snippet.strip():
            continue

        name_node = js_field(node, "name")
        name = js_raw(data, name_node) if name_node else ""
        category = ""
        context = ""
        reconstruction = ""

        if node.type in {
            "function_declaration",
            "generator_function_declaration",
        }:
            if not name:
                continue
            category = node.type
            reconstruction = snippet

        elif node.type == "method_definition":
            if not name:
                continue

            parent_type = node.parent.type if node.parent else ""

            if parent_type == "class_body":
                category = "class_method"
                reconstruction = "class __ScenarioC__ {\n" + snippet + "\n}"
            elif parent_type == "object":
                category = "object_method"
                reconstruction = "({\n" + snippet + "\n});"
            else:
                continue

        else:
            owner = js_owner(node, data)
            if not owner:
                continue

            owner_kind, name, owner_node, declaration_kind = owner
            owner_source = js_raw(data, owner_node)
            category = owner_kind

            if owner_kind == "variable_binding":
                reconstruction = (declaration_kind or "const") + " " + owner_source + ";"
            elif owner_kind == "assignment_binding":
                reconstruction = "(" + owner_source + ");"
            elif owner_kind == "object_property":
                reconstruction = "({" + owner_source + "});"

        if not name:
            continue

        wrapped_tree = js_parser.parse(reconstruction.encode("utf-8"))
        valid = not wrapped_tree.root_node.has_error

        results.append({
            "language": "javascript",
            "category": category,
            "symbol": name,
            "owner": None,
            "context": context,
            "imports": "",
            "docstring": "",
            "signature": snippet.splitlines()[0][:500],
            "code": reconstruction,
            "syntax_valid": valid,
            "start_line": node.start_point.row + 1,
            "end_line": node.end_point.row + 1,
            "reconstruction": reconstruction,
        })

    return results


def dependency_flags(language, code):
    flags = []

    patterns = {
        "this": r"\bthis\b",
        "super": r"\bsuper\b",
        "await": r"\bawait\b",
        "yield": r"\byield\b",
        "require": r"\brequire\s*\(",
        "jsx": r"</?[A-Z][A-Za-z0-9.]*[\s/>]",
    }

    for label, pattern in patterns.items():
        if re.search(pattern, code):
            flags.append(label)

    if language == "python":
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id in {"self", "cls"}:
                        flags.append(node.id)
        except SyntaxError:
            pass

    return sorted(set(flags))


def main():
    global duplicates, provenance_missing

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    OUT.mkdir(parents=True, exist_ok=True)

    parent_count = Counter()
    candidate_count = 0
    retained_count = 0
    output_records = []

    with INPUT.open("r", encoding="utf-8") as f:
        parents = [json.loads(line) for line in f if line.strip()]

    print("Loaded parent files:", len(parents), flush=True)

    for parent_index, parent in enumerate(parents, 1):
        language = parent.get("language")
        parent_count[language] += 1

        if language == "python":
            extracted = python_candidates(parent)
        elif language == "javascript":
            extracted = js_candidates(parent)
        else:
            continue

        for item in extracted:
            candidate_count += 1

            if not item["syntax_valid"]:
                by_language[language]["reconstruction_invalid"] += 1
                continue

            code_hash = sha(language + "\0" + item["reconstruction"])
            if code_hash in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(code_hash)

            required_provenance = (
                "source_id", "source_snapshot", "repo", "path",
                "license_id", "license_evidence", "source_url",
                "origin_partition", "shard",
            )
            missing = [
                key for key in required_provenance
                if not parent.get(key)
            ]
            if missing:
                provenance_missing += 1

            serialized = serialize_code(
                language,
                "function",
                file=parent.get("path", ""),
                context=item["context"],
                imports=item["imports"],
                docstring=item["docstring"],
                signature=item["signature"],
                code=item["code"],
            )
            ids = tokenizer.encode(serialized).ids
            token_count = len(ids)

            flags = dependency_flags(language, item["code"])
            for flag in flags:
                dependency_counts[language][flag] += 1

            by_language[language]["extracted_valid"] += 1
            by_type[language][item["category"]] += 1
            token_lengths[language].append(token_count)

            if token_count <= MAX_TOKENS:
                by_language[language]["fits_512"] += 1
            else:
                by_language[language]["over_512"] += 1

            record_id = sha(
                parent.get("source_id", "") + "\0" +
                parent.get("repo", "") + "\0" +
                parent.get("path", "") + "\0" +
                item["category"] + "\0" +
                item["symbol"] + "\0" +
                str(item["start_line"]) + "\0" +
                code_hash
            )

            output = {
                "record_id": record_id,
                "parent_record_id": None,
                "source_id": parent.get("source_id"),
                "source_snapshot": parent.get("source_snapshot"),
                "repo": parent.get("repo"),
                "path": parent.get("path"),
                "license_id": parent.get("license_id"),
                "license_evidence": parent.get("license_evidence"),
                "source_url": parent.get("source_url"),
                "origin_partition": parent.get("origin_partition"),
                "partition": parent.get("origin_partition"),
                "language": language,
                "task": "function",
                "category": item["category"],
                "symbol": item["symbol"],
                "parent_category": parent.get("category"),
                "content_kind": "code",
                "extraction_method": "python_ast" if language == "python" else "tree_sitter",
                "start_line": item["start_line"],
                "end_line": item["end_line"],
                "parent_shard": parent.get("shard"),
                "parent_source_index": parent.get("source_index"),
                "parent_sha256": sha(parent.get("raw_content") or ""),
                "code_sha256": code_hash,
                "dependency_flags": flags,
                "token_count": token_count,
                "fits_512": token_count <= MAX_TOKENS,
                "schema_version": "2.0.0",
                "text": serialized,
                "code": item["code"],
                "syntax_valid": item["syntax_valid"],
            }

            output_records.append(output)
            retained_count += 1

        if parent_index % 100 == 0:
            print(
                f"Processed {parent_index}/{len(parents)} parents | "
                f"valid candidates {retained_count}",
                flush=True,
            )

    with CANDIDATES.open("w", encoding="utf-8", newline="\n") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def distribution(values):
        if not values:
            return {}
        ordered = sorted(values)
        n = len(ordered)

        def percentile(p):
            return ordered[min(n - 1, int((n - 1) * p))]

        return {
            "count": n,
            "min": ordered[0],
            "median": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "max": ordered[-1],
            "within_512": sum(v <= 512 for v in ordered),
            "over_512": sum(v > 512 for v in ordered),
        }

    report = {
        "status": "AUDIT_ONLY",
        "input": str(INPUT),
        "parent_files": len(parents),
        "parent_files_by_language": dict(parent_count),
        "candidate_occurrences": candidate_count,
        "retained_syntax_valid_unique_candidates": retained_count,
        "invalid_reconstructions": dict(stats),
        "duplicates_removed": duplicates,
        "missing_provenance_records": provenance_missing,
        "languages": {
            lang: {
                "counts": dict(by_language[lang]),
                "types": dict(by_type[lang]),
                "token_distribution": distribution(token_lengths[lang]),
                "dependency_flags": dict(dependency_counts[lang]),
            }
            for lang in ("python", "javascript")
        },
        "tokenizer": str(TOKENIZER_PATH),
        "vocabulary_size": tokenizer.get_vocab_size(),
        "context_limit": MAX_TOKENS,
        "output": str(CANDIDATES),
    }

    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== SCENARIO C EXTRACTION SUMMARY ==========")
    print(json.dumps(report, indent=2))
    print("\nCandidates:", CANDIDATES)
    print("Report:", REPORT)


if __name__ == "__main__":
    main()
