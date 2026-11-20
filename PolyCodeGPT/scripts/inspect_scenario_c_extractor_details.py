import ast
from pathlib import Path

p = Path("scripts/audit_scenario_c_extraction.py")
source = p.read_text(encoding="utf-8-sig")
tree = ast.parse(source)

wanted = {
    "python_candidates",
    "js_candidates",
    "dependency_flags",
    "main",
}

for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
        print("\n" + "=" * 25, node.name, "=" * 25)
        print(ast.get_source_segment(source, node))

print("\n" + "=" * 25, "IMPORTS AND CONSTANTS", "=" * 25)
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        print(ast.get_source_segment(source, node))
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        names = []
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
        if names:
            print(ast.get_source_segment(source, node))

