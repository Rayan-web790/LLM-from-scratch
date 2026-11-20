import ast
from pathlib import Path

p = Path("scripts/audit_scenario_c_extraction.py")
source = p.read_text(encoding="utf-8-sig")
tree = ast.parse(source)

print("========== EXISTING EXTRACTOR INTERFACE ==========")
print("File:", p)
print("Lines:", len(source.splitlines()))

for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = [a.arg for a in node.args.args]
        print(f"FUNCTION {node.name}({', '.join(args)})")
    elif isinstance(node, ast.ClassDef):
        print(f"CLASS {node.name}")

print("\n========== TOP-LEVEL EXECUTION ==========")
for node in tree.body:
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        print("Top-level call at line:", node.lineno)
    elif isinstance(node, ast.If):
        test = ast.get_source_segment(source, node.test)
        print("Top-level if at line:", node.lineno, "|", test)

print("\nINTERFACE INSPECTION COMPLETE")
