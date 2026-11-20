import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

with open(project_root / 'scripts' / 'evaluate_phase10_benchmark.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add unique_token_ratio_generated function after unique_token_ratio function
old_utr = '''def unique_token_ratio(ids: list) -> float:
    """Fraction of distinct tokens in the code sequence."""
    n = count_code_tokens(ids)
    if n == 0:
        return 0.0
    code_start = ids.index(CODE_ID) + 1
    code_end = None
    for i in range(code_start, len(ids)):
        if ids[i] == EOS_ID:
            code_end = i
            break
    if code_end is None:
        code_tail = ids[code_start:]
    else:
        code_tail = ids[code_start:code_end]
    if not code_tail:
        return 0.0
    distinct = len(set(code_tail))
    return distinct / n'''

new_utr = old_utr + '''


def unique_token_ratio_generated(generated_ids: list) -> float:
    """Unique-token ratio for generated code tokens only (BPE).

    Only model-generated code token IDs contribute. Excludes prompt,
    language markers, docstring markers, code stub, and EOS.
    Empty generation returns 0.0.
    """
    if not generated_ids:
        return 0.0
    distinct = len(set(generated_ids))
    return distinct / len(generated_ids)'''

assert old_utr in content, 'unique_token_ratio not found'
content = content.replace(old_utr, new_utr, 1)

# 2. Fix repeated_4gram_ratio_tokens to count occurrences (not distinct types)
old_r4 = '''def repeated_4gram_ratio_tokens(token_ids: list) -> float:
    """Fraction of overlapping 4-token sequences (BPE) that appear multiple times.

    Uses BPE token IDs from tokenizer v1.1.0, as specified in the Phase 10
    benchmark design. Measures only the generated code tokens (not prompt).
    """
    tokens = list(token_ids)
    if len(tokens) < 8:
        return 0.0
    grams = [tuple(tokens[i : i + 4]) for i in range(len(tokens) - 3)]
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    total = len(grams)
    return repeated / total'''

# The current implementation already uses sum(c for ...) from previous fix
# But let me verify it's correct
assert old_r4 in content, 'repeated_4gram_ratio_tokens not found'
# Already correct - no change needed

# 3. Update run_benchmark to use generated-only unique token ratio
old_utr_call = '''        utr = unique_token_ratio(full_ids)'''
new_utr_call = '''        utr = unique_token_ratio_generated(generated_ids)'''
assert old_utr_call in content, 'utr call not found'
content = content.replace(old_utr_call, new_utr_call, 1)

# Also in JS section
old_utr_call2 = '''        utr = unique_token_ratio(full_ids)'''
# This was already replaced by the first replace (global), but let me check
count = content.count('utr = unique_token_ratio_generated(generated_ids)')
assert count == 2, f'Expected 2 replacements, got {count}'

# Remove the now-unused full_ids variable lines (optional, keep for clarity)
# Actually, keep them as they document what was used before

with open(project_root / 'scripts' / 'evaluate_phase10_benchmark.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('File updated successfully')
