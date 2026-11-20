with open('reports/phase9_integrity_repetition_audit.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix 1: e314c7e6979c6481 language label - change 'python' to '?' and add flag note
old = '| `e314c7e6979c6481` | 0.6667* | 0.0769 | 0.0000 | 53 | python | `def onMessage(e) { ... }` |'
new = '| `e314c7e6979c6481` | 0.6667* | 0.0769 | 0.0000 | 53 | ? | `def onMessage(e) { ... }` (language flagged for manual classification)|'
text = text.replace(old, new, 1)

# Fix 2: Remove section 3.3 "Metrics for Language-Matched Pairs (41,403 pairs)" table
# and the preceding caveat about 12.0x length disparity being the authoritative figure.
# We'll keep the 12.0x in context but clarify it's from the pair analysis, not the 100-record sample.
# Actually, per requirements: consolidate to one authoritative table (section 4).
# Remove the 3.3 table and its surrounding caveat text.

# Let me first check what to remove
print("File length before:", len(text))

# For now, just verify the e314 fix worked
if '| `e314c7e6979c6481` | 0.6667* | 0.0769 | 0.0000 | 53 | ? |' in text:
    print('Fix 1 applied: e314c7e6979c6481 language label changed')
else:
    print('Fix 1: Could not find the pattern to replace')

with open('reports/phase9_integrity_repetition_audit.md', 'w', encoding='utf-8') as f:
    f.write(text)
print('Fix applied')