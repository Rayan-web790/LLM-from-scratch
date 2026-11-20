"""Find the actual pattern that causes the extraction worker to hang."""
import ast
import io
import sys
import time
import tokenize

print("Python:", sys.version)

# Pattern 1: Long repeated string concatenation (CPython < 3.12 quadratic JoinedStr)
s1 = '"a" + ' * 5000 + '"z"'
print("Testing repeated string concat 5000...")
t0 = time.time()
try:
    ast.parse(s1)
    print(f"  ok in {time.time()-t0:.3f}s")
except Exception as e:
    print(f"  {type(e).__name__} in {time.time()-t0:.3f}s")

# Pattern 2: Long chained comparison
s2 = " < ".join(["x"] * 2000)
print("Testing chained comparison 2000...")
t0 = time.time()
try:
    ast.parse(s2)
    print(f"  ok in {time.time()-t0:.3f}s")
except Exception as e:
    print(f"  {type(e).__name__} in {time.time()-t0:.3f}s")

# Pattern 3: Huge file with many top-level functions (deep walk)
lines = ["def f" + str(i) + "():\n    return " + str(i) + "\n" for i in range(10000)]
s3 = "\n".join(lines)
print("Testing 10000 top-level functions...")
t0 = time.time()
try:
    ast.parse(s3)
    print(f"  ok in {time.time()-t0:.3f}s")
except Exception as e:
    print(f"  {type(e).__name__} in {time.time()-t0:.3f}s")

# Pattern 4: tokenize on huge multiline string (500KB)
big = "a" * 500_000
s4 = "x = " + repr(big)
print("Testing tokenize 500KB string...")
t0 = time.time()
try:
    list(tokenize.generate_tokens(io.StringIO(s4).readline))
    print(f"  ok in {time.time()-t0:.3f}s")
except Exception as e:
    print(f"  {type(e).__name__} in {time.time()-t0:.3f}s")

# Pattern 5: deeply nested function calls (not brackets, but calls)
s5 = "f(" * 1000 + "1" + ")" * 1000
print("Testing deeply nested calls 1000...")
t0 = time.time()
try:
    ast.parse(s5)
    print(f"  ok in {time.time()-t0:.3f}s")
except Exception as e:
    print(f"  {type(e).__name__} in {time.time()-t0:.3f}s")

# Pattern 6: Very large file (5MB of code)
lines6 = ["x" + str(i) + " = " + str(i) + " * " + str(i) + " + 1" for i in range(100_000)]
s6 = "\n".join(lines6)
print(f"Testing 100K assignments ({len(s6)//1024}KB)...")
t0 = time.time()
try:
    ast.parse(s6)
    print(f"  ok in {time.time()-t0:.3f}s")
except Exception as e:
    print(f"  {type(e).__name__} in {time.time()-t0:.3f}s")
