"""Independent functional specifications for the Phase 12B benchmark (Python).

These tasks are deliberately disjoint from ``scripts.phase10_tasks`` and from the
Phase 11 task list. Docstrings and stubs are defined here (not imported), so
this benchmark cannot drift against the development benchmark.
"""

from __future__ import annotations

DOC: tuple[str, ...] = (
    "Return the n-th triangular number T(n) = n*(n+1)//2 for integer n >= 0; T(0) == 0.",
    "Return True iff a and b are anagrams (same multiset of characters, case-insensitive); ignore nothing.",
    "Return the running (prefix) sums of nums as a new list: out[i] = sum(nums[:i+1]).",
    "Return a new list with duplicates removed, preserving first-occurrence order.",
    "Return the longest common prefix of the strings in strs; '' if strs is empty or has no common prefix.",
    "Clamp x to the inclusive range [lo, hi]: return max(lo, min(hi, x)).",
    "Count whitespace-separated words in s (case-sensitive) and return a dict mapping word -> count.",
    "Return a new list with nums rotated right by k positions (k may exceed len(nums)).",
    "Return True iff n is a power of two (n > 0).",
    "Return the second largest distinct value in nums, or None if fewer than two distinct values exist.",
)

STUB: tuple[str, ...] = (
    "def triangularSum(n):",
    "def areAnagrams(a, b):",
    "def running_sum(nums):",
    "def dropDuplicateValues(nums):",
    "def longest_common_prefix(strs):",
    "def boundValue(x, lo, hi):",
    "def tallyWords(s):",
    "def rotate_list(nums, k):",
    "def is_power_of_two(n):",
    "def second_largest(nums):",
)


def c(name, args, expected, kind="normal", convention=False):
    case = {"name": name, "kind": kind, "args": args, "expected": expected}
    if convention:
        case["convention"] = True
    return case


def e(name, args, error):
    return {"name": name, "kind": "error", "args": args, "expect_error": error}


def T(entry, behavior, cases, resolution=None, convention=None):
    return {"entry": entry, "behavior": behavior, "resolution": resolution,
            "convention_note": convention, "cases": cases}


PY_SPECS: list[dict] = [
    T("triangularSum",
      "Return n*(n+1)//2 for integer n >= 0; T(0) == 0.",
      [c("n=0", [0], 0, "boundary"), c("n=1", [1], 1, "boundary"),
       c("n=5", [5], 15), c("n=10", [10], 55)]),
    T("areAnagrams",
      "Return True iff a and b contain the same characters with the same multiplicities, "
      "compared case-insensitively.",
      [c("anagram", ["listen", "silent"], True), c("not anagram", ["hello", "world"], False),
       c("case differs", ["Tea", "Eat"], True, "boundary"), c("empty", ["", ""], True, "boundary"),
       c("different lengths", ["ab", "abc"], False)]),
    T("running_sum",
      "Return a new list where out[i] = sum(nums[:i+1]). Empty input gives [].",
      [c("basic", [[1, 2, 3, 4]], [1, 3, 6, 10]),
       c("single", [[5]], [5], "boundary"),
       c("empty", [[]], [], "boundary"),
       c("negatives", [[-1, 2, -3]], [-1, 1, -2])]),
    T("dropDuplicateValues",
      "Return a new list with duplicate values removed, keeping the first occurrence of each.",
      [c("basic", [[1, 2, 2, 3, 1]], [1, 2, 3]),
       c("none dup", [[1, 2, 3]], [1, 2, 3]),
       c("empty", [[]], [], "boundary"),
       c("all same", [[4, 4, 4]], [4], "boundary")]),
    T("longest_common_prefix",
      "Return the longest string that is a prefix of every string in strs. Return '' if strs is empty.",
      [c("common", [["flower", "flow", "flight"]], "fl"),
       c("none", [["dog", "racecar", "car"]], ""),
       c("single", [["alone"]], "alone", "boundary"),
       c("empty list", [[]], "", "boundary"),
       c("identical", [["abc", "abc"]], "abc")]),
    T("boundValue",
      "Return max(lo, min(hi, x)).",
      [c("within", [5, 0, 10], 5), c("below", [-3, 0, 10], 0, "boundary"),
       c("above", [15, 0, 10], 10, "boundary"), c("equal lo", [0, 0, 10], 0),
       c("equal hi", [10, 0, 10], 10)]),
    T("tallyWords",
      "Split s on whitespace and return a dict mapping each word to its count (case-sensitive).",
      [c("basic", ["a b a c"], {"a": 2, "b": 1, "c": 1}),
       c("empty", [""], {}, "boundary"),
       c("single", ["hello"], {"hello": 1}, "boundary"),
       c("repeats", ["x x x"], {"x": 3})]),
    T("rotate_list",
      "Return a new list with nums rotated right by k positions. k may exceed len(nums).",
      [c("basic", [[1, 2, 3, 4, 5], 2], [4, 5, 1, 2, 3]),
       c("rotate one", [[1, 2, 3], 1], [3, 1, 2]),
       c("k=0", [[1, 2, 3], 0], [1, 2, 3], "boundary"),
       c("k>len", [[1, 2, 3], 4], [3, 1, 2], "boundary"),
       c("empty", [[], 3], [], "boundary")]),
    T("is_power_of_two",
      "Return True iff n is a power of two (n > 0).",
      [c("1", [1], True, "boundary"), c("2", [2], True, "boundary"),
       c("4", [4], True), c("3", [3], False), c("0", [0], False),
       c("8", [8], True), c("6", [6], False)]),
    T("second_largest",
      "Return the second largest distinct value in nums, or None if fewer than two distinct values.",
      [c("basic", [[1, 2, 3, 4]], 3), c("dup max", [[1, 2, 2, 3]], 2),
       c("two", [[5, 1]], 1, "boundary"), c("one distinct", [[7, 7]], None, "boundary"),
       c("empty", [[]], None, "boundary")]),
]
