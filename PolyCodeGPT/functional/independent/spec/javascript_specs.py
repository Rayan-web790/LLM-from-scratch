"""Independent functional specifications for the Phase 12B benchmark (JavaScript)."""

from __future__ import annotations

DOC: tuple[str, ...] = (
    "Return the sum of all numbers in arr (0 for an empty array).",
    "Capitalize the first letter of each whitespace-separated word in s; leave the rest unchanged.",
    "Return a new array containing the unique values of arr in first-occurrence order.",
    "Flatten arr by exactly one level.",
    "Return true if n is even.",
    "Return s repeated n times (n >= 0); '' when n == 0.",
    "Return the maximum value in arr, or null if arr is empty.",
    "Count how many elements of arr strictly equal val (===).",
    "Return the reverse of string s.",
    "Return the arithmetic mean of arr, or 0 if arr is empty.",
)

STUB: tuple[str, ...] = (
    "function totalOfArray(arr)",
    "function titleCaseWords(s)",
    "function distinctValues(arr)",
    "function flattenOnce(arr)",
    "function isDivisibleByTwo(n)",
    "function repeatText(s, n)",
    "function maxOfArray(arr)",
    "function countOccurrences(arr, val)",
    "function flipString(s)",
    "function computeMean(arr)",
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


JS_SPECS: list[dict] = [
    T("totalOfArray", "Return the sum of all numbers in arr; 0 for an empty array.",
      [c("basic", [[1, 2, 3, 4]], 10), c("empty", [[]], 0, "boundary"),
       c("negatives", [[-1, 2, -3]], -2), c("single", [[7]], 7, "boundary")]),
    T("titleCaseWords", "Capitalize the first character of each whitespace-separated word.",
      [c("basic", ["hello world"], "Hello World"),
       c("single", ["hi"], "Hi", "boundary"),
       c("empty", [""], "", "boundary"),
       c("already", ["Hello World"], "Hello World")]),
    T("distinctValues", "Return a new array of the unique values of arr in first-occurrence order.",
      [c("basic", [[1, 2, 2, 3, 1]], [1, 2, 3]),
       c("none dup", [[1, 2, 3]], [1, 2, 3]),
       c("empty", [[]], [], "boundary"),
       c("all same", [[4, 4, 4]], [4], "boundary")]),
    T("flattenOnce", "Flatten arr by exactly one level.",
      [c("basic", [[[1, 2], [3, 4]]], [1, 2, 3, 4]),
       c("mixed", [[1, [2, 3], 4]], [1, 2, 3, 4]),
       c("empty", [[]], [], "boundary"),
       c("no nested", [[1, 2, 3]], [1, 2, 3])]),
    T("isDivisibleByTwo", "Return true if n is even.",
      [c("even", [4], True), c("odd", [7], False),
       c("zero", [0], True, "boundary"), c("negative", [-2], True, "boundary")]),
    T("repeatText", "Return s repeated n times; '' when n == 0.",
      [c("basic", ["ab", 3], "ababab"),
       c("once", ["x", 1], "x", "boundary"),
       c("zero", ["ab", 0], "", "boundary"),
       c("empty s", ["", 5], "")]),
    T("maxOfArray", "Return the maximum value in arr, or null if arr is empty.",
      [c("basic", [[1, 5, 3]], 5), c("single", [[9]], 9, "boundary"),
       c("empty", [[]], None, "boundary"),
       c("negatives", [[-3, -1, -7]], -1)]),
    T("countOccurrences", "Count elements of arr strictly equal to val (===).",
      [c("basic", [[1, 2, 2, 3], 2], 2),
       c("none", [[1, 2, 3], 5], 0, "boundary"),
       c("empty", [[], 1], 0, "boundary"),
       c("type strict", [[1, "1", 1], 1], 2)]),
    T("flipString", "Return the reverse of string s.",
      [c("basic", ["hello"], "olleh"),
       c("empty", [""], "", "boundary"),
       c("single", ["a"], "a", "boundary"),
       c("palindrome", ["aba"], "aba")]),
    T("computeMean", "Return the arithmetic mean of arr, or 0 if arr is empty.",
      [c("basic", [[2, 4, 6]], 4.0),
       c("single", [[5]], 5.0, "boundary"),
       c("empty", [[]], 0, "boundary"),
       c("decimals", [[1, 2], ], None) if False else c("decimals", [[1, 2]], 1.5)]),
]