"""Phase 11 functional specifications for the 50 Python Phase 10 tasks.

Specs are positional: task *i* here is ``scripts.phase10_tasks.PYTHON_TASKS[i]``,
so docstrings/stubs are imported rather than copied and cannot drift.

Each case declares ``kind`` (``normal`` / ``boundary`` / ``error``) and either an
``expected`` value or an ``expect_error`` class name. Cases marked
``convention=True`` encode a return convention the docstring does not state; the
evaluator reports success both with and without them so the ambiguity is visible
in the results instead of silently deciding the outcome for the model.
"""

from __future__ import annotations

from scripts.phase10_tasks import PYTHON_TASKS

DOC, STUB = zip(*PYTHON_TASKS)


def c(name: str, args: list, expected, kind: str = "normal", convention: bool = False) -> dict:
    case = {"name": name, "kind": kind, "args": args, "expected": expected}
    if convention:
        case["convention"] = True
    return case


def e(name: str, args: list, error: str) -> dict:
    return {"name": name, "kind": "error", "args": args, "expect_error": error}


def T(entry: str, behavior: str, cases: list[dict], resolution: str | None = None,
      convention: str | None = None) -> dict:
    return {
        "entry": entry,
        "behavior": behavior,
        "resolution": resolution,
        "convention_note": convention,
        "cases": cases,
    }


PY_SPECS: list[dict] = [
    T("factorial",
      "Return n! as an int for integer n >= 0; 0! == 1. Recursive implementation expected.",
      [c("n=5", [5], 120), c("n=7", [7], 5040), c("n=0", [0], 1, "boundary"),
       c("n=1", [1], 1, "boundary")],
      resolution="Docstring is silent on negative n; negative input is not tested, so no "
                 "error contract is imposed."),
    T("fibonacci",
      "Return the n-th Fibonacci number with 0-based indexing: F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2).",
      [c("n=0", [0], 0, "boundary"), c("n=1", [1], 1, "boundary"), c("n=10", [10], 55),
       c("n=20", [20], 6765)],
      resolution="Indexing is not stated. Resolved to 0-based, the standard convention for "
                 "'nth Fibonacci number' when F(0)=0 is included."),
    T("is_palindrome",
      "Return True iff s reads the same forwards and backwards; comparison is case-sensitive "
      "and whitespace-sensitive.",
      [c("racecar", ["racecar"], True), c("hello", ["hello"], False, "normal"),
       c("empty", [""], True, "boundary"), c("abba", ["abba"], True),
       c("single char", ["a"], True, "boundary")],
      resolution="Case folding is not specified; resolved to exact character comparison "
                 "(no lowercase normalisation), so 'Racecar' is not a palindrome."),
    T("merge_sorted",
      "Return a new ascending list containing every element of a and b; inputs are already "
      "sorted ascending; duplicates are preserved.",
      [c("interleaved", [[1, 3, 5], [2, 4, 6]], [1, 2, 3, 4, 5, 6]),
       c("empty + list", [[], [1, 2]], [1, 2], "boundary"),
       c("both empty", [[], []], [], "boundary"),
       c("duplicates", [[1, 1], [1]], [1, 1, 1])]),
    T("binary_search",
      "Search the ascending list arr for target; return its index or -1 when absent.",
      [c("found middle", [[1, 3, 5, 7, 9], 5], 2), c("found first", [[1, 3, 5, 7, 9], 1], 0),
       c("found last", [[1, 3, 5, 7, 9], 9], 4), c("absent", [[1, 3, 5, 7, 9], 4], -1, "normal", True),
       c("empty", [[], 1], -1, "boundary", True)],
      resolution="Return value for a miss is not stated. Resolved to -1 (the common "
                 "convention); miss cases are flagged as conventions because None is an "
                 "equally defensible reading.",
      convention="A candidate returning None on a miss fails only the convention cases; the "
                 "sensitivity metric in the report excludes them."),
    T("nth_prime",
      "Return the n-th prime with 1-based indexing: nth_prime(1) == 2.",
      [c("n=1", [1], 2, "boundary"), c("n=5", [5], 11), c("n=10", [10], 29),
       c("n=20", [20], 71)],
      resolution="Whether the count is 0- or 1-based is unstated; resolved to 1-based, so the "
                 "first prime is index 1."),
    T("reverse_string",
      "Return s with its characters in reverse order.",
      [c("hello", ["hello"], "olleh"), c("empty", [""], "", "boundary"),
       c("single", ["a"], "a", "boundary"), c("two", ["ab"], "ba")]),
    T("is_anagram",
      "Return True iff s1 and s2 contain the same characters with the same multiplicities; "
      "comparison is case-sensitive and ignores nothing else.",
      [c("listen/silent", ["listen", "silent"], True), c("hello/world", ["hello", "world"], False),
       c("abc/cba", ["abc", "cba"], True), c("aab/abb", ["aab", "abb"], False, "boundary")],
      resolution="Case sensitivity is unstated; resolved to case-sensitive exact matching."),
    T("gcd",
      "Return the greatest common divisor of integers a and b; gcd(a, 0) == a.",
      [c("12,18", [12, 18], 6), c("7,13", [7, 13], 1), c("0,5", [0, 5], 5, "boundary"),
       c("5,0", [5, 0], 5, "boundary"), c("48,180", [48, 180], 12)],
      resolution="Sign handling is unstated; only non-negative inputs are tested, so the "
                 "Python-specific sign of gcd is never exercised."),
    T("format_date",
      "Convert a 'YYYY-MM-DD' string to 'DD/MM/YYYY'.",
      [c("2024-01-05", ["2024-01-05"], "05/01/2024"),
       c("1999-12-31", ["1999-12-31"], "31/12/1999"),
       c("2026-11-17", ["2026-11-17"], "17/11/2026"),
       c("0001-01-01", ["0001-01-01"], "01/01/0001", "boundary")]),
    T("is_prime",
      "Return True iff n is a prime number; every n < 2 (including 0, 1 and negatives) is False.",
      [c("2", [2], True, "boundary"), c("17", [17], True), c("1", [1], False, "boundary"),
       c("0", [0], False, "boundary"), c("9", [9], False), c("97", [97], True),
       c("-7", [-7], False, "boundary")],
      resolution="Behaviour for n < 2 is unstated; resolved to False (mathematical convention), "
                 "documented here so negative inputs are well defined."),
    T("triangular_number",
      "Return the n-th triangular number n*(n+1)/2 for n >= 0.",
      [c("n=1", [1], 1, "boundary"), c("n=5", [5], 15), c("n=10", [10], 55),
       c("n=0", [0], 0, "boundary")]),
    T("linear_search",
      "Scan arr in order and return the index of the first occurrence of target, or -1.",
      [c("last", [[4, 2, 9], 9], 2), c("first", [[4, 2, 9], 4], 0),
       c("absent", [[4, 2, 9], 7], -1, "normal", True),
       c("empty", [[], 1], -1, "boundary", True)],
      resolution="Miss value unstated; resolved to -1 and flagged as a convention case.",
      convention="A candidate returning None on a miss fails only the miss cases, which are "
                 "excluded from the sensitivity metric."),
    T("find_max",
      "Return the largest element of the non-empty list arr.",
      [c("1,5,3", [[1, 5, 3]], 5), c("negatives", [[-4, -2, -9]], -2),
       c("single", [[7]], 7, "boundary"), c("all equal", [[2, 2]], 2, "boundary")],
      resolution="Empty input is unstated and untested; no error contract is imposed."),
    T("bubble_sort",
      "Return a new list with arr sorted ascending using bubble sort; input is not modified.",
      [c("3,1,2", [[3, 1, 2]], [1, 2, 3]), c("empty", [[]], [], "boundary"),
       c("single", [[5]], [5], "boundary"), c("duplicates", [[2, 2, 1]], [1, 2, 2]),
       c("negatives", [[-1, -5, 3]], [-5, -1, 3])]),
    T("is_even",
      "Return True iff n is divisible by 2, for positive, zero and negative integers.",
      [c("4", [4], True), c("7", [7], False), c("0", [0], True, "boundary"),
       c("-2", [-2], True, "boundary"), c("-3", [-3], False, "boundary")]),
    T("sum_list",
      "Return the arithmetic sum of lst; the empty list sums to 0.",
      [c("1,2,3", [[1, 2, 3]], 6), c("empty", [[]], 0, "boundary"),
       c("-1,1", [[-1, 1]], 0), c("single", [[10]], 10, "boundary")]),
    T("factorial_iter",
      "Return n! for integer n >= 0 using an iterative loop; 0! == 1.",
      [c("n=5", [5], 120), c("n=0", [0], 1, "boundary"), c("n=1", [1], 1, "boundary"),
       c("n=6", [6], 720)]),
    T("my_len",
      "Return the number of characters in s without calling the built-in len().",
      [c("hello", ["hello"], 5), c("empty", [""], 0, "boundary"),
       c("single", ["a"], 1, "boundary"), c("spaces", ["a b"], 3)],
      resolution="The 'without len()' constraint is a style requirement that cannot be "
                 "verified by behaviour alone; it is not enforced by the evaluator."),
    T("is_leap_year",
      "Return True iff y is a Gregorian leap year: divisible by 4, except centuries, except "
      "multiples of 400.",
      [c("2024", [2024], True), c("1900", [1900], False, "boundary"),
       c("2000", [2000], True, "boundary"), c("2023", [2023], False),
       c("1600", [1600], True, "boundary")]),
    T("countdown",
      "Return a list of integers counting down from n to 1 inclusive; n <= 0 returns [].",
      [c("5", [5], [5, 4, 3, 2, 1]), c("1", [1], [1], "boundary"),
       c("0", [0], [], "boundary"), c("3", [3], [3, 2, 1])],
      resolution="Return shape is unstated (a printed countdown is equally plausible). "
                 "Resolved to returning a list; a printing-only implementation fails."),
    T("safe_factorial",
      "Return n! for integer n >= 0 and raise ValueError when n < 0.",
      [c("n=5", [5], 120), c("n=3", [3], 6), c("n=0", [0], 1, "boundary"),
       e("n=-1 raises ValueError", [-1], "ValueError"),
       e("n=-5 raises ValueError", [-5], "ValueError")],
      resolution="The docstring says 'with error checking' but names no type; resolved to "
                 "ValueError, the conventional Python choice, and stated in the specification."),
    T("caesar_cipher",
      "Shift every ASCII letter of s forward by n positions, wrapping within the same case "
      "(a-z and A-Z separately); non-letters are unchanged.",
      [c("abc+1", ["abc", 1], "bcd"), c("xyz+3", ["xyz", 3], "abc"),
       c("hello+0", ["hello", 0], "hello", "boundary"), c("abc+25", ["abc", 25], "zab"),
       c("mixed+1", ["Hello, World!", 1], "Ifmmp, Xpsme!", "normal", True)],
      resolution="Alphabet, case handling and non-letter handling are unstated. Resolved to "
                 "case-preserving ASCII letters with wrap and unchanged punctuation; the mixed "
                 "case case is flagged as a convention.",
      convention="Lowercase-only cases are mandatory; the mixed-case/punctuation case is "
                 "reported separately."),
    T("fib_memo",
      "Return the n-th Fibonacci number (0-based, F(0)=0, F(1)=1) using the memo dict for "
      "caching; calling with a single argument must work.",
      [c("n=0", [0], 0, "boundary"), c("n=1", [1], 1, "boundary"), c("n=10", [10], 55),
       c("n=25", [25], 75025)],
      resolution="Calling convention for memo is unstated; every case calls with a single "
                 "argument, so the default from the stub must work (a mutable default is fine)."),
    T("get_evens",
      "Return the even elements of lst in their original order.",
      [c("1,2,3,4", [[1, 2, 3, 4]], [2, 4]), c("empty", [[]], [], "boundary"),
       c("all even", [[2, 4]], [2, 4]), c("no even", [[1, 3]], [], "boundary")]),
    T("get_odds",
      "Return the odd elements of lst in their original order.",
      [c("1,2,3,4", [[1, 2, 3, 4]], [1, 3]), c("empty", [[]], [], "boundary"),
       c("no odd", [[2, 4]], [], "boundary"), c("all odd", [[1, 3]], [1, 3])]),
    T("filter_positives",
      "Return the elements of lst strictly greater than zero, preserving order.",
      [c("-2,0,3,5", [[-2, 0, 3, 5]], [3, 5]), c("empty", [[]], [], "boundary"),
       c("only zero", [[0]], [], "boundary"), c("single positive", [[1]], [1])],
      resolution="'Positive' is resolved to > 0, so 0 is excluded (the standard mathematical "
                 "meaning)."),
    T("string_lengths",
      "Map each string in lst to its integer character count.",
      [c("a,bb,ccc", [["a", "bb", "ccc"]], [1, 2, 3]), c("ab,c", [["ab", "c"]], [2, 1]),
       c("empty", [[]], [], "boundary"),
       c("with empty string", [["", "ab"]], [0, 2], "boundary")]),
    T("is_sorted",
      "Return True iff lst is in non-decreasing order; the empty list and singletons are sorted.",
      [c("ascending", [[1, 2, 3]], True), c("descending", [[3, 2, 1]], False),
       c("empty", [[]], True, "boundary"), c("single", [[5]], True, "boundary"),
       c("equal pair", [[1, 1, 2]], True, "boundary")],
      resolution="'Sorted' is resolved to non-decreasing (equal neighbours allowed)."),
    T("fib_recursive",
      "Return the n-th Fibonacci number (0-based) using recursion.",
      [c("n=0", [0], 0, "boundary"), c("n=1", [1], 1, "boundary"), c("n=7", [7], 13),
       c("n=15", [15], 610)]),
    T("remove_duplicates",
      "Return a list of the distinct elements of lst, keeping the order of first appearance.",
      [c("1,2,2,3,1", [[1, 2, 2, 3, 1]], [1, 2, 3]), c("empty", [[]], [], "boundary"),
       c("all same", [[1, 1, 1]], [1], "boundary"), c("already unique", [[3, 1, 2]], [3, 1, 2])],
      resolution="Output order for duplicates is unstated; resolved to first-occurrence order."),
    T("list_intersection",
      "Return the distinct elements that occur in both l1 and l2, in l1 order.",
      [c("1,2,3 & 2,3,4", [[1, 2, 3], [2, 3, 4]], [2, 3]),
       c("empty left", [[], [1]], [], "boundary"),
       c("disjoint", [[1, 2], [3, 4]], [], "normal"),
       c("duplicate in l1", [[1, 2, 2], [2]], [2], "boundary", True)],
      resolution="Set semantics vs multiset semantics and output order are unstated. Resolved "
                 "to distinct elements in l1 order; the duplicate case is flagged as a "
                 "convention because a multiset implementation would return [2, 2].",
      convention="A multiset implementation fails only the duplicate case, which is excluded "
                 "from the sensitivity metric."),
    T("list_union",
      "Return the distinct elements of l1 followed by the distinct elements of l2 not already "
      "present.",
      [c("disjoint", [[1, 2], [3, 4]], [1, 2, 3, 4]),
       c("overlap", [[1, 2], [2, 3]], [1, 2, 3]),
       c("empty left", [[], [1]], [1], "boundary"),
       c("both empty", [[], []], [], "boundary")],
      resolution="Order of the union is unstated; resolved to l1 order then new l2 elements."),
    T("flatten",
      "Return a flat list containing every non-list element of the arbitrarily nested lst, in "
      "depth-first order.",
      [c("nested", [[1, [2, [3, 4]], 5]], [1, 2, 3, 4, 5]),
       c("empty inner", [[[], [1]]], [1], "boundary"),
       c("already flat", [[1, 2]], [1, 2]),
       c("deep", [[[[1]], 2]], [1, 2], "boundary")],
      resolution="Nesting depth is unstated; resolved to full recursive flattening, so "
                 "single-level implementations fail the deep case."),
    T("binary_search_simple",
      "Return True iff target occurs in the ascending list arr, else False.",
      [c("present", [[1, 3, 5], 3], True), c("absent", [[1, 3, 5], 4], False),
       c("empty", [[], 1], False, "boundary")],
      resolution="'Checks if a value is in a sorted list' is resolved to a boolean membership "
                 "result (unlike binary_search, which returns an index)."),
    T("running_total",
      "Return the list of prefix sums of lst; the empty list yields [].",
      [c("1,2,3", [[1, 2, 3]], [1, 3, 6]), c("empty", [[]], [], "boundary"),
       c("single", [[5]], [5], "boundary"), c("-1,1", [[-1, 1]], [-1, 0])]),
    T("selection_sort",
      "Return a new list with arr sorted ascending using selection sort.",
      [c("64,25,12", [[64, 25, 12]], [12, 25, 64]), c("empty", [[]], [], "boundary"),
       c("single", [[3]], [3], "boundary"), c("duplicates", [[2, 2, 1]], [1, 2, 2])]),
    T("caesar_wrap",
      "Shift ASCII letters of s by n, where n may exceed 25 and wraps modulo 26; case is "
      "preserved and non-letters are unchanged.",
      [c("xyz+3", ["xyz", 3], "abc"), c("hello+2", ["hello", 2], "jgnnq"),
       c("abc+29", ["abc", 29], "def", "boundary"),
       c("hello+0", ["hello", 0], "hello", "boundary"), c("abc+52", ["abc", 52], "abc", "boundary")],
      resolution="Wrap-around beyond 25 is the stated point of this task; resolved to shift "
                 "modulo 26. Negative shifts are unspecified and untested."),
    T("same_word",
      "Return True iff w1 and w2 are equal ignoring ASCII letter case.",
      [c("Hello/hello", ["Hello", "hello"], True), c("abc/abd", ["abc", "abd"], False),
       c("A/a", ["A", "a"], True, "boundary"), c("same", ["test", "test"], True)]),
    T("my_abs",
      "Return the absolute value of x without calling the built-in abs().",
      [c("-5", [-5], 5), c("5", [5], 5), c("0", [0], 0, "boundary"),
       c("-3.5", [-3.5], 3.5, "boundary"), c("2.5", [2.5], 2.5)],
      resolution="The 'without abs()' constraint is stylistic and not enforced behaviourally."),
    T("power",
      "Return base raised to the non-negative integer exponent exp, computed by repeated "
      "multiplication; exp == 0 yields 1 for any base.",
      [c("2^10", [2, 10], 1024), c("5^0", [5, 0], 1, "boundary"), c("3^3", [3, 3], 27),
       c("1^100", [1, 100], 1, "boundary")],
      resolution="Negative exponents are unspecified and untested."),
    T("is_digits",
      "Return True iff s is non-empty and every character is an ASCII digit 0-9.",
      [c("123", ["123"], True), c("12a", ["12a"], False), c("empty", [""], False, "boundary"),
       c("leading zero", ["007"], True, "boundary")],
      resolution="The empty string is unstated; resolved to False, matching str.isdigit('')."),
    T("min_of_three",
      "Return the smallest of a, b and c.",
      [c("1,2,3", [1, 2, 3], 1), c("5,2,9", [5, 2, 9], 2), c("all negative", [-1, -5, -3], -5),
       c("all equal", [2, 2, 2], 2, "boundary")]),
    T("c_to_f",
      "Convert a Celsius temperature to Fahrenheit: F = C * 9/5 + 32.",
      [c("0C", [0], 32, "boundary"), c("100C", [100], 212), c("-40C", [-40], -40, "boundary"),
       c("10C", [10], 50)]),
    T("is_positive",
      "Return True iff n is strictly greater than zero.",
      [c("5", [5], True), c("-1", [-1], False), c("0", [0], False, "boundary")]),
    T("nth_even",
      "Return the n-th positive even number with 1-based indexing: nth_even(1) == 2.",
      [c("n=1", [1], 2, "boundary"), c("n=5", [5], 10), c("n=10", [10], 20)],
      resolution="Indexing unstated; resolved to 1-based so the first even number is index 1."),
    T("my_mod",
      "Return a modulo b (the remainder of integer division) without calling the built-in % "
      "operator.",
      [c("10%3", [10, 3], 1), c("9%3", [9, 3], 0, "boundary"), c("7%4", [7, 4], 3)],
      resolution="Only non-negative operands are tested because Python and JavaScript disagree "
                 "on the sign of a negative remainder, and the docstring does not specify."),
    T("is_blank",
      "Return True iff s is empty or contains only whitespace.",
      [c("empty", [""], True, "boundary"), c("spaces", ["   "], True, "boundary"),
       c("padded text", [" a "], False), c("text", ["a"], False)],
      resolution="'Blank' is unstated; resolved to empty-or-whitespace-only."),
    T("square",
      "Return the square of x.",
      [c("4", [4], 16), c("0", [0], 0, "boundary"), c("-3", [-3], 9),
       c("2.5", [2.5], 6.25)]),
    T("fib_iter",
      "Return the n-th Fibonacci number (0-based, F(0)=0, F(1)=1) using an iterative loop.",
      [c("n=0", [0], 0, "boundary"), c("n=1", [1], 1, "boundary"), c("n=10", [10], 55),
       c("n=30", [30], 832040)]),
]

assert len(PY_SPECS) == len(PYTHON_TASKS) == 50, (len(PY_SPECS), len(PYTHON_TASKS))
