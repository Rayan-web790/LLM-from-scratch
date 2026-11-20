# Phase 11 — Functional Benchmark Specification

100 tasks (50 Python, 50 JavaScript) taken from the frozen Phase 10 inventory. Each task defines an entry point, required behaviour, normal and boundary cases, and error behaviour where the requirements justify one.

Specs are positional against `scripts/phase10_tasks.py` — the docstrings the model saw in Phase 10 are imported, not re-typed, so the benchmark cannot silently drift.

## Case counts

| language | tasks | cases | normal | boundary | error | convention | documented ambiguities |
|---|---|---|---|---|---|---|---|
| python | 50 | 210 | 116 | 92 | 2 | 6 | 29 |
| javascript | 50 | 210 | 116 | 92 | 2 | 6 | 28 |

A task is functionally successful only when **every** case passes. Cases marked *convention* encode a return convention the docstring does not state; they are also reported separately as a sensitivity check so the ambiguity is visible in the results.

## Ambiguity policy

Ambiguities are resolved once, up front, and recorded per task in `resolution`. The resolutions used are:

1. **Miss value for searches** (`binary_search`, `linear_search`, JS `binarySearch`, `linearSearch`) → `-1`, flagged as convention cases.
2. **Indexing** of ordinal tasks (`fibonacci`, `nth_prime`, `nth_even`) → the n-th item is 1-based except Fibonacci, which is 0-based with F(0)=0.
3. **Case sensitivity** of text predicates → exact comparison unless the docstring says otherwise (`same_word`/`sameWord` are explicitly case-insensitive).
4. **Empty-input semantics** → documented per task; untested where the docstring gives no basis (empty `find_max`, negative `factorial`).
5. **Return shape** for `countdown` → a list/array, not printed output.
6. **Error contracts** → `ValueError` for Python `safe_factorial`; any `Error` instance for JS `safeFactorial`.
7. **Negative operands** for `my_mod`/`myMod` → untested, because Python and JavaScript define remainder signs differently and the docstring specifies neither.

Style-only constraints that cannot be observed behaviourally (`without len()`, `without abs()`, `without %`, `without Math.abs`, `without s.length`) are recorded but not enforced.

## Python tasks

### py_01 — `def factorial(n):`

- **Docstring:** Compute the factorial of n recursively.
- **Behaviour:** Return n! as an int for integer n >= 0; 0! == 1. Recursive implementation expected.
- **Resolution:** Docstring is silent on negative n; negative input is not tested, so no error contract is imposed.

| case | kind | args | expected |
|---|---|---|---|
| n=5 | normal | `[5]` | `120` |
| n=7 | normal | `[7]` | `5040` |
| n=0 | boundary | `[0]` | `1` |
| n=1 | boundary | `[1]` | `1` |

### py_02 — `def fibonacci(n):`

- **Docstring:** Return the nth Fibonacci number.
- **Behaviour:** Return the n-th Fibonacci number with 0-based indexing: F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2).
- **Resolution:** Indexing is not stated. Resolved to 0-based, the standard convention for 'nth Fibonacci number' when F(0)=0 is included.

| case | kind | args | expected |
|---|---|---|---|
| n=0 | boundary | `[0]` | `0` |
| n=1 | boundary | `[1]` | `1` |
| n=10 | normal | `[10]` | `55` |
| n=20 | normal | `[20]` | `6765` |

### py_03 — `def is_palindrome(s):`

- **Docstring:** Check whether a string is a palindrome.
- **Behaviour:** Return True iff s reads the same forwards and backwards; comparison is case-sensitive and whitespace-sensitive.
- **Resolution:** Case folding is not specified; resolved to exact character comparison (no lowercase normalisation), so 'Racecar' is not a palindrome.

| case | kind | args | expected |
|---|---|---|---|
| racecar | normal | `["racecar"]` | `true` |
| hello | normal | `["hello"]` | `false` |
| empty | boundary | `[""]` | `true` |
| abba | normal | `["abba"]` | `true` |
| single char | boundary | `["a"]` | `true` |

### py_04 — `def merge_sorted(a, b):`

- **Docstring:** Merge two sorted lists into one sorted list.
- **Behaviour:** Return a new ascending list containing every element of a and b; inputs are already sorted ascending; duplicates are preserved.

| case | kind | args | expected |
|---|---|---|---|
| interleaved | normal | `[[1, 3, 5], [2, 4, 6]]` | `[1, 2, 3, 4, 5, 6]` |
| empty + list | boundary | `[[], [1, 2]]` | `[1, 2]` |
| both empty | boundary | `[[], []]` | `[]` |
| duplicates | normal | `[[1, 1], [1]]` | `[1, 1, 1]` |

### py_05 — `def binary_search(arr, target):`

- **Docstring:** Binary search on a sorted list.
- **Behaviour:** Search the ascending list arr for target; return its index or -1 when absent.
- **Resolution:** Return value for a miss is not stated. Resolved to -1 (the common convention); miss cases are flagged as conventions because None is an equally defensible reading.
- **Convention:** A candidate returning None on a miss fails only the convention cases; the sensitivity metric in the report excludes them.

| case | kind | args | expected |
|---|---|---|---|
| found middle | normal | `[[1, 3, 5, 7, 9], 5]` | `2` |
| found first | normal | `[[1, 3, 5, 7, 9], 1]` | `0` |
| found last | normal | `[[1, 3, 5, 7, 9], 9]` | `4` |
| absent | normal* | `[[1, 3, 5, 7, 9], 4]` | `-1` |
| empty | boundary* | `[[], 1]` | `-1` |

### py_06 — `def nth_prime(n):`

- **Docstring:** Return the nth prime number.
- **Behaviour:** Return the n-th prime with 1-based indexing: nth_prime(1) == 2.
- **Resolution:** Whether the count is 0- or 1-based is unstated; resolved to 1-based, so the first prime is index 1.

| case | kind | args | expected |
|---|---|---|---|
| n=1 | boundary | `[1]` | `2` |
| n=5 | normal | `[5]` | `11` |
| n=10 | normal | `[10]` | `29` |
| n=20 | normal | `[20]` | `71` |

### py_07 — `def reverse_string(s):`

- **Docstring:** Reverse a string.
- **Behaviour:** Return s with its characters in reverse order.

| case | kind | args | expected |
|---|---|---|---|
| hello | normal | `["hello"]` | `"olleh"` |
| empty | boundary | `[""]` | `""` |
| single | boundary | `["a"]` | `"a"` |
| two | normal | `["ab"]` | `"ba"` |

### py_08 — `def is_anagram(s1, s2):`

- **Docstring:** Check if two strings are anagrams.
- **Behaviour:** Return True iff s1 and s2 contain the same characters with the same multiplicities; comparison is case-sensitive and ignores nothing else.
- **Resolution:** Case sensitivity is unstated; resolved to case-sensitive exact matching.

| case | kind | args | expected |
|---|---|---|---|
| listen/silent | normal | `["listen", "silent"]` | `true` |
| hello/world | normal | `["hello", "world"]` | `false` |
| abc/cba | normal | `["abc", "cba"]` | `true` |
| aab/abb | boundary | `["aab", "abb"]` | `false` |

### py_09 — `def gcd(a, b):`

- **Docstring:** Compute the greatest common divisor.
- **Behaviour:** Return the greatest common divisor of integers a and b; gcd(a, 0) == a.
- **Resolution:** Sign handling is unstated; only non-negative inputs are tested, so the Python-specific sign of gcd is never exercised.

| case | kind | args | expected |
|---|---|---|---|
| 12,18 | normal | `[12, 18]` | `6` |
| 7,13 | normal | `[7, 13]` | `1` |
| 0,5 | boundary | `[0, 5]` | `5` |
| 5,0 | boundary | `[5, 0]` | `5` |
| 48,180 | normal | `[48, 180]` | `12` |

### py_10 — `def format_date(date_str):`

- **Docstring:** Format a date from YYYY-MM-DD to DD/MM/YYYY.
- **Behaviour:** Convert a 'YYYY-MM-DD' string to 'DD/MM/YYYY'.

| case | kind | args | expected |
|---|---|---|---|
| 2024-01-05 | normal | `["2024-01-05"]` | `"05/01/2024"` |
| 1999-12-31 | normal | `["1999-12-31"]` | `"31/12/1999"` |
| 2026-11-17 | normal | `["2026-11-17"]` | `"17/11/2026"` |
| 0001-01-01 | boundary | `["0001-01-01"]` | `"01/01/0001"` |

### py_11 — `def is_prime(n):`

- **Docstring:** Check if a number is prime.
- **Behaviour:** Return True iff n is a prime number; every n < 2 (including 0, 1 and negatives) is False.
- **Resolution:** Behaviour for n < 2 is unstated; resolved to False (mathematical convention), documented here so negative inputs are well defined.

| case | kind | args | expected |
|---|---|---|---|
| 2 | boundary | `[2]` | `true` |
| 17 | normal | `[17]` | `true` |
| 1 | boundary | `[1]` | `false` |
| 0 | boundary | `[0]` | `false` |
| 9 | normal | `[9]` | `false` |
| 97 | normal | `[97]` | `true` |
| -7 | boundary | `[-7]` | `false` |

### py_12 — `def triangular_number(n):`

- **Docstring:** Calculate the nth triangular number.
- **Behaviour:** Return the n-th triangular number n*(n+1)/2 for n >= 0.

| case | kind | args | expected |
|---|---|---|---|
| n=1 | boundary | `[1]` | `1` |
| n=5 | normal | `[5]` | `15` |
| n=10 | normal | `[10]` | `55` |
| n=0 | boundary | `[0]` | `0` |

### py_13 — `def linear_search(arr, target):`

- **Docstring:** Implement a linear search.
- **Behaviour:** Scan arr in order and return the index of the first occurrence of target, or -1.
- **Resolution:** Miss value unstated; resolved to -1 and flagged as a convention case.
- **Convention:** A candidate returning None on a miss fails only the miss cases, which are excluded from the sensitivity metric.

| case | kind | args | expected |
|---|---|---|---|
| last | normal | `[[4, 2, 9], 9]` | `2` |
| first | normal | `[[4, 2, 9], 4]` | `0` |
| absent | normal* | `[[4, 2, 9], 7]` | `-1` |
| empty | boundary* | `[[], 1]` | `-1` |

### py_14 — `def find_max(arr):`

- **Docstring:** Find the maximum element in a list.
- **Behaviour:** Return the largest element of the non-empty list arr.
- **Resolution:** Empty input is unstated and untested; no error contract is imposed.

| case | kind | args | expected |
|---|---|---|---|
| 1,5,3 | normal | `[[1, 5, 3]]` | `5` |
| negatives | normal | `[[-4, -2, -9]]` | `-2` |
| single | boundary | `[[7]]` | `7` |
| all equal | boundary | `[[2, 2]]` | `2` |

### py_15 — `def bubble_sort(arr):`

- **Docstring:** Implement a simple sort (bubble sort).
- **Behaviour:** Return a new list with arr sorted ascending using bubble sort; input is not modified.

| case | kind | args | expected |
|---|---|---|---|
| 3,1,2 | normal | `[[3, 1, 2]]` | `[1, 2, 3]` |
| empty | boundary | `[[]]` | `[]` |
| single | boundary | `[[5]]` | `[5]` |
| duplicates | normal | `[[2, 2, 1]]` | `[1, 2, 2]` |
| negatives | normal | `[[-1, -5, 3]]` | `[-5, -1, 3]` |

### py_16 — `def is_even(n):`

- **Docstring:** Check if a number is even.
- **Behaviour:** Return True iff n is divisible by 2, for positive, zero and negative integers.

| case | kind | args | expected |
|---|---|---|---|
| 4 | normal | `[4]` | `true` |
| 7 | normal | `[7]` | `false` |
| 0 | boundary | `[0]` | `true` |
| -2 | boundary | `[-2]` | `true` |
| -3 | boundary | `[-3]` | `false` |

### py_17 — `def sum_list(lst):`

- **Docstring:** Compute the sum of list elements.
- **Behaviour:** Return the arithmetic sum of lst; the empty list sums to 0.

| case | kind | args | expected |
|---|---|---|---|
| 1,2,3 | normal | `[[1, 2, 3]]` | `6` |
| empty | boundary | `[[]]` | `0` |
| -1,1 | normal | `[[-1, 1]]` | `0` |
| single | boundary | `[[10]]` | `10` |

### py_18 — `def factorial_iter(n):`

- **Docstring:** Implement factorial iteratively.
- **Behaviour:** Return n! for integer n >= 0 using an iterative loop; 0! == 1.

| case | kind | args | expected |
|---|---|---|---|
| n=5 | normal | `[5]` | `120` |
| n=0 | boundary | `[0]` | `1` |
| n=1 | boundary | `[1]` | `1` |
| n=6 | normal | `[6]` | `720` |

### py_19 — `def my_len(s):`

- **Docstring:** Implement string length without len().
- **Behaviour:** Return the number of characters in s without calling the built-in len().
- **Resolution:** The 'without len()' constraint is a style requirement that cannot be verified by behaviour alone; it is not enforced by the evaluator.

| case | kind | args | expected |
|---|---|---|---|
| hello | normal | `["hello"]` | `5` |
| empty | boundary | `[""]` | `0` |
| single | boundary | `["a"]` | `1` |
| spaces | normal | `["a b"]` | `3` |

### py_20 — `def is_leap_year(y):`

- **Docstring:** Check if a year is a leap year.
- **Behaviour:** Return True iff y is a Gregorian leap year: divisible by 4, except centuries, except multiples of 400.

| case | kind | args | expected |
|---|---|---|---|
| 2024 | normal | `[2024]` | `true` |
| 1900 | boundary | `[1900]` | `false` |
| 2000 | boundary | `[2000]` | `true` |
| 2023 | normal | `[2023]` | `false` |
| 1600 | boundary | `[1600]` | `true` |

### py_21 — `def countdown(n):`

- **Docstring:** Implement a countdown.
- **Behaviour:** Return a list of integers counting down from n to 1 inclusive; n <= 0 returns [].
- **Resolution:** Return shape is unstated (a printed countdown is equally plausible). Resolved to returning a list; a printing-only implementation fails.

| case | kind | args | expected |
|---|---|---|---|
| 5 | normal | `[5]` | `[5, 4, 3, 2, 1]` |
| 1 | boundary | `[1]` | `[1]` |
| 0 | boundary | `[0]` | `[]` |
| 3 | normal | `[3]` | `[3, 2, 1]` |

### py_22 — `def safe_factorial(n):`

- **Docstring:** Implement factorial with error checking.
- **Behaviour:** Return n! for integer n >= 0 and raise ValueError when n < 0.
- **Resolution:** The docstring says 'with error checking' but names no type; resolved to ValueError, the conventional Python choice, and stated in the specification.

| case | kind | args | expected |
|---|---|---|---|
| n=5 | normal | `[5]` | `120` |
| n=3 | normal | `[3]` | `6` |
| n=0 | boundary | `[0]` | `1` |
| n=-1 raises ValueError | error | `[-1]` | `raises ValueError` |
| n=-5 raises ValueError | error | `[-5]` | `raises ValueError` |

### py_23 — `def caesar_cipher(s, n):`

- **Docstring:** Implement a simple cipher (shift by n).
- **Behaviour:** Shift every ASCII letter of s forward by n positions, wrapping within the same case (a-z and A-Z separately); non-letters are unchanged.
- **Resolution:** Alphabet, case handling and non-letter handling are unstated. Resolved to case-preserving ASCII letters with wrap and unchanged punctuation; the mixed case case is flagged as a convention.
- **Convention:** Lowercase-only cases are mandatory; the mixed-case/punctuation case is reported separately.

| case | kind | args | expected |
|---|---|---|---|
| abc+1 | normal | `["abc", 1]` | `"bcd"` |
| xyz+3 | normal | `["xyz", 3]` | `"abc"` |
| hello+0 | boundary | `["hello", 0]` | `"hello"` |
| abc+25 | normal | `["abc", 25]` | `"zab"` |
| mixed+1 | normal* | `["Hello, World!", 1]` | `"Ifmmp, Xpsme!"` |

### py_24 — `def fib_memo(n, memo={}):`

- **Docstring:** Implement Fibonacci with memoization.
- **Behaviour:** Return the n-th Fibonacci number (0-based, F(0)=0, F(1)=1) using the memo dict for caching; calling with a single argument must work.
- **Resolution:** Calling convention for memo is unstated; every case calls with a single argument, so the default from the stub must work (a mutable default is fine).

| case | kind | args | expected |
|---|---|---|---|
| n=0 | boundary | `[0]` | `0` |
| n=1 | boundary | `[1]` | `1` |
| n=10 | normal | `[10]` | `55` |
| n=25 | normal | `[25]` | `75025` |

### py_25 — `def get_evens(lst):`

- **Docstring:** Implement a function that returns list of even numbers.
- **Behaviour:** Return the even elements of lst in their original order.

| case | kind | args | expected |
|---|---|---|---|
| 1,2,3,4 | normal | `[[1, 2, 3, 4]]` | `[2, 4]` |
| empty | boundary | `[[]]` | `[]` |
| all even | normal | `[[2, 4]]` | `[2, 4]` |
| no even | boundary | `[[1, 3]]` | `[]` |

### py_26 — `def get_odds(lst):`

- **Docstring:** Implement a function that returns list of odd numbers.
- **Behaviour:** Return the odd elements of lst in their original order.

| case | kind | args | expected |
|---|---|---|---|
| 1,2,3,4 | normal | `[[1, 2, 3, 4]]` | `[1, 3]` |
| empty | boundary | `[[]]` | `[]` |
| no odd | boundary | `[[2, 4]]` | `[]` |
| all odd | normal | `[[1, 3]]` | `[1, 3]` |

### py_27 — `def filter_positives(lst):`

- **Docstring:** Implement a function that filters positive numbers.
- **Behaviour:** Return the elements of lst strictly greater than zero, preserving order.
- **Resolution:** 'Positive' is resolved to > 0, so 0 is excluded (the standard mathematical meaning).

| case | kind | args | expected |
|---|---|---|---|
| -2,0,3,5 | normal | `[[-2, 0, 3, 5]]` | `[3, 5]` |
| empty | boundary | `[[]]` | `[]` |
| only zero | boundary | `[[0]]` | `[]` |
| single positive | normal | `[[1]]` | `[1]` |

### py_28 — `def string_lengths(lst):`

- **Docstring:** Implement a function that maps strings to their lengths.
- **Behaviour:** Map each string in lst to its integer character count.

| case | kind | args | expected |
|---|---|---|---|
| a,bb,ccc | normal | `[["a", "bb", "ccc"]]` | `[1, 2, 3]` |
| ab,c | normal | `[["ab", "c"]]` | `[2, 1]` |
| empty | boundary | `[[]]` | `[]` |
| with empty string | boundary | `[["", "ab"]]` | `[0, 2]` |

### py_29 — `def is_sorted(lst):`

- **Docstring:** Implement a function that checks if a list is sorted.
- **Behaviour:** Return True iff lst is in non-decreasing order; the empty list and singletons are sorted.
- **Resolution:** 'Sorted' is resolved to non-decreasing (equal neighbours allowed).

| case | kind | args | expected |
|---|---|---|---|
| ascending | normal | `[[1, 2, 3]]` | `true` |
| descending | normal | `[[3, 2, 1]]` | `false` |
| empty | boundary | `[[]]` | `true` |
| single | boundary | `[[5]]` | `true` |
| equal pair | boundary | `[[1, 1, 2]]` | `true` |

### py_30 — `def fib_recursive(n):`

- **Docstring:** Implement a function that returns the nth Fibonacci number using recursion.
- **Behaviour:** Return the n-th Fibonacci number (0-based) using recursion.

| case | kind | args | expected |
|---|---|---|---|
| n=0 | boundary | `[0]` | `0` |
| n=1 | boundary | `[1]` | `1` |
| n=7 | normal | `[7]` | `13` |
| n=15 | normal | `[15]` | `610` |

### py_31 — `def remove_duplicates(lst):`

- **Docstring:** Implement a function that removes duplicates from a list.
- **Behaviour:** Return a list of the distinct elements of lst, keeping the order of first appearance.
- **Resolution:** Output order for duplicates is unstated; resolved to first-occurrence order.

| case | kind | args | expected |
|---|---|---|---|
| 1,2,2,3,1 | normal | `[[1, 2, 2, 3, 1]]` | `[1, 2, 3]` |
| empty | boundary | `[[]]` | `[]` |
| all same | boundary | `[[1, 1, 1]]` | `[1]` |
| already unique | normal | `[[3, 1, 2]]` | `[3, 1, 2]` |

### py_32 — `def list_intersection(l1, l2):`

- **Docstring:** Implement a function that finds the intersection of two lists.
- **Behaviour:** Return the distinct elements that occur in both l1 and l2, in l1 order.
- **Resolution:** Set semantics vs multiset semantics and output order are unstated. Resolved to distinct elements in l1 order; the duplicate case is flagged as a convention because a multiset implementation would return [2, 2].
- **Convention:** A multiset implementation fails only the duplicate case, which is excluded from the sensitivity metric.

| case | kind | args | expected |
|---|---|---|---|
| 1,2,3 & 2,3,4 | normal | `[[1, 2, 3], [2, 3, 4]]` | `[2, 3]` |
| empty left | boundary | `[[], [1]]` | `[]` |
| disjoint | normal | `[[1, 2], [3, 4]]` | `[]` |
| duplicate in l1 | boundary* | `[[1, 2, 2], [2]]` | `[2]` |

### py_33 — `def list_union(l1, l2):`

- **Docstring:** Implement a function that finds the union of two lists.
- **Behaviour:** Return the distinct elements of l1 followed by the distinct elements of l2 not already present.
- **Resolution:** Order of the union is unstated; resolved to l1 order then new l2 elements.

| case | kind | args | expected |
|---|---|---|---|
| disjoint | normal | `[[1, 2], [3, 4]]` | `[1, 2, 3, 4]` |
| overlap | normal | `[[1, 2], [2, 3]]` | `[1, 2, 3]` |
| empty left | boundary | `[[], [1]]` | `[1]` |
| both empty | boundary | `[[], []]` | `[]` |

### py_34 — `def flatten(lst):`

- **Docstring:** Implement a function that flattens a nested list.
- **Behaviour:** Return a flat list containing every non-list element of the arbitrarily nested lst, in depth-first order.
- **Resolution:** Nesting depth is unstated; resolved to full recursive flattening, so single-level implementations fail the deep case.

| case | kind | args | expected |
|---|---|---|---|
| nested | normal | `[[1, [2, [3, 4]], 5]]` | `[1, 2, 3, 4, 5]` |
| empty inner | boundary | `[[[], [1]]]` | `[1]` |
| already flat | normal | `[[1, 2]]` | `[1, 2]` |
| deep | boundary | `[[[[1]], 2]]` | `[1, 2]` |

### py_35 — `def binary_search_simple(arr, target):`

- **Docstring:** Implement a function that checks if a value is in a sorted list.
- **Behaviour:** Return True iff target occurs in the ascending list arr, else False.
- **Resolution:** 'Checks if a value is in a sorted list' is resolved to a boolean membership result (unlike binary_search, which returns an index).

| case | kind | args | expected |
|---|---|---|---|
| present | normal | `[[1, 3, 5], 3]` | `true` |
| absent | normal | `[[1, 3, 5], 4]` | `false` |
| empty | boundary | `[[], 1]` | `false` |

### py_36 — `def running_total(lst):`

- **Docstring:** Implement a function that computes running total.
- **Behaviour:** Return the list of prefix sums of lst; the empty list yields [].

| case | kind | args | expected |
|---|---|---|---|
| 1,2,3 | normal | `[[1, 2, 3]]` | `[1, 3, 6]` |
| empty | boundary | `[[]]` | `[]` |
| single | boundary | `[[5]]` | `[5]` |
| -1,1 | normal | `[[-1, 1]]` | `[-1, 0]` |

### py_37 — `def selection_sort(arr):`

- **Docstring:** Implement a function that sorts a list using selection sort.
- **Behaviour:** Return a new list with arr sorted ascending using selection sort.

| case | kind | args | expected |
|---|---|---|---|
| 64,25,12 | normal | `[[64, 25, 12]]` | `[12, 25, 64]` |
| empty | boundary | `[[]]` | `[]` |
| single | boundary | `[[3]]` | `[3]` |
| duplicates | normal | `[[2, 2, 1]]` | `[1, 2, 2]` |

### py_38 — `def caesar_wrap(s, n):`

- **Docstring:** Implement a function that implements a simple Caesar cipher with wrap.
- **Behaviour:** Shift ASCII letters of s by n, where n may exceed 25 and wraps modulo 26; case is preserved and non-letters are unchanged.
- **Resolution:** Wrap-around beyond 25 is the stated point of this task; resolved to shift modulo 26. Negative shifts are unspecified and untested.

| case | kind | args | expected |
|---|---|---|---|
| xyz+3 | normal | `["xyz", 3]` | `"abc"` |
| hello+2 | normal | `["hello", 2]` | `"jgnnq"` |
| abc+29 | boundary | `["abc", 29]` | `"def"` |
| hello+0 | boundary | `["hello", 0]` | `"hello"` |
| abc+52 | boundary | `["abc", 52]` | `"abc"` |

### py_39 — `def same_word(w1, w2):`

- **Docstring:** Implement a function that checks if two words are the same case-insensitive.
- **Behaviour:** Return True iff w1 and w2 are equal ignoring ASCII letter case.

| case | kind | args | expected |
|---|---|---|---|
| Hello/hello | normal | `["Hello", "hello"]` | `true` |
| abc/abd | normal | `["abc", "abd"]` | `false` |
| A/a | boundary | `["A", "a"]` | `true` |
| same | normal | `["test", "test"]` | `true` |

### py_40 — `def my_abs(x):`

- **Docstring:** Implement a function that returns the absolute value.
- **Behaviour:** Return the absolute value of x without calling the built-in abs().
- **Resolution:** The 'without abs()' constraint is stylistic and not enforced behaviourally.

| case | kind | args | expected |
|---|---|---|---|
| -5 | normal | `[-5]` | `5` |
| 5 | normal | `[5]` | `5` |
| 0 | boundary | `[0]` | `0` |
| -3.5 | boundary | `[-3.5]` | `3.5` |
| 2.5 | normal | `[2.5]` | `2.5` |

### py_41 — `def power(base, exp):`

- **Docstring:** Implement a function that computes power (base^exp).
- **Behaviour:** Return base raised to the non-negative integer exponent exp, computed by repeated multiplication; exp == 0 yields 1 for any base.
- **Resolution:** Negative exponents are unspecified and untested.

| case | kind | args | expected |
|---|---|---|---|
| 2^10 | normal | `[2, 10]` | `1024` |
| 5^0 | boundary | `[5, 0]` | `1` |
| 3^3 | normal | `[3, 3]` | `27` |
| 1^100 | boundary | `[1, 100]` | `1` |

### py_42 — `def is_digits(s):`

- **Docstring:** Implement a function that checks if a string contains only digits.
- **Behaviour:** Return True iff s is non-empty and every character is an ASCII digit 0-9.
- **Resolution:** The empty string is unstated; resolved to False, matching str.isdigit('').

| case | kind | args | expected |
|---|---|---|---|
| 123 | normal | `["123"]` | `true` |
| 12a | normal | `["12a"]` | `false` |
| empty | boundary | `[""]` | `false` |
| leading zero | boundary | `["007"]` | `true` |

### py_43 — `def min_of_three(a, b, c):`

- **Docstring:** Implement a function that finds the minimum of three numbers.
- **Behaviour:** Return the smallest of a, b and c.

| case | kind | args | expected |
|---|---|---|---|
| 1,2,3 | normal | `[1, 2, 3]` | `1` |
| 5,2,9 | normal | `[5, 2, 9]` | `2` |
| all negative | normal | `[-1, -5, -3]` | `-5` |
| all equal | boundary | `[2, 2, 2]` | `2` |

### py_44 — `def c_to_f(c):`

- **Docstring:** Implement a function that converts Celsius to Fahrenheit.
- **Behaviour:** Convert a Celsius temperature to Fahrenheit: F = C * 9/5 + 32.

| case | kind | args | expected |
|---|---|---|---|
| 0C | boundary | `[0]` | `32` |
| 100C | normal | `[100]` | `212` |
| -40C | boundary | `[-40]` | `-40` |
| 10C | normal | `[10]` | `50` |

### py_45 — `def is_positive(n):`

- **Docstring:** Implement a function that checks if a number is positive.
- **Behaviour:** Return True iff n is strictly greater than zero.

| case | kind | args | expected |
|---|---|---|---|
| 5 | normal | `[5]` | `true` |
| -1 | normal | `[-1]` | `false` |
| 0 | boundary | `[0]` | `false` |

### py_46 — `def nth_even(n):`

- **Docstring:** Implement a function that returns the nth even number.
- **Behaviour:** Return the n-th positive even number with 1-based indexing: nth_even(1) == 2.
- **Resolution:** Indexing unstated; resolved to 1-based so the first even number is index 1.

| case | kind | args | expected |
|---|---|---|---|
| n=1 | boundary | `[1]` | `2` |
| n=5 | normal | `[5]` | `10` |
| n=10 | normal | `[10]` | `20` |

### py_47 — `def my_mod(a, b):`

- **Docstring:** Implement a function that computes the modulus.
- **Behaviour:** Return a modulo b (the remainder of integer division) without calling the built-in % operator.
- **Resolution:** Only non-negative operands are tested because Python and JavaScript disagree on the sign of a negative remainder, and the docstring does not specify.

| case | kind | args | expected |
|---|---|---|---|
| 10%3 | normal | `[10, 3]` | `1` |
| 9%3 | boundary | `[9, 3]` | `0` |
| 7%4 | normal | `[7, 4]` | `3` |

### py_48 — `def is_blank(s):`

- **Docstring:** Implement a function that checks if a string is blank.
- **Behaviour:** Return True iff s is empty or contains only whitespace.
- **Resolution:** 'Blank' is unstated; resolved to empty-or-whitespace-only.

| case | kind | args | expected |
|---|---|---|---|
| empty | boundary | `[""]` | `true` |
| spaces | boundary | `["   "]` | `true` |
| padded text | normal | `[" a "]` | `false` |
| text | normal | `["a"]` | `false` |

### py_49 — `def square(x):`

- **Docstring:** Implement a function that returns the square of a number.
- **Behaviour:** Return the square of x.

| case | kind | args | expected |
|---|---|---|---|
| 4 | normal | `[4]` | `16` |
| 0 | boundary | `[0]` | `0` |
| -3 | normal | `[-3]` | `9` |
| 2.5 | normal | `[2.5]` | `6.25` |

### py_50 — `def fib_iter(n):`

- **Docstring:** Implement a function that computes the nth Fibonacci iteratively.
- **Behaviour:** Return the n-th Fibonacci number (0-based, F(0)=0, F(1)=1) using an iterative loop.

| case | kind | args | expected |
|---|---|---|---|
| n=0 | boundary | `[0]` | `0` |
| n=1 | boundary | `[1]` | `1` |
| n=10 | normal | `[10]` | `55` |
| n=30 | normal | `[30]` | `832040` |

## Javascript tasks

### js_01 — `function factorial(n) {`

- **Docstring:** Compute the factorial of n recursively.
- **Behaviour:** Return n! as a number for integer n >= 0; 0! === 1. Recursive implementation expected.
- **Resolution:** Negative n is unspecified and untested; no error contract is imposed.

| case | kind | args | expected |
|---|---|---|---|
| n=5 | normal | `[5]` | `120` |
| n=7 | normal | `[7]` | `5040` |
| n=0 | boundary | `[0]` | `1` |
| n=1 | boundary | `[1]` | `1` |

### js_02 — `function fibonacci(n) {`

- **Docstring:** Return the nth Fibonacci number.
- **Behaviour:** Return the n-th Fibonacci number with 0-based indexing: F(0)=0, F(1)=1.
- **Resolution:** Indexing is unstated; resolved to 0-based.

| case | kind | args | expected |
|---|---|---|---|
| n=0 | boundary | `[0]` | `0` |
| n=1 | boundary | `[1]` | `1` |
| n=10 | normal | `[10]` | `55` |
| n=20 | normal | `[20]` | `6765` |

### js_03 — `function isPalindrome(s) {`

- **Docstring:** Check whether a string is a palindrome.
- **Behaviour:** Return true iff s reads the same forwards and backwards; case-sensitive, no normalisation.
- **Resolution:** Case folding is unstated; resolved to exact comparison.

| case | kind | args | expected |
|---|---|---|---|
| racecar | normal | `["racecar"]` | `true` |
| hello | normal | `["hello"]` | `false` |
| empty | boundary | `[""]` | `true` |
| abba | normal | `["abba"]` | `true` |
| single | boundary | `["a"]` | `true` |

### js_04 — `function mergeSorted(a, b) {`

- **Docstring:** Merge two sorted arrays into one sorted array.
- **Behaviour:** Return a new ascending array containing every element of a and b (both already sorted); duplicates preserved.

| case | kind | args | expected |
|---|---|---|---|
| interleaved | normal | `[[1, 3, 5], [2, 4, 6]]` | `[1, 2, 3, 4, 5, 6]` |
| empty + list | boundary | `[[], [1, 2]]` | `[1, 2]` |
| both empty | boundary | `[[], []]` | `[]` |
| duplicates | normal | `[[1, 1], [1]]` | `[1, 1, 1]` |

### js_05 — `function binarySearch(arr, target) {`

- **Docstring:** Binary search on a sorted array.
- **Behaviour:** Search the ascending array arr for target; return its index or -1 when absent.
- **Resolution:** Miss value is unstated; resolved to -1 and flagged as a convention case (null, undefined or false are defensible alternatives).
- **Convention:** Miss cases are excluded from the sensitivity metric.

| case | kind | args | expected |
|---|---|---|---|
| found middle | normal | `[[1, 3, 5, 7, 9], 5]` | `2` |
| found first | normal | `[[1, 3, 5, 7, 9], 1]` | `0` |
| found last | normal | `[[1, 3, 5, 7, 9], 9]` | `4` |
| absent | normal* | `[[1, 3, 5, 7, 9], 4]` | `-1` |
| empty | boundary* | `[[], 1]` | `-1` |

### js_06 — `function nthPrime(n) {`

- **Docstring:** Return the nth prime number.
- **Behaviour:** Return the n-th prime with 1-based indexing: nthPrime(1) === 2.
- **Resolution:** Indexing is unstated; resolved to 1-based.

| case | kind | args | expected |
|---|---|---|---|
| n=1 | boundary | `[1]` | `2` |
| n=5 | normal | `[5]` | `11` |
| n=10 | normal | `[10]` | `29` |
| n=20 | normal | `[20]` | `71` |

### js_07 — `function reverseString(s) {`

- **Docstring:** Reverse a string.
- **Behaviour:** Return s with its characters in reverse order.

| case | kind | args | expected |
|---|---|---|---|
| hello | normal | `["hello"]` | `"olleh"` |
| empty | boundary | `[""]` | `""` |
| single | boundary | `["a"]` | `"a"` |
| two | normal | `["ab"]` | `"ba"` |

### js_08 — `function isAnagram(s1, s2) {`

- **Docstring:** Check if two strings are anagrams.
- **Behaviour:** Return true iff s1 and s2 have the same characters with the same multiplicities; case-sensitive.
- **Resolution:** Case sensitivity is unstated; resolved to exact, case-sensitive matching.

| case | kind | args | expected |
|---|---|---|---|
| listen/silent | normal | `["listen", "silent"]` | `true` |
| hello/world | normal | `["hello", "world"]` | `false` |
| abc/cba | normal | `["abc", "cba"]` | `true` |
| aab/abb | boundary | `["aab", "abb"]` | `false` |

### js_09 — `function gcd(a, b) {`

- **Docstring:** Compute the greatest common divisor.
- **Behaviour:** Return the greatest common divisor of a and b; gcd(a, 0) === a.

| case | kind | args | expected |
|---|---|---|---|
| 12,18 | normal | `[12, 18]` | `6` |
| 7,13 | normal | `[7, 13]` | `1` |
| 0,5 | boundary | `[0, 5]` | `5` |
| 5,0 | boundary | `[5, 0]` | `5` |
| 48,180 | normal | `[48, 180]` | `12` |

### js_10 — `function formatDate(dateStr) {`

- **Docstring:** Format a date from YYYY-MM-DD to DD/MM/YYYY.
- **Behaviour:** Convert a 'YYYY-MM-DD' string to 'DD/MM/YYYY'.

| case | kind | args | expected |
|---|---|---|---|
| 2024-01-05 | normal | `["2024-01-05"]` | `"05/01/2024"` |
| 1999-12-31 | normal | `["1999-12-31"]` | `"31/12/1999"` |
| 2026-11-17 | normal | `["2026-11-17"]` | `"17/11/2026"` |
| 0001-01-01 | boundary | `["0001-01-01"]` | `"01/01/0001"` |

### js_11 — `function isPrime(n) {`

- **Docstring:** Check if a number is prime.
- **Behaviour:** Return true iff n is prime; every n < 2 (including negatives) is false.
- **Resolution:** n < 2 is unstated; resolved to false.

| case | kind | args | expected |
|---|---|---|---|
| 2 | boundary | `[2]` | `true` |
| 17 | normal | `[17]` | `true` |
| 1 | boundary | `[1]` | `false` |
| 0 | boundary | `[0]` | `false` |
| 9 | normal | `[9]` | `false` |
| 97 | normal | `[97]` | `true` |
| -7 | boundary | `[-7]` | `false` |

### js_12 — `function triangularNumber(n) {`

- **Docstring:** Calculate the nth triangular number.
- **Behaviour:** Return the n-th triangular number n*(n+1)/2 for n >= 0.

| case | kind | args | expected |
|---|---|---|---|
| n=1 | boundary | `[1]` | `1` |
| n=5 | normal | `[5]` | `15` |
| n=10 | normal | `[10]` | `55` |
| n=0 | boundary | `[0]` | `0` |

### js_13 — `function linearSearch(arr, target) {`

- **Docstring:** Implement a linear search.
- **Behaviour:** Scan arr in order and return the index of the first occurrence of target, or -1.
- **Resolution:** Miss value is unstated; resolved to -1 and flagged as a convention case.
- **Convention:** A candidate returning null/undefined on a miss fails only the miss cases, which are excluded from the sensitivity metric.

| case | kind | args | expected |
|---|---|---|---|
| last | normal | `[[4, 2, 9], 9]` | `2` |
| first | normal | `[[4, 2, 9], 4]` | `0` |
| absent | normal* | `[[4, 2, 9], 7]` | `-1` |
| empty | boundary* | `[[], 1]` | `-1` |

### js_14 — `function findMax(arr) {`

- **Docstring:** Find the maximum element in an array.
- **Behaviour:** Return the largest element of the non-empty array arr.
- **Resolution:** Empty input is unspecified and untested.

| case | kind | args | expected |
|---|---|---|---|
| 1,5,3 | normal | `[[1, 5, 3]]` | `5` |
| negatives | normal | `[[-4, -2, -9]]` | `-2` |
| single | boundary | `[[7]]` | `7` |
| all equal | boundary | `[[2, 2]]` | `2` |

### js_15 — `function bubbleSort(arr) {`

- **Docstring:** Implement a simple sort (bubble sort).
- **Behaviour:** Return a new ascending array using bubble sort; the input array is not modified.

| case | kind | args | expected |
|---|---|---|---|
| 3,1,2 | normal | `[[3, 1, 2]]` | `[1, 2, 3]` |
| empty | boundary | `[[]]` | `[]` |
| single | boundary | `[[5]]` | `[5]` |
| duplicates | normal | `[[2, 2, 1]]` | `[1, 2, 2]` |
| negatives | normal | `[[-1, -5, 3]]` | `[-5, -1, 3]` |

### js_16 — `function isEven(n) {`

- **Docstring:** Check if a number is even.
- **Behaviour:** Return true iff n is divisible by 2, for positive, zero and negative integers.

| case | kind | args | expected |
|---|---|---|---|
| 4 | normal | `[4]` | `true` |
| 7 | normal | `[7]` | `false` |
| 0 | boundary | `[0]` | `true` |
| -2 | boundary | `[-2]` | `true` |
| -3 | boundary | `[-3]` | `false` |

### js_17 — `function sumArray(arr) {`

- **Docstring:** Compute the sum of array elements.
- **Behaviour:** Return the arithmetic sum of arr; the empty array sums to 0.

| case | kind | args | expected |
|---|---|---|---|
| 1,2,3 | normal | `[[1, 2, 3]]` | `6` |
| empty | boundary | `[[]]` | `0` |
| -1,1 | normal | `[[-1, 1]]` | `0` |
| single | boundary | `[[10]]` | `10` |

### js_18 — `function factorialIter(n) {`

- **Docstring:** Implement factorial iteratively.
- **Behaviour:** Return n! for integer n >= 0 using a loop; 0! === 1.

| case | kind | args | expected |
|---|---|---|---|
| n=5 | normal | `[5]` | `120` |
| n=0 | boundary | `[0]` | `1` |
| n=1 | boundary | `[1]` | `1` |
| n=6 | normal | `[6]` | `720` |

### js_19 — `function myLen(s) {`

- **Docstring:** Implement string length without length.
- **Behaviour:** Return the number of characters in s without using s.length.
- **Resolution:** The 'without length' constraint is stylistic and not enforced behaviourally.

| case | kind | args | expected |
|---|---|---|---|
| hello | normal | `["hello"]` | `5` |
| empty | boundary | `[""]` | `0` |
| single | boundary | `["a"]` | `1` |
| spaces | normal | `["a b"]` | `3` |

### js_20 — `function isLeapYear(y) {`

- **Docstring:** Check if a year is a leap year.
- **Behaviour:** Return true iff y is a Gregorian leap year: divisible by 4, except centuries, except multiples of 400.

| case | kind | args | expected |
|---|---|---|---|
| 2024 | normal | `[2024]` | `true` |
| 1900 | boundary | `[1900]` | `false` |
| 2000 | boundary | `[2000]` | `true` |
| 2023 | normal | `[2023]` | `false` |
| 1600 | boundary | `[1600]` | `true` |

### js_21 — `function countdown(n) {`

- **Docstring:** Implement a countdown.
- **Behaviour:** Return an array of integers from n down to 1 inclusive; n <= 0 returns [].
- **Resolution:** Return shape is unstated; resolved to returning an array.

| case | kind | args | expected |
|---|---|---|---|
| 5 | normal | `[5]` | `[5, 4, 3, 2, 1]` |
| 1 | boundary | `[1]` | `[1]` |
| 0 | boundary | `[0]` | `[]` |
| 3 | normal | `[3]` | `[3, 2, 1]` |

### js_22 — `function safeFactorial(n) {`

- **Docstring:** Implement factorial with error checking.
- **Behaviour:** Return n! for integer n >= 0 and throw an Error when n < 0.
- **Resolution:** The docstring says 'with error checking' but names no type. Resolved to throwing an Error instance (plain Error, RangeError or TypeError all accepted).

| case | kind | args | expected |
|---|---|---|---|
| n=5 | normal | `[5]` | `120` |
| n=3 | normal | `[3]` | `6` |
| n=0 | boundary | `[0]` | `1` |
| n=-1 throws | error | `[-1]` | `raises ['Error', 'RangeError', 'TypeError']` |
| n=-5 throws | error | `[-5]` | `raises ['Error', 'RangeError', 'TypeError']` |

### js_23 — `function caesarCipher(s, n) {`

- **Docstring:** Implement a simple cipher (shift by n).
- **Behaviour:** Shift every ASCII letter of s forward by n, wrapping within the same case (a-z, A-Z separately); non-letters unchanged.
- **Resolution:** Alphabet, case and non-letter handling are unstated; resolved to case-preserving ASCII shifting with punctuation unchanged.
- **Convention:** Lowercase-only cases are mandatory; the mixed-case case is reported separately.

| case | kind | args | expected |
|---|---|---|---|
| abc+1 | normal | `["abc", 1]` | `"bcd"` |
| xyz+3 | normal | `["xyz", 3]` | `"abc"` |
| hello+0 | boundary | `["hello", 0]` | `"hello"` |
| abc+25 | normal | `["abc", 25]` | `"zab"` |
| mixed+1 | normal* | `["Hello, World!", 1]` | `"Ifmmp, Xpsme!"` |

### js_24 — `function fibMemo(n, memo) {`

- **Docstring:** Implement Fibonacci with memoization.
- **Behaviour:** Return the n-th Fibonacci number (0-based) using the memo object for caching.
- **Resolution:** The stub's memo parameter has no default, so every case passes an empty object explicitly; this mirrors the Python cases, which rely on memo={}.

| case | kind | args | expected |
|---|---|---|---|
| n=0 | boundary | `[0, {}]` | `0` |
| n=1 | boundary | `[1, {}]` | `1` |
| n=10 | normal | `[10, {}]` | `55` |
| n=25 | normal | `[25, {}]` | `75025` |

### js_25 — `function getEvens(arr) {`

- **Docstring:** Implement function that returns list of even numbers.
- **Behaviour:** Return the even elements of arr in their original order.

| case | kind | args | expected |
|---|---|---|---|
| 1,2,3,4 | normal | `[[1, 2, 3, 4]]` | `[2, 4]` |
| empty | boundary | `[[]]` | `[]` |
| all even | normal | `[[2, 4]]` | `[2, 4]` |
| no even | boundary | `[[1, 3]]` | `[]` |

### js_26 — `function getOdds(arr) {`

- **Docstring:** Implement function that returns list of odd numbers.
- **Behaviour:** Return the odd elements of arr in their original order.

| case | kind | args | expected |
|---|---|---|---|
| 1,2,3,4 | normal | `[[1, 2, 3, 4]]` | `[1, 3]` |
| empty | boundary | `[[]]` | `[]` |
| no odd | boundary | `[[2, 4]]` | `[]` |
| all odd | normal | `[[1, 3]]` | `[1, 3]` |

### js_27 — `function filterPositives(arr) {`

- **Docstring:** Implement function that filters positive numbers.
- **Behaviour:** Return the elements of arr strictly greater than zero, preserving order.
- **Resolution:** 'Positive' resolved to > 0, so 0 is excluded.

| case | kind | args | expected |
|---|---|---|---|
| -2,0,3,5 | normal | `[[-2, 0, 3, 5]]` | `[3, 5]` |
| empty | boundary | `[[]]` | `[]` |
| only zero | boundary | `[[0]]` | `[]` |
| single positive | normal | `[[1]]` | `[1]` |

### js_28 — `function stringLengths(arr) {`

- **Docstring:** Implement function that maps strings to lengths.
- **Behaviour:** Map each string in arr to its character count.

| case | kind | args | expected |
|---|---|---|---|
| a,bb,ccc | normal | `[["a", "bb", "ccc"]]` | `[1, 2, 3]` |
| ab,c | normal | `[["ab", "c"]]` | `[2, 1]` |
| empty | boundary | `[[]]` | `[]` |
| with empty string | boundary | `[["", "ab"]]` | `[0, 2]` |

### js_29 — `function isSorted(arr) {`

- **Docstring:** Implement function that checks if a list is sorted.
- **Behaviour:** Return true iff arr is in non-decreasing order; empty arrays and singletons are sorted.
- **Resolution:** 'Sorted' resolved to non-decreasing.

| case | kind | args | expected |
|---|---|---|---|
| ascending | normal | `[[1, 2, 3]]` | `true` |
| descending | normal | `[[3, 2, 1]]` | `false` |
| empty | boundary | `[[]]` | `true` |
| single | boundary | `[[5]]` | `true` |
| equal pair | boundary | `[[1, 1, 2]]` | `true` |

### js_30 — `function fibRecursive(n) {`

- **Docstring:** Implement Fibonacci using recursion.
- **Behaviour:** Return the n-th Fibonacci number (0-based) using recursion.

| case | kind | args | expected |
|---|---|---|---|
| n=0 | boundary | `[0]` | `0` |
| n=1 | boundary | `[1]` | `1` |
| n=7 | normal | `[7]` | `13` |
| n=15 | normal | `[15]` | `610` |

### js_31 — `function removeDuplicates(arr) {`

- **Docstring:** Implement function that removes duplicates from array.
- **Behaviour:** Return the distinct elements of arr, keeping first-occurrence order.
- **Resolution:** Output order unstated; resolved to first-occurrence order.

| case | kind | args | expected |
|---|---|---|---|
| 1,2,2,3,1 | normal | `[[1, 2, 2, 3, 1]]` | `[1, 2, 3]` |
| empty | boundary | `[[]]` | `[]` |
| all same | boundary | `[[1, 1, 1]]` | `[1]` |
| already unique | normal | `[[3, 1, 2]]` | `[3, 1, 2]` |

### js_32 — `function listIntersection(arr1, arr2) {`

- **Docstring:** Implement function that finds intersection of two arrays.
- **Behaviour:** Return the distinct elements present in both arr1 and arr2, in arr1 order.
- **Resolution:** Set vs multiset semantics and order are unstated; resolved to distinct elements in arr1 order, with the duplicate case flagged as a convention.
- **Convention:** A multiset implementation fails only the duplicate case, which is excluded from the sensitivity metric.

| case | kind | args | expected |
|---|---|---|---|
| 1,2,3 & 2,3,4 | normal | `[[1, 2, 3], [2, 3, 4]]` | `[2, 3]` |
| empty left | boundary | `[[], [1]]` | `[]` |
| disjoint | normal | `[[1, 2], [3, 4]]` | `[]` |
| duplicate in arr1 | boundary* | `[[1, 2, 2], [2]]` | `[2]` |

### js_33 — `function listUnion(arr1, arr2) {`

- **Docstring:** Implement function that finds union of two arrays.
- **Behaviour:** Return the distinct elements of arr1 followed by the distinct elements of arr2 not already present.
- **Resolution:** Order unstated; resolved to arr1 order then new arr2 elements.

| case | kind | args | expected |
|---|---|---|---|
| disjoint | normal | `[[1, 2], [3, 4]]` | `[1, 2, 3, 4]` |
| overlap | normal | `[[1, 2], [2, 3]]` | `[1, 2, 3]` |
| empty left | boundary | `[[], [1]]` | `[1]` |
| both empty | boundary | `[[], []]` | `[]` |

### js_34 — `function flatten(arr) {`

- **Docstring:** Implement function that flattens nested array.
- **Behaviour:** Return a flat array of every non-array element of the arbitrarily nested arr, depth-first.
- **Resolution:** Nesting depth unstated; resolved to full recursive flattening.

| case | kind | args | expected |
|---|---|---|---|
| nested | normal | `[[1, [2, [3, 4]], 5]]` | `[1, 2, 3, 4, 5]` |
| empty inner | boundary | `[[[], [1]]]` | `[1]` |
| already flat | normal | `[[1, 2]]` | `[1, 2]` |
| deep | boundary | `[[[[1]], 2]]` | `[1, 2]` |

### js_35 — `function includesSorted(arr, target) {`

- **Docstring:** Implement function that checks if value is in sorted array.
- **Behaviour:** Return true iff target occurs in the ascending array arr, else false.
- **Resolution:** Resolved to a boolean membership result (not an index).

| case | kind | args | expected |
|---|---|---|---|
| present | normal | `[[1, 3, 5], 3]` | `true` |
| absent | normal | `[[1, 3, 5], 4]` | `false` |
| empty | boundary | `[[], 1]` | `false` |

### js_36 — `function runningTotal(arr) {`

- **Docstring:** Implement function that computes running total.
- **Behaviour:** Return the array of prefix sums of arr; empty input yields [].

| case | kind | args | expected |
|---|---|---|---|
| 1,2,3 | normal | `[[1, 2, 3]]` | `[1, 3, 6]` |
| empty | boundary | `[[]]` | `[]` |
| single | boundary | `[[5]]` | `[5]` |
| -1,1 | normal | `[[-1, 1]]` | `[-1, 0]` |

### js_37 — `function selectionSort(arr) {`

- **Docstring:** Implement selection sort.
- **Behaviour:** Return a new ascending array using selection sort.

| case | kind | args | expected |
|---|---|---|---|
| 64,25,12 | normal | `[[64, 25, 12]]` | `[12, 25, 64]` |
| empty | boundary | `[[]]` | `[]` |
| single | boundary | `[[3]]` | `[3]` |
| duplicates | normal | `[[2, 2, 1]]` | `[1, 2, 2]` |

### js_38 — `function caesarWrap(s, n) {`

- **Docstring:** Implement Caesar cipher with wrap-around.
- **Behaviour:** Shift ASCII letters of s by n, with n wrapping modulo 26; case preserved, non-letters unchanged.
- **Resolution:** Wrap beyond 25 is the point of the task; resolved to modulo 26. Negative shifts are unspecified and untested.

| case | kind | args | expected |
|---|---|---|---|
| xyz+3 | normal | `["xyz", 3]` | `"abc"` |
| hello+2 | normal | `["hello", 2]` | `"jgnnq"` |
| abc+29 | boundary | `["abc", 29]` | `"def"` |
| hello+0 | boundary | `["hello", 0]` | `"hello"` |
| abc+52 | boundary | `["abc", 52]` | `"abc"` |

### js_39 — `function sameWord(w1, w2) {`

- **Docstring:** Implement function that checks same word case-insensitive.
- **Behaviour:** Return true iff w1 and w2 are equal ignoring letter case.

| case | kind | args | expected |
|---|---|---|---|
| Hello/hello | normal | `["Hello", "hello"]` | `true` |
| abc/abd | normal | `["abc", "abd"]` | `false` |
| A/a | boundary | `["A", "a"]` | `true` |
| same | normal | `["test", "test"]` | `true` |

### js_40 — `function myAbs(n) {`

- **Docstring:** Implement function that returns absolute value.
- **Behaviour:** Return the absolute value of n without calling Math.abs.
- **Resolution:** The 'without Math.abs' constraint is stylistic and not enforced behaviourally.

| case | kind | args | expected |
|---|---|---|---|
| -5 | normal | `[-5]` | `5` |
| 5 | normal | `[5]` | `5` |
| 0 | boundary | `[0]` | `0` |
| -3.5 | boundary | `[-3.5]` | `3.5` |
| 2.5 | normal | `[2.5]` | `2.5` |

### js_41 — `function power(base, exp) {`

- **Docstring:** Implement function that computes power (base^exp).
- **Behaviour:** Return base raised to the non-negative integer exponent exp, computed by repeated multiplication; exp === 0 yields 1.
- **Resolution:** Negative exponents are unspecified and untested.

| case | kind | args | expected |
|---|---|---|---|
| 2^10 | normal | `[2, 10]` | `1024` |
| 5^0 | boundary | `[5, 0]` | `1` |
| 3^3 | normal | `[3, 3]` | `27` |
| 1^100 | boundary | `[1, 100]` | `1` |

### js_42 — `function isDigits(s) {`

- **Docstring:** Implement function that checks if string contains only digits.
- **Behaviour:** Return true iff s is non-empty and every character is an ASCII digit 0-9.
- **Resolution:** The empty string is unstated; resolved to false.

| case | kind | args | expected |
|---|---|---|---|
| 123 | normal | `["123"]` | `true` |
| 12a | normal | `["12a"]` | `false` |
| empty | boundary | `[""]` | `false` |
| leading zero | boundary | `["007"]` | `true` |

### js_43 — `function minOfThree(a, b, c) {`

- **Docstring:** Implement function that finds minimum of three numbers.
- **Behaviour:** Return the smallest of a, b and c.

| case | kind | args | expected |
|---|---|---|---|
| 1,2,3 | normal | `[1, 2, 3]` | `1` |
| 5,2,9 | normal | `[5, 2, 9]` | `2` |
| all negative | normal | `[-1, -5, -3]` | `-5` |
| all equal | boundary | `[2, 2, 2]` | `2` |

### js_44 — `function cToF(c) {`

- **Docstring:** Implement function that converts Celsius to Fahrenheit.
- **Behaviour:** Convert Celsius to Fahrenheit: F = C * 9/5 + 32.

| case | kind | args | expected |
|---|---|---|---|
| 0C | boundary | `[0]` | `32` |
| 100C | normal | `[100]` | `212` |
| -40C | boundary | `[-40]` | `-40` |
| 10C | normal | `[10]` | `50` |

### js_45 — `function isPositive(n) {`

- **Docstring:** Implement function that checks if a number is positive.
- **Behaviour:** Return true iff n is strictly greater than zero.

| case | kind | args | expected |
|---|---|---|---|
| 5 | normal | `[5]` | `true` |
| -1 | normal | `[-1]` | `false` |
| 0 | boundary | `[0]` | `false` |

### js_46 — `function nthEven(n) {`

- **Docstring:** Implement function that returns the nth even number.
- **Behaviour:** Return the n-th positive even number with 1-based indexing: nthEven(1) === 2.
- **Resolution:** Indexing unstated; resolved to 1-based.

| case | kind | args | expected |
|---|---|---|---|
| n=1 | boundary | `[1]` | `2` |
| n=5 | normal | `[5]` | `10` |
| n=10 | normal | `[10]` | `20` |

### js_47 — `function myMod(a, b) {`

- **Docstring:** Implement function that computes modulus.
- **Behaviour:** Return a modulo b (the remainder of integer division) without using the % operator.
- **Resolution:** Only non-negative operands are tested; negative remainders differ between the two languages and the docstring does not specify them.

| case | kind | args | expected |
|---|---|---|---|
| 10%3 | normal | `[10, 3]` | `1` |
| 9%3 | boundary | `[9, 3]` | `0` |
| 7%4 | normal | `[7, 4]` | `3` |

### js_48 — `function isBlank(s) {`

- **Docstring:** Implement function that checks if string is blank.
- **Behaviour:** Return true iff s is empty or contains only whitespace.
- **Resolution:** 'Blank' resolved to empty-or-whitespace-only.

| case | kind | args | expected |
|---|---|---|---|
| empty | boundary | `[""]` | `true` |
| spaces | boundary | `["   "]` | `true` |
| padded text | normal | `[" a "]` | `false` |
| text | normal | `["a"]` | `false` |

### js_49 — `function square(n) {`

- **Docstring:** Implement function that returns the square of a number.
- **Behaviour:** Return the square of n.

| case | kind | args | expected |
|---|---|---|---|
| 4 | normal | `[4]` | `16` |
| 0 | boundary | `[0]` | `0` |
| -3 | normal | `[-3]` | `9` |
| 2.5 | normal | `[2.5]` | `6.25` |

### js_50 — `function fibIter(n) {`

- **Docstring:** Implement function that computes the nth Fibonacci iteratively.
- **Behaviour:** Return the n-th Fibonacci number (0-based) using a loop.

| case | kind | args | expected |
|---|---|---|---|
| n=0 | boundary | `[0]` | `0` |
| n=1 | boundary | `[1]` | `1` |
| n=10 | normal | `[10]` | `55` |
| n=30 | normal | `[30]` | `832040` |

* = convention case (excluded from the sensitivity metric).
