"""Persistent Phase 10 benchmark task definitions.

This module provides the fixed 100-task inventory (50 Python + 50 JavaScript)
as specified in reports/phase10_benchmark_spec.md. These tasks are independent
of validation data, held-out test sets, or any dynamic data loading.

Each task is a (docstring, code_stub) pair. The generation prefix is built
via build_generation_prefix(language, docstring, code_stub), which produces
<|lang:X|><|docstring|>{docstring}<|code|>{code_stub} without a trailing
EOS marker. The canonical tokenizer is tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Python tasks (50)
# ─────────────────────────────────────────────────────────────────────────────

PYTHON_TASKS: tuple[tuple[str, str], ...] = (
    # Category: Function Implementation (1–10)
    ("Compute the factorial of n recursively.", "def factorial(n):"),
    ("Return the nth Fibonacci number.", "def fibonacci(n):"),
    ("Check whether a string is a palindrome.", "def is_palindrome(s):"),
    ("Merge two sorted lists into one sorted list.", "def merge_sorted(a, b):"),
    ("Binary search on a sorted list.", "def binary_search(arr, target):"),
    ("Return the nth prime number.", "def nth_prime(n):"),
    ("Reverse a string.", "def reverse_string(s):"),
    ("Check if two strings are anagrams.", "def is_anagram(s1, s2):"),
    ("Compute the greatest common divisor.", "def gcd(a, b):"),
    ("Format a date from YYYY-MM-DD to DD/MM/YYYY.", "def format_date(date_str):"),
    # Category: Algorithms (11–20)
    ("Check if a number is prime.", "def is_prime(n):"),
    ("Calculate the nth triangular number.", "def triangular_number(n):"),
    ("Implement a linear search.", "def linear_search(arr, target):"),
    ("Find the maximum element in a list.", "def find_max(arr):"),
    ("Implement a simple sort (bubble sort).", "def bubble_sort(arr):"),
    ("Check if a number is even.", "def is_even(n):"),
    ("Compute the sum of list elements.", "def sum_list(lst):"),
    ("Implement factorial iteratively.", "def factorial_iter(n):"),
    ("Implement string length without len().", "def my_len(s):"),
    ("Check if a year is a leap year.", "def is_leap_year(y):"),
    # Category: Data Structures (21–30)
    ("Implement a countdown.", "def countdown(n):"),
    ("Implement factorial with error checking.", "def safe_factorial(n):"),
    ("Implement a simple cipher (shift by n).", "def caesar_cipher(s, n):"),
    ("Implement Fibonacci with memoization.", "def fib_memo(n, memo={}):"),
    ("Implement a function that returns list of even numbers.", "def get_evens(lst):"),
    ("Implement a function that returns list of odd numbers.", "def get_odds(lst):"),
    ("Implement a function that filters positive numbers.", "def filter_positives(lst):"),
    ("Implement a function that maps strings to their lengths.", "def string_lengths(lst):"),
    ("Implement a function that checks if a list is sorted.", "def is_sorted(lst):"),
    ("Implement a function that returns the nth Fibonacci number using recursion.", "def fib_recursive(n):"),
    ("Implement a function that removes duplicates from a list.", "def remove_duplicates(lst):"),
    # Category: Debugging / Instruction Following (31–40)
    ("Implement a function that finds the intersection of two lists.", "def list_intersection(l1, l2):"),
    ("Implement a function that finds the union of two lists.", "def list_union(l1, l2):"),
    ("Implement a function that flattens a nested list.", "def flatten(lst):"),
    ("Implement a function that checks if a value is in a sorted list.", "def binary_search_simple(arr, target):"),
    ("Implement a function that computes running total.", "def running_total(lst):"),
    ("Implement a function that sorts a list using selection sort.", "def selection_sort(arr):"),
    ("Implement a function that implements a simple Caesar cipher with wrap.", "def caesar_wrap(s, n):"),
    ("Implement a function that checks if two words are the same case-insensitive.", "def same_word(w1, w2):"),
    ("Implement a function that returns the absolute value.", "def my_abs(x):"),
    ("Implement a function that computes power (base^exp).", "def power(base, exp):"),
    ("Implement a function that checks if a string contains only digits.", "def is_digits(s):"),
    # Category: More Functions (41–50)
    ("Implement a function that finds the minimum of three numbers.", "def min_of_three(a, b, c):"),
    ("Implement a function that converts Celsius to Fahrenheit.", "def c_to_f(c):"),
    ("Implement a function that checks if a number is positive.", "def is_positive(n):"),
    ("Implement a function that returns the nth even number.", "def nth_even(n):"),
    ("Implement a function that computes the modulus.", "def my_mod(a, b):"),
    ("Implement a function that checks if a string is blank.", "def is_blank(s):"),
    ("Implement a function that returns the square of a number.", "def square(x):"),
    ("Implement a function that computes the nth Fibonacci iteratively.", "def fib_iter(n):"),
)

# ─────────────────────────────────────────────────────────────────────────────
# JavaScript tasks (50)
# ─────────────────────────────────────────────────────────────────────────────

JS_TASKS: tuple[tuple[str, str], ...] = (
    # Category: Function Implementation (1–10)
    ("Compute the factorial of n recursively.", "function factorial(n) {"),
    ("Return the nth Fibonacci number.", "function fibonacci(n) {"),
    ("Check whether a string is a palindrome.", "function isPalindrome(s) {"),
    ("Merge two sorted arrays into one sorted array.", "function mergeSorted(a, b) {"),
    ("Binary search on a sorted array.", "function binarySearch(arr, target) {"),
    ("Return the nth prime number.", "function nthPrime(n) {"),
    ("Reverse a string.", "function reverseString(s) {"),
    ("Check if two strings are anagrams.", "function isAnagram(s1, s2) {"),
    ("Compute the greatest common divisor.", "function gcd(a, b) {"),
    ("Format a date from YYYY-MM-DD to DD/MM/YYYY.", "function formatDate(dateStr) {"),
    # Category: Algorithms (11–20)
    ("Check if a number is prime.", "function isPrime(n) {"),
    ("Calculate the nth triangular number.", "function triangularNumber(n) {"),
    ("Implement a linear search.", "function linearSearch(arr, target) {"),
    ("Find the maximum element in an array.", "function findMax(arr) {"),
    ("Implement a simple sort (bubble sort).", "function bubbleSort(arr) {"),
    ("Check if a number is even.", "function isEven(n) {"),
    ("Compute the sum of array elements.", "function sumArray(arr) {"),
    ("Implement factorial iteratively.", "function factorialIter(n) {"),
    ("Implement string length without length.", "function myLen(s) {"),
    ("Check if a year is a leap year.", "function isLeapYear(y) {"),
    # Category: Data Structures (21–30)
    ("Implement a countdown.", "function countdown(n) {"),
    ("Implement factorial with error checking.", "function safeFactorial(n) {"),
    ("Implement a simple cipher (shift by n).", "function caesarCipher(s, n) {"),
    ("Implement Fibonacci with memoization.", "function fibMemo(n, memo) {"),
    ("Implement function that returns list of even numbers.", "function getEvens(arr) {"),
    ("Implement function that returns list of odd numbers.", "function getOdds(arr) {"),
    ("Implement function that filters positive numbers.", "function filterPositives(arr) {"),
    ("Implement function that maps strings to lengths.", "function stringLengths(arr) {"),
    ("Implement function that checks if a list is sorted.", "function isSorted(arr) {"),
    ("Implement Fibonacci using recursion.", "function fibRecursive(n) {"),
    ("Implement function that removes duplicates from array.", "function removeDuplicates(arr) {"),
    # Category: Debugging / Instruction Following (31–40)
    ("Implement function that finds intersection of two arrays.", "function listIntersection(arr1, arr2) {"),
    ("Implement function that finds union of two arrays.", "function listUnion(arr1, arr2) {"),
    ("Implement function that flattens nested array.", "function flatten(arr) {"),
    ("Implement function that checks if value is in sorted array.", "function includesSorted(arr, target) {"),
    ("Implement function that computes running total.", "function runningTotal(arr) {"),
    ("Implement selection sort.", "function selectionSort(arr) {"),
    ("Implement Caesar cipher with wrap-around.", "function caesarWrap(s, n) {"),
    ("Implement function that checks same word case-insensitive.", "function sameWord(w1, w2) {"),
    ("Implement function that returns absolute value.", "function myAbs(n) {"),
    ("Implement function that computes power (base^exp).", "function power(base, exp) {"),
    # Category: More JS Functions (41–50)
    ("Implement function that checks if string contains only digits.", "function isDigits(s) {"),
    ("Implement function that finds minimum of three numbers.", "function minOfThree(a, b, c) {"),
    ("Implement function that converts Celsius to Fahrenheit.", "function cToF(c) {"),
    ("Implement function that checks if a number is positive.", "function isPositive(n) {"),
    ("Implement function that returns the nth even number.", "function nthEven(n) {"),
    ("Implement function that computes modulus.", "function myMod(a, b) {"),
    ("Implement function that checks if string is blank.", "function isBlank(s) {"),
    ("Implement function that returns the square of a number.", "function square(n) {"),
    ("Implement function that computes the nth Fibonacci iteratively.", "function fibIter(n) {"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Task inventory helpers
# ─────────────────────────────────────────────────────────────────────────────

def python_tasks() -> tuple[tuple[str, str], ...]:
    """Return the 50 fixed Python task (docstring, code_stub) pairs."""
    return PYTHON_TASKS


def javascript_tasks() -> tuple[tuple[str, str], ...]:
    """Return the 50 fixed JavaScript task (docstring, code_stub) pairs."""
    return JS_TASKS


def task_counts() -> dict[str, int]:
    """Return the number of tasks per language."""
    return {"python": len(PYTHON_TASKS), "javascript": len(JS_TASKS), "total": len(PYTHON_TASKS) + len(JS_TASKS)}


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: produce all 100 generation prefixes (no model load)
# ─────────────────────────────────────────────────────────────────────────────

def all_generation_prefixes(
    tokenizer_path: str = "tokenizer/artifacts/polycodegpt_tokenizer_v1_1.json",
) -> dict[str, list[dict]]:
    """Build every generation prefix and verify tokenizer roundtrip.

    Returns a dict with keys ``python`` and ``javascript``, each mapping to
    a list of per-task dicts containing the prefix string, token counts, and
    encode/decode roundtrip status.

    This function does *not* load the model or access any binary dataset.
    """
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.evaluate_phase10_benchmark import build_generation_prefix
    from src.polycodegpt.data.tokenizer_io import load_tokenizer

    tokenizer = load_tokenizer(tokenizer_path)

    result: dict[str, list[dict]] = {"python": [], "javascript": []}

    for docstring, code_stub in PYTHON_TASKS:
        prefix = build_generation_prefix("python", docstring, code_stub)
        encoded = tokenizer.encode(prefix, add_special_tokens=False).ids
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        result["python"].append(
            {
                "docstring": docstring,
                "code_stub": code_stub,
                "prefix": prefix,
                "code_count": prefix.count("<|code|>"),
                "ds_count": prefix.count("<|docstring|>"),
                "eos_count": prefix.count("<|eos|>"),
                "roundtrip_ok": decoded == prefix,
            }
        )

    for docstring, code_stub in JS_TASKS:
        prefix = build_generation_prefix("javascript", docstring, code_stub)
        encoded = tokenizer.encode(prefix, add_special_tokens=False).ids
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        result["javascript"].append(
            {
                "docstring": docstring,
                "code_stub": code_stub,
                "prefix": prefix,
                "code_count": prefix.count("<|code|>"),
                "ds_count": prefix.count("<|docstring|>"),
                "eos_count": prefix.count("<|eos|>"),
                "roundtrip_ok": decoded == prefix,
            }
        )

    return result