"""Phase 11 trusted reference implementations for the 50 Python tasks.

Written from the specification in ``functional/manifest.json`` only. These are the
oracle used to validate the test cases; they are never compared against, merged
with, or shown to candidate code, and they are executed in the same sandbox as
candidates when validating the benchmark itself.
"""

from __future__ import annotations

# ── 01-10 ────────────────────────────────────────────────────────────────────

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def fibonacci(n):
    if n < 2:
        return n
    previous, current = 0, 1
    for _ in range(n - 1):
        previous, current = current, previous + current
    return current


def is_palindrome(s):
    return s == s[::-1]


def merge_sorted(a, b):
    merged = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1
    merged.extend(a[i:])
    merged.extend(b[j:])
    return merged


def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def nth_prime(n):
    count = 0
    candidate = 1
    while count < n:
        candidate += 1
        if is_prime(candidate):
            count += 1
    return candidate


def reverse_string(s):
    return s[::-1]


def is_anagram(s1, s2):
    return sorted(s1) == sorted(s2)


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def format_date(date_str):
    year, month, day = date_str.split("-")
    return f"{day}/{month}/{year}"


# ── 11-20 ────────────────────────────────────────────────────────────────────

def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    divisor = 3
    while divisor * divisor <= n:
        if n % divisor == 0:
            return False
        divisor += 2
    return True


def triangular_number(n):
    return n * (n + 1) // 2


def linear_search(arr, target):
    for index, value in enumerate(arr):
        if value == target:
            return index
    return -1


def find_max(arr):
    best = arr[0]
    for value in arr[1:]:
        if value > best:
            best = value
    return best


def bubble_sort(arr):
    result = list(arr)
    n = len(result)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True
        if not swapped:
            break
    return result


def is_even(n):
    return n % 2 == 0


def sum_list(lst):
    total = 0
    for value in lst:
        total += value
    return total


def factorial_iter(n):
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result


def my_len(s):
    count = 0
    for _ in s:
        count += 1
    return count


def is_leap_year(y):
    if y % 400 == 0:
        return True
    if y % 100 == 0:
        return False
    return y % 4 == 0


# ── 21-30 ────────────────────────────────────────────────────────────────────

def countdown(n):
    return list(range(n, 0, -1))


def safe_factorial(n):
    if n < 0:
        raise ValueError("factorial is undefined for negative integers")
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result


def _shift_char(ch, n):
    if "a" <= ch <= "z":
        return chr((ord(ch) - 97 + n) % 26 + 97)
    if "A" <= ch <= "Z":
        return chr((ord(ch) - 65 + n) % 26 + 65)
    return ch


def caesar_cipher(s, n):
    return "".join(_shift_char(ch, n) for ch in s)


def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n < 2:
        memo[n] = n
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]


def get_evens(lst):
    return [value for value in lst if value % 2 == 0]


def get_odds(lst):
    return [value for value in lst if value % 2 != 0]


def filter_positives(lst):
    return [value for value in lst if value > 0]


def string_lengths(lst):
    return [my_len(item) for item in lst]


def is_sorted(lst):
    for i in range(1, len(lst)):
        if lst[i - 1] > lst[i]:
            return False
    return True


def fib_recursive(n):
    if n < 2:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)


# ── 31-40 ────────────────────────────────────────────────────────────────────

def remove_duplicates(lst):
    seen = []
    for value in lst:
        if value not in seen:
            seen.append(value)
    return seen


def list_intersection(l1, l2):
    result = []
    for value in l1:
        if value in l2 and value not in result:
            result.append(value)
    return result


def list_union(l1, l2):
    result = []
    for value in list(l1) + list(l2):
        if value not in result:
            result.append(value)
    return result


def flatten(lst):
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat


def binary_search_simple(arr, target):
    return binary_search(arr, target) != -1


def running_total(lst):
    total = 0
    result = []
    for value in lst:
        total += value
        result.append(total)
    return result


def selection_sort(arr):
    result = list(arr)
    for i in range(len(result)):
        smallest = i
        for j in range(i + 1, len(result)):
            if result[j] < result[smallest]:
                smallest = j
        result[i], result[smallest] = result[smallest], result[i]
    return result


def caesar_wrap(s, n):
    return "".join(_shift_char(ch, n) for ch in s)


def same_word(w1, w2):
    return w1.lower() == w2.lower()


def my_abs(x):
    return -x if x < 0 else x


# ── 41-50 ────────────────────────────────────────────────────────────────────

def power(base, exp):
    result = 1
    for _ in range(exp):
        result *= base
    return result


def is_digits(s):
    if not s:
        return False
    for ch in s:
        if ch < "0" or ch > "9":
            return False
    return True


def min_of_three(a, b, c):
    smallest = a
    if b < smallest:
        smallest = b
    if c < smallest:
        smallest = c
    return smallest


def c_to_f(c):
    return c * 9 / 5 + 32


def is_positive(n):
    return n > 0


def nth_even(n):
    return 2 * n


def my_mod(a, b):
    quotient = a // b
    return a - quotient * b


def is_blank(s):
    return s.strip() == ""


def square(x):
    return x * x


def fib_iter(n):
    previous, current = 0, 1
    for _ in range(n):
        previous, current = current, previous + current
    return previous
