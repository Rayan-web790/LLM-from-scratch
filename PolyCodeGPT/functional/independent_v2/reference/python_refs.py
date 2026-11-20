"""Independent v2 trusted reference implementations (Python)."""

from __future__ import annotations


def sumSquaredValues(nums):
    return sum(x * x for x in nums)


def count_positives(nums):
    return sum(1 for x in nums if x > 0)


def count_negatives(nums):
    return sum(1 for x in nums if x < 0)


def max_abs(nums):
    return max((abs(x) for x in nums), default=None)


def min_abs(nums):
    return min((abs(x) for x in nums), default=None)


def all_positive(nums):
    return all(x > 0 for x in nums)


def any_negative(nums):
    return any(x < 0 for x in nums)


def count_zeros(nums):
    return sum(1 for x in nums if x == 0)


def sum_first_n(nums, n):
    return sum(nums[:n])


def sum_last_n(nums, n):
    return sum(nums[-n:]) if n else 0


def mean_rounded(nums):
    if not nums:
        return 0
    return round(sum(nums) / len(nums), 2)


def median_simple(nums):
    if not nums:
        return 0.0
    s = sorted(nums)
    m = len(s) // 2
    if len(s) % 2:
        return float(s[m])
    return (s[m - 1] + s[m]) / 2


def range_of(nums):
    if not nums:
        return 0
    return max(nums) - min(nums)


def normalize_minmax(nums):
    if not nums:
        return []
    lo, hi = min(nums), max(nums)
    if lo == hi:
        return [0.0 for _ in nums]
    return [(x - lo) / (hi - lo) for x in nums]


def reverse_words(s):
    return " ".join(s.split()[::-1])


def count_words(s):
    return len(s.split())


def longest_word(s):
    words = s.split()
    if not words:
        return ""
    best = words[0]
    for w in words[1:]:
        if len(w) > len(best):
            best = w
    return best


def shortest_word(s):
    words = s.split()
    if not words:
        return ""
    best = words[0]
    for w in words[1:]:
        if len(w) < len(best):
            best = w
    return best


def words_sorted(s):
    return sorted(s.split())


def remove_punctuation(s):
    return "".join(ch if ch.isalnum() or ch.isspace() else "" for ch in s)


def is_substring(a, b):
    return a in b


def count_substring(s, sub):
    if not sub:
        return 0
    count = 0
    i = 0
    while True:
        j = s.find(sub, i)
        if j == -1:
            break
        count += 1
        i = j + len(sub)
    return count


def replaceAllOccurrences(s, old, new):
    return s.replace(old, new)


def starts_with_vowel(s):
    return bool(s) and s[0].lower() in "aeiou"


def ends_with_digit(s):
    return bool(s) and s[-1].isdigit()


def alternating_case(s):
    out = []
    for i, ch in enumerate(s):
        out.append(ch.upper() if i % 2 == 0 else ch.lower())
    return "".join(out)


def collapseWhitespaceRuns(s):
    return " ".join(s.split())


def truncateText(s, n):
    if len(s) <= n:
        return s
    return s[:n] + "..."


def pad_left(s, width, char=" "):
    if len(s) >= width:
        return s
    return char * (width - len(s)) + s


def checkPrimality(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def next_prime(n):
    if n < 2:
        n = 2
    while not checkPrimality(n):
        n += 1
    return n


def prime_count_upto(n):
    return sum(1 for x in range(2, n + 1) if checkPrimality(x))


def gcd_two(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def lcm_two(a, b):
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd_two(a, b)


def iterativeFactorialValue(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def raiseToPower(base, exp):
    r = 1
    for _ in range(exp):
        r *= base
    return r


def sqrt_floor(n):
    if n < 0:
        return 0
    i = 0
    while (i + 1) * (i + 1) <= n:
        i += 1
    return i


def abs_diff(a, b):
    return abs(a - b)


def signOfNumber(n):
    if n < 0:
        return -1
    if n > 0:
        return 1
    return 0


def clamp_list(nums, lo, hi):
    return [max(lo, min(hi, x)) for x in nums]


def filter_positive(nums):
    return [x for x in nums if x > 0]


def filter_negative(nums):
    return [x for x in nums if x < 0]


def square_list(nums):
    return [x * x for x in nums]


def cube_list(nums):
    return [x ** 3 for x in nums]


def double_list(nums):
    return [x * 2 for x in nums]


def negate_list(nums):
    return [-x for x in nums]


def sort_ascending(nums):
    return sorted(nums)


def sort_descending(nums):
    return sorted(nums, reverse=True)


def unique_sorted(nums):
    return sorted(set(nums))


def product_of_list(nums):
    r = 1
    for x in nums:
        r *= x
    return r
