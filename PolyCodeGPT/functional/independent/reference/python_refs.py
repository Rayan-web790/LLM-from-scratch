"""Independent trusted reference implementations (Python)."""

from __future__ import annotations


def triangularSum(n):
    return n * (n + 1) // 2


def areAnagrams(a, b):
    return sorted(a.lower()) == sorted(b.lower())


def running_sum(nums):
    out = []
    total = 0
    for x in nums:
        total += x
        out.append(total)
    return out


def dropDuplicateValues(nums):
    seen = set()
    out = []
    for x in nums:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def longest_common_prefix(strs):
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def boundValue(x, lo, hi):
    return max(lo, min(hi, x))


def tallyWords(s):
    counts = {}
    for w in s.split():
        counts[w] = counts.get(w, 0) + 1
    return counts


def rotate_list(nums, k):
    if not nums:
        return []
    k = k % len(nums)
    return nums[-k:] + nums[:-k] if k else list(nums)


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def second_largest(nums):
    distinct = sorted(set(nums), reverse=True)
    if len(distinct) < 2:
        return None
    return distinct[1]
