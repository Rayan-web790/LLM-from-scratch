# Phase 10 Benchmark Specification

## 100-Task Fixed Benchmark

### Overview
- **Total tasks**: 100 (50 Python + 50 JavaScript)
- **Language markers**: `<|lang:python|>` / `<|lang:javascript|>`
- **Serialization format**: `<|lang:X|><|docstring|>{docstring}<|code|>{code_stub}<|eos|>`
- **Generation**: temperature=0, max_new_tokens=64
- **Tokenizer**: v1.1.0 (BPE)
- **No test data access**: only valid.jsonl for any ablations

### Task Categories (each language: 10 per category)

| Category | Description | Python Examples | JavaScript Examples |
|---|---|---|---|
| **Function Implementation** | Write a function from docstring spec | factorial, fibonacci, palindrome check | factorial, fibonacci, palindrome check |
| **Algorithms** | Sorting, searching, graph traversal | merge sorted lists, binary search | merge sorted arrays, binary search |
| **Data Structures** | Stack, queue, tree, list operations | not strongly represented | not strongly represented |
| **Debugging** | Fix broken code / find bug | not strongly represented | not strongly represented |
| **Code Completion** | Complete a partial function stub | def merge_sorted(...), def is_palindrome(...) | function mergeSorted(...), function isPalindrome(...) |
| **Instruction Following** | Follow specific formatting/constraints | return type hints, specific variable names | return type annotations, specific variable names |

### Python Tasks (50)

| # | Docstring | Code Stub | Test Cases |
|---|-----------|-----------|------------|
| 1 | Compute the factorial of n recursively. | `def factorial(n):` | `factorial(5) == 120`, `factorial(0) == 1`, `factorial(1) == 1` |
| 2 | Return the nth Fibonacci number. | `def fibonacci(n):` | `fibonacci(0) == 0`, `fibonacci(1) == 1`, `fibonacci(10) == 55` |
| 3 | Check whether a string is a palindrome. | `def is_palindrome(s):` | `is_palindrome("racecar") == True`, `is_palindrome("hello") == False`, `is_palindrome("") == True` |
| 4 | Merge two sorted lists into one sorted list. | `def merge_sorted(a, b):` | `merge_sorted([1,3,5], [2,4,6]) == [1,2,3,4,5,6]` |
| 5 | Binary search on a sorted list. | `def binary_search(arr, target):` | `binary_search([1,2,3,4,5], 3) == 2`, `binary_search([1,2,3,4,5], 6) == -1` |
| 6 | Return the nth prime number. | `def nth_prime(n):` | `nth_prime(1) == 2`, `nth_prime(2) == 3`, `nth_prime(3) == 5` |
| 7 | Reverse a string. | `def reverse_string(s):` | `reverse_string("hello") == "olleh"`, `reverse_string("") == ""` |
| 8 | Check if two strings are anagrams. | `def is_anagram(s1, s2):` | `is_anagram("listen", "silent") == True`, `is_anagram("hello", "world") == False` |
| 9 | Compute the greatest common divisor. | `def gcd(a, b):` | `gcd(48, 18) == 6`, `gcd(17, 5) == 1` |
| 10 | Format a date from YYYY-MM-DD to DD/MM/YYYY. | `def format_date(date_str):` | `format_date("2023-01-15") == "15/01/2023"` |
| 11 | Check if a number is prime. | `def is_prime(n):` | `is_prime(7) == True`, `is_prime(1) == False`, `is_prime(4) == False` |
| 12 | Calculate the nth triangular number. | `def triangular_number(n):` | `triangular_number(1) == 1`, `triangular_number(4) == 10`, `triangular_number(10) == 55` |
| 13 | Implement a linear search. | `def linear_search(arr, target):` | `linear_search([1,2,3], 2) == 1`, `linear_search([1,2,3], 5) == -1` |
| 14 | Find the maximum element in a list. | `def find_max(arr):` | `find_max([1,5,3]) == 5`, `find_max([-1,-5,3]) == 3` |
| 15 | Implement a simple sort (bubble sort). | `def bubble_sort(arr):` | `bubble_sort([3,1,2]) == [1,2,3]`, `bubble_sort([]) == []` |
| 16 | Check if a number is even. | `def is_even(n):` | `is_even(4) == True`, `is_even(5) == False` |
| 17 | Compute the sum of list elements. | `def sum_list(lst):` | `sum_list([1,2,3]) == 6`, `sum_list([]) == 0` |
| 18 | Find the factorial iteratively. | `def factorial_iter(n):` | `factorial_iter(5) == 120`, `factorial_iter(0) == 1` |
| 19 | Implement string length without len(). | `def my_len(s):` | `my_len("hello") == 5`, `my_len("") == 0` |
| 20 | Check if a year is a leap year. | `def is_leap_year(y):` | `is_leap_year(2020) == True`, `is_leap_year(2019) == False` |
| 21 | Implement a countdown. | `def countdown(n):` | `list(countdown(3)) == [3,2,1,0]` or similar |
| 22 | Implement a factorial with error checking. | `def safe_factorial(n):` | `safe_factorial(5) == 120`, `safe_factorial(-1) raises ValueError` |
| 23 | Implement a simple cipher (shift by n). | `def caesar_cipher(s, n):` | `caesar_cipher("abc", 1) == "bcd"` (lowercase only) |
| 24 | Implement a Fibonacci with memoization. | `def fib_memo(n, memo={}):` | `fib_memo(0) == 0`, `fib_memo(1) == 1`, `fib_memo(5) == 5` |
| 25 | Implement a function that returns list of even numbers. | `def get_evens(lst):` | `get_evens([1,2,3,4]) == [2,4]` |
| 26 | Implement a function that returns list of odd numbers. | `def get_odds(lst):` | `get_odds([1,2,3,4]) == [1,3]` |
| 27 | Implement a function that filters positive numbers. | `def filter_positives(lst):` | `filter_positives([-1,2,-3,4]) == [2,4]` |
| 28 | Implement a function that maps strings to their lengths. | `def string_lengths(lst):` | `string_lengths(["hi", "hello"]) == [2, 5]` |
| 29 | Implement a function that checks if a list is sorted. | `def is_sorted(lst):` | `is_sorted([1,2,3]) == True`, `is_sorted([3,2,1]) == False` |
| 30 | Implement a function that returns the nth Fibonacci number using recursion. | `def fib_recursive(n):` | `fib_recursive(0) == 0`, `fib_recursive(1) == 1`, `fib_recursive(6) == 8` |
| 31 | Implement a function that removes duplicates from a list. | `def remove_duplicates(lst):` | `remove_duplicates([1,2,2,3]) == [1,2,3]` |
| 32 | Implement a function that finds the intersection of two lists. | `def list_intersection(l1, l2):` | `list_intersection([1,2,3], [2,3,4]) == [2,3]` |
| 33 | Implement a function that finds the union of two lists. | `def list_union(l1, l2):` | `list_union([1,2], [2,3]) == [1,2,3]` |
| 34 | Implement a function that flattens a nested list. | `def flatten(lst):` | `flatten([[1,2],[3,4]]) == [1,2,3,4]` |
| 35 | Implement a function that checks if a value is in a sorted list. | `def binary_search_simple(arr, target):` | `binary_search_simple([1,2,3,4,5], 3) == True` |
| 36 | Implement a function that computes running total. | `def running_total(lst):` | `running_total([1,2,3]) == [1,3,6]` |
| 37 | Implement a function that sorts a list using selection sort. | `def selection_sort(arr):` | `selection_sort([3,1,2]) == [1,2,3]` |
| 38 | Implement a function that implements a simple Caesar cipher with wrap. | `def caesar_wrap(s, n):` | `caesar_wrap("xyz", 2) == "zab"` |
| 39 | Implement a function that checks if two words are the same case-insensitive. | `def same_word(w1, w2):` | `same_word("Hello", "hello") == True` |
| 40 | Implement a function that returns the absolute value. | `def my_abs(x):` | `my_abs(-5) == 5`, `my_abs(3) == 3`, `my_abs(0) == 0` |
| 41 | Implement a function that computes power (base^exp). | `def power(base, exp):` | `power(2, 3) == 8`, `power(5, 0) == 1` |
| 42 | Implement a function that checks if a string contains only digits. | `def is_digits(s):` | `is_digits("123") == True`, `is_digits("abc") == False` |
| 43 | Implement a function that finds the minimum of three numbers. | `def min_of_three(a, b, c):` | `min_of_three(3, 7, 1) == 1` |
| 44 | Implement a function that converts Celsius to Fahrenheit. | `def c_to_f(c):` | `c_to_f(0) == 32`, `c_to_f(100) == 212` |
| 45 | Implement a function that checks if a number is positive. | `def is_positive(n):` | `is_positive(5) == True`, `is_positive(-3) == False`, `is_positive(0) == False` |
| 46 | Implement a function that returns the nth even number. | `def nth_even(n):` | `nth_even(1) == 2`, `nth_even(3) == 6` |
| 47 | Implement a function that computes the modulus. | `def my_mod(a, b):` | `my_mod(7, 3) == 1`, `my_mod(-7, 3) == 2` (Python-style) |
| 48 | Implement a function that checks if a string is blank. | `def is_blank(s):` | `is_blank("") == True`, `is_blank(" ") == True`, `is_blank("x") == False` |
| 49 | Implement a function that returns the square of a number. | `def square(x):` | `square(4) == 16`, `square(-3) == 9` |
| 50 | Implement a function that computes the nth Fibonacci iteratively. | `def fib_iter(n):` | `fib_iter(0) == 0`, `fib_iter(1) == 1`, `fib_iter(7) == 13` |

### JavaScript Tasks (50)

| # | Docstring | Code Stub | Test Cases |
|---|-----------|-----------|------------|
| 1 | Compute the factorial of n recursively. | `function factorial(n) {` | `factorial(5) == 120`, `factorial(0) == 1`, `factorial(1) == 1` |
| 2 | Return the nth Fibonacci number. | `function fibonacci(n) {` | `fibonacci(0) == 0`, `fibonacci(1) == 1`, `fibonacci(10) == 55` |
| 3 | Check whether a string is a palindrome. | `function isPalindrome(s) {` | `isPalindrome("racecar") === true`, `isPalindrome("hello") === false`, `isPalindrome("") === true` |
| 4 | Merge two sorted arrays into one sorted array. | `function mergeSorted(a, b) {` | `mergeSorted([1,3,5], [2,4,6])`.toEqual([1,2,3,4,5,6]) |
| 5 | Binary search on a sorted array. | `function binarySearch(arr, target) {` | `binarySearch([1,2,3,4,5], 3) === 2`, `binarySearch([1,2,3,4,5], 6) === -1` |
| 6 | Return the nth prime number. | `function nthPrime(n) {` | `nthPrime(1) == 2`, `nthPrime(2) == 3`, `nthPrime(3) == 5` |
| 7 | Reverse a string. | `function reverseString(s) {` | `reverseString("hello") === "olleh"`, `reverseString("") === ""` |
| 8 | Check if two strings are anagrams. | `function isAnagram(s1, s2) {` | `isAnagram("listen", "silent") === true`, `isAnagram("hello", "world") === false` |
| 9 | Compute the greatest common divisor. | `function gcd(a, b) {` | `gcd(48, 18) === 6`, `gcd(17, 5) === 1` |
| 10 | Format a date from YYYY-MM-DD to DD/MM/YYYY. | `function formatDate(dateStr) {` | `formatDate("2023-01-15") === "15/01/2023"` |
| 11 | Check if a number is prime. | `function isPrime(n) {` | `isPrime(7) === true`, `isPrime(1) === false`, `isPrime(4) === false` |
| 12 | Calculate the nth triangular number. | `function triangularNumber(n) {` | `triangularNumber(1) === 1`, `triangularNumber(4) === 10`, `triangularNumber(10) === 55` |
| 13 | Implement a linear search. | `function linearSearch(arr, target) {` | `linearSearch([1,2,3], 2) === 1`, `linearSearch([1,2,3], 5) === -1` |
| 14 | Find the maximum element in an array. | `function findMax(arr) {` | `findMax([1,5,3]) === 5`, `findMax([-1,-5,3]) === 3` |
| 15 | Implement a simple sort (bubble sort). | `function bubbleSort(arr) {` | `bubbleSort([3,1,2])`.toEqual([1,2,3]), `bubbleSort([]).toEqual([])` |
| 16 | Check if a number is even. | `function isEven(n) {` | `isEven(4) === true`, `isEven(5) === false` |
| 17 | Compute the sum of array elements. | `function sumArray(arr) {` | `sumArray([1,2,3]) === 6`, `sumArray([]) === 0` |
| 18 | Implement factorial iteratively. | `function factorialIter(n) {` | `factorialIter(5) === 120`, `factorialIter(0) === 1` |
| 19 | Implement string length without length. | `function myLen(s) {` | `myLen("hello") === 5`, `myLen("") === 0` |
| 20 | Check if a year is a leap year. | `function isLeapYear(y) {` | `isLeapYear(2020) === true`, `isLeapYear(2019) === false` |
| 21 | Implement a countdown. | `function countdown(n) {` | Returns array [3,2,1,0] or similar for countdown(3) |
| 22 | Implement factorial with error checking. | `function safeFactorial(n) {` | `safeFactorial(5) === 120`, throws on negative input |
| 23 | Implement a simple cipher (shift by n). | `function caesarCipher(s, n) {` | `caesarCipher("abc", 1) === "bcd"` (lowercase only) |
| 24 | Implement Fibonacci with memoization. | `function fibMemo(n, memo) {` | `fibMemo(0) === 0`, `fibMemo(1) === 1`, `fibMemo(5) === 5` |
| 25 | Implement function that returns list of even numbers. | `function getEvens(arr) {` | `getEvens([1,2,3,4])`.toEqual([2,4]) |
| 26 | Implement function that returns list of odd numbers. | `function getOdds(arr) {` | `getOdds([1,2,3,4])`.toEqual([1,3]) |
| 27 | Implement function that filters positive numbers. | `function filterPositives(arr) {` | `filterPositives([-1,2,-3,4])`.toEqual([2,4]) |
| 28 | Implement function that maps strings to lengths. | `function stringLengths(arr) {` | `stringLengths(["hi","hello"])`.toEqual([2,5]) |
| 29 | Implement function that checks if a list is sorted. | `function isSorted(arr) {` | `isSorted([1,2,3]) === true`, `isSorted([3,2,1]) === false` |
| 30 | Implement Fibonacci using recursion. | `function fibRecursive(n) {` | `fibRecursive(0) === 0`, `fibRecursive(1) === 1`, `fibRecursive(6) === 8` |
| 31 | Implement function that removes duplicates from array. | `function removeDuplicates(arr) {` | `removeDuplicates([1,2,2,3])`.toEqual([1,2,3]) |
| 32 | Implement function that finds intersection of two arrays. | `function listIntersection(arr1, arr2) {` | `listIntersection([1,2,3],[2,3,4])`.toEqual([2,3]) |
| 33 | Implement function that finds union of two arrays. | `function listUnion(arr1, arr2) {` | `listUnion([1,2],[2,3])`.toEqual([1,2,3]) |
| 34 | Implement function that flattens nested array. | `function flatten(arr) {` | `flatten([[1,2],[3,4]])`.toEqual([1,2,3,4]) |
| 35 | Implement function that checks if value is in sorted array. | `function includesSorted(arr, target) {` | `includesSorted([1,2,3,4,5], 3) === true` |
| 36 | Implement function that computes running total. | `function runningTotal(arr) {` | `runningTotal([1,2,3])`.toEqual([1,3,6]) |
| 37 | Implement selection sort. | `function selectionSort(arr) {` | `selectionSort([3,1,2])`.toEqual([1,2,3]) |
| 38 | Implement Caesar cipher with wrap-around. | `function caesarWrap(s, n) {` | `caesarWrap("xyz", 2) === "zab"` |
| 39 | Implement function that checks same word case-insensitive. | `function sameWord(w1, w2) {` | `sameWord("Hello","hello") === true` |
| 40 | Implement function that returns absolute value. | `function myAbs(n) {` | `myAbs(-5) === 5`, `myAbs(3) === 3`, `myAbs(0) === 0` |
| 41 | Implement function that computes power (base^exp). | `function power(base, exp) {` | `power(2, 3) === 8`, `power(5, 0) === 1` |
| 42 | Implement function that checks if string contains only digits. | `function isDigits(s) {` | `isDigits("123") === true`, `isDigits("abc") === false` |
| 43 | Implement function that finds minimum of three numbers. | `function minOfThree(a, b, c) {` | `minOfThree(3, 7, 1) === 1` |
| 44 | Implement function that converts Celsius to Fahrenheit. | `function cToF(c) {` | `cToF(0) === 32`, `cToF(100) === 212` |
| 45 | Implement function that checks if a number is positive. | `function isPositive(n) {` | `isPositive(5) === true`, `isPositive(-3) === false`, `isPositive(0) === false` |
| 46 | Implement function that returns the nth even number. | `function nthEven(n) {` | `nthEven(1) === 2`, `nthEven(3) === 6` |
| 47 | Implement function that computes modulus. | `function myMod(a, b) {` | `myMod(7, 3) === 1`, careful with negative mod in JS |
| 48 | Implement function that checks if string is blank. | `function isBlank(s) {` | `isBlank("") === true`, `isBlank(" ") === true`, `isBlank("x") === false` |
| 49 | Implement function that returns the square of a number. | `function square(n) {` | `square(4) === 16`, `square(-3) === 9` |
| 50 | Implement function that computes the nth Fibonacci iteratively. | `function fibIter(n) {` | `fibIter(0) === 0`, `fibIter(1) === 1`, `fibIter(7) === 13` |

### Prompt Format (for all 100 tasks)

```
<|lang:X|><|docstring|>{docstring}<|code|>{code_stub}<|eos|>
```

Where:
- `X` is `python` or `javascript`
- `docstring` is the task description
- `code_stub` is the function stub (varies by task)
- `<|eos|>` is the EOS token (ID 2)

### Generation Parameters (canonical)

| Parameter | Value |
|---|---|
| temperature | 0 (greedy / deterministic) |
| max_new_tokens | 64 |
| do_sample | False |
| eos_token_id | 2 (<|eos|>) |

### Test Execution (functional correctness, 20 prompts)

**Python**: `python -m py_compile` or `python -c "exec(gen)"` — does it compile/run without error?
**JavaScript**: Node.js `node -e "..."` — does it execute without error?

**Limited to**: 10 Python + 10 JavaScript prompts (the first 10 of each language in the benchmark).

The remaining 80 prompts are evaluated for: syntax validity, repetition metrics, completeness (EOS hit ratio), and compliance with docstring specification.

### Difficulty Categorization

| Difficulty | Criteria |
|---|---|
| Easy | Straightforward implementation, minimal edge cases |
| Medium | Requires some algorithmic thinking, some edge cases |
| Hard | Requires understanding of algorithms, multiple edge cases |

Distribution: 20 Easy + 20 Medium + 10 Hard per language (the 50 split is approximate based on category).

### Node.js Availability Check

Before claiming JavaScript functional evaluation support, check whether Node.js is installed:
- `node --version` returns version string → Node.js available
- Not available → skip functional correctness for JavaScript, report only syntax validity/repetition

Generation must not proceed assuming Node.js availability; the evaluation harness must detect and conditionally enable.