# Phase 11 — Functional Correctness Results

Generated 2026-11-18T02:20:13+00:00 from the frozen Phase 10 report `phase10_generation_evaluation_20261118_031356.json` (checkpoint step 16260, seed 42). No model was loaded, no generation was re-run, and no checkpoint, tokenizer or dataset file was touched.

Execution was performed exclusively inside the verified Docker sandbox (`reports/phase11_sandbox_security.md`: 16/16 containment probes passed). Candidates with invalid syntax were rejected without execution.

## Headline metrics

| metric | Python | JavaScript |
|---|---|---|
| syntax-valid rate | 82.0% | 20.0% |
| functional task success rate | 4.0% | 0.0% |
| test-case pass rate | 9.0% | 2.4% |
| runtime-error rate | 40.0% | 4.0% |
| timeout rate | 0.0% | 0.0% |
| completion rate | 82.0% | 20.0% |
| edge-case success rate | 14.9% | 2.1% |
| functional success (executed candidates only) | 4.9% | 0.0% |
| success excluding convention cases | 4.0% | 0.0% |
| missing-entry rate | 0.0% | 0.0% |
| resource-limit rate | 0.0% | 0.0% |
| syntax-failure rate | 18.0% | 80.0% |

Rates are over all 50 tasks per language; a task that never compiled is a failure for every execution-based metric, never a silent pass.

## Outcome distribution

| outcome | Python | JavaScript |
|---|---|---|
| Syntax failure (not executed) | 9 | 40 |
| Missing function / entry point | 0 | 0 |
| Runtime exception | 20 | 2 |
| Timeout | 0 | 0 |
| Resource-limit violation | 0 | 0 |
| Incorrect output | 19 | 8 |
| Complete task success | 2 | 0 |

## Python per-task results

| task | entry | outcome | cases passed | detail |
|---|---|---|---|---|
| py_01 | `factorial` | runtime_error | 0/4 | NameError on case n=5: name 'np' is not defined |
| py_02 | `fibonacci` | runtime_error | 0/4 | UnboundLocalError on case n=0: cannot access local variable 'nth' where it is not associated with a value |
| py_03 | `is_palindrome` | syntax_failure | 0/5 | candidate did not parse; not executed |
| py_04 | `merge_sorted` | incorrect_output | 1/4 | 3 case(s) wrong, first: interleaved got [1, 3, 5] want [1, 2, 3, 4, 5, 6] |
| py_05 | `binary_search` | runtime_error | 0/5 | AttributeError on case found middle: 'int' object has no attribute 'startswith' |
| py_06 | `nth_prime` | runtime_error | 0/4 | UnboundLocalError on case n=1: cannot access local variable 'nth' where it is not associated with a value |
| py_07 | `reverse_string` | incorrect_output | 2/4 | 2 case(s) wrong, first: hello got 'hello' want 'olleh' |
| py_08 | `is_anagram` | syntax_failure | 0/4 | candidate did not parse; not executed |
| py_09 | `gcd` | syntax_failure | 0/5 | candidate did not parse; not executed |
| py_10 | `format_date` | runtime_error | 0/4 | TypeError on case 2024-01-05: replace() takes at least 2 positional arguments (0 given) |
| py_11 | `is_prime` | incorrect_output | 0/7 | 7 case(s) wrong, first: 2 got None want True |
| py_12 | `triangular_number` | runtime_error | 0/4 | AttributeError on case n=1: 'int' object has no attribute 'shape' |
| py_13 | `linear_search` | incorrect_output | 0/4 | 4 case(s) wrong, first: last got [4, 2, 9] want 2 |
| py_14 | `find_max` | success | 4/4 |  |
| py_15 | `bubble_sort` | runtime_error | 0/5 | AttributeError on case 3,1,2: 'list' object has no attribute 'sort_sort' |
| py_16 | `is_even` | incorrect_output | 1/5 | 4 case(s) wrong, first: 4 got None want True |
| py_17 | `sum_list` | success | 4/4 |  |
| py_18 | `factorial_iter` | incorrect_output | 1/4 | 3 case(s) wrong, first: n=5 got 5 want 120 |
| py_19 | `my_len` | incorrect_output | 0/4 | 4 case(s) wrong, first: hello got 'hello' want 5 |
| py_20 | `is_leap_year` | syntax_failure | 0/5 | candidate did not parse; not executed |
| py_21 | `countdown` | incorrect_output | 0/4 | 4 case(s) wrong, first: 5 got 5 want [5, 4, 3, 2, 1] |
| py_22 | `safe_factorial` | incorrect_output | 0/5 | 5 case(s) wrong, first: n=5 got 5 want 120 |
| py_23 | `caesar_cipher` | syntax_failure | 0/5 | candidate did not parse; not executed |
| py_24 | `fib_memo` | incorrect_output | 0/4 | 4 case(s) wrong, first: n=0 got {} want 0 |
| py_25 | `get_evens` | runtime_error | 0/4 | AttributeError on case 1,2,3,4: 'list' object has no attribute 'filter' |
| py_26 | `get_odds` | runtime_error | 0/4 | AttributeError on case 1,2,3,4: 'list' object has no attribute 'get_odds' |
| py_27 | `filter_positives` | runtime_error | 0/4 | AttributeError on case -2,0,3,5: 'list' object has no attribute 'filter' |
| py_28 | `string_lengths` | syntax_failure | 0/4 | candidate did not parse; not executed |
| py_29 | `is_sorted` | incorrect_output | 0/5 | 5 case(s) wrong, first: ascending got [1, 2, 3] want True |
| py_30 | `fib_recursive` | runtime_error | 0/4 | UnboundLocalError on case n=0: cannot access local variable 'nth' where it is not associated with a value |
| py_31 | `remove_duplicates` | runtime_error | 1/4 | IndexError on case already unique: list index out of range |
| py_32 | `list_intersection` | incorrect_output | 1/4 | 3 case(s) wrong, first: 1,2,3 & 2,3,4 got [2, 3, 4] want [2, 3] |
| py_33 | `list_union` | incorrect_output | 2/4 | 2 case(s) wrong, first: disjoint got [3, 4] want [1, 2, 3, 4] |
| py_34 | `flatten` | runtime_error | 0/4 | AttributeError on case nested: 'list' object has no attribute 'items' |
| py_35 | `binary_search_simple` | incorrect_output | 0/3 | 3 case(s) wrong, first: present got [1, 3, 5] want True |
| py_36 | `running_total` | runtime_error | 0/4 | AttributeError on case 1,2,3: 'list' object has no attribute 'split' |
| py_37 | `selection_sort` | runtime_error | 0/4 | AttributeError on case 64,25,12: 'list' object has no attribute 'sort_sort' |
| py_38 | `caesar_wrap` | runtime_error | 0/5 | AttributeError on case xyz+3: 'list' object has no attribute 'split' |
| py_39 | `same_word` | incorrect_output | 0/4 | 4 case(s) wrong, first: Hello/hello got 'hello' want True |
| py_40 | `my_abs` | syntax_failure | 0/5 | candidate did not parse; not executed |
| py_41 | `power` | incorrect_output | 0/4 | 4 case(s) wrong, first: 2^10 got 10 want 1024 |
| py_42 | `is_digits` | runtime_error | 0/4 | AttributeError on case 123: 'str' object has no attribute 'digits' |
| py_43 | `min_of_three` | incorrect_output | 2/4 | 2 case(s) wrong, first: 5,2,9 got 5 want 2 |
| py_44 | `c_to_f` | runtime_error | 0/4 | AttributeError on case 0C: 'int' object has no attribute 'get_c_from_c' |
| py_45 | `is_positive` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_46 | `nth_even` | runtime_error | 0/3 | UnboundLocalError on case n=1: cannot access local variable 'nth' where it is not associated with a value |
| py_47 | `my_mod` | incorrect_output | 0/3 | 3 case(s) wrong, first: 10%3 got 10 want 1 |
| py_48 | `is_blank` | incorrect_output | 0/4 | 4 case(s) wrong, first: empty got '' want True |
| py_49 | `square` | syntax_failure | 0/4 | candidate did not parse; not executed |
| py_50 | `fib_iter` | runtime_error | 0/4 | UnboundLocalError on case n=0: cannot access local variable 'nth' where it is not associated with a value |

## Javascript per-task results

| task | entry | outcome | cases passed | detail |
|---|---|---|---|---|
| js_01 | `factorial` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_02 | `fibonacci` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_03 | `isPalindrome` | syntax_failure | 0/5 | candidate did not parse; not executed |
| js_04 | `mergeSorted` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_05 | `binarySearch` | incorrect_output | 0/5 | 5 case(s) wrong, first: found middle got [1,3,5,7,9] want 2 |
| js_06 | `nthPrime` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_07 | `reverseString` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_08 | `isAnagram` | runtime_error | 0/4 | RangeError on case listen/silent: Maximum call stack size exceeded |
| js_09 | `gcd` | syntax_failure | 0/5 | candidate did not parse; not executed |
| js_10 | `formatDate` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_11 | `isPrime` | syntax_failure | 0/7 | candidate did not parse; not executed |
| js_12 | `triangularNumber` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_13 | `linearSearch` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_14 | `findMax` | incorrect_output | 0/4 | 4 case(s) wrong, first: 1,5,3 got [1,5,3] want 5 |
| js_15 | `bubbleSort` | runtime_error | 1/5 | TypeError on case 3,1,2: The comparison function must be either a function or undefined: 3 |
| js_16 | `isEven` | syntax_failure | 0/5 | candidate did not parse; not executed |
| js_17 | `sumArray` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_18 | `factorialIter` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_19 | `myLen` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_20 | `isLeapYear` | syntax_failure | 0/5 | candidate did not parse; not executed |
| js_21 | `countdown` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_22 | `safeFactorial` | syntax_failure | 0/5 | candidate did not parse; not executed |
| js_23 | `caesarCipher` | syntax_failure | 0/5 | candidate did not parse; not executed |
| js_24 | `fibMemo` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_25 | `getEvens` | incorrect_output | 0/4 | 4 case(s) wrong, first: 1,2,3,4 got 1 want [2,4] |
| js_26 | `getOdds` | incorrect_output | 0/4 | 4 case(s) wrong, first: 1,2,3,4 got 1 want [1,3] |
| js_27 | `filterPositives` | incorrect_output | 0/4 | 4 case(s) wrong, first: -2,0,3,5 got -2 want [3,5] |
| js_28 | `stringLengths` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_29 | `isSorted` | syntax_failure | 0/5 | candidate did not parse; not executed |
| js_30 | `fibRecursive` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_31 | `removeDuplicates` | incorrect_output | 1/4 | 3 case(s) wrong, first: 1,2,2,3,1 got [] want [1,2,3] |
| js_32 | `listIntersection` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_33 | `listUnion` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_34 | `flatten` | incorrect_output | 1/4 | 3 case(s) wrong, first: nested got [1,[2,[3,4]],5] want [1,2,3,4,5] |
| js_35 | `includesSorted` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_36 | `runningTotal` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_37 | `selectionSort` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_38 | `caesarWrap` | syntax_failure | 0/5 | candidate did not parse; not executed |
| js_39 | `sameWord` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_40 | `myAbs` | syntax_failure | 0/5 | candidate did not parse; not executed |
| js_41 | `power` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_42 | `isDigits` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_43 | `minOfThree` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_44 | `cToF` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_45 | `isPositive` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_46 | `nthEven` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_47 | `myMod` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_48 | `isBlank` | incorrect_output | 2/4 | 2 case(s) wrong, first: empty got false want true |
| js_49 | `square` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_50 | `fibIter` | syntax_failure | 0/4 | candidate did not parse; not executed |

## Limitations

- Behaviour is judged only through the documented specification; style constraints that are not observable (`without len()`, `without %`, recursiveness) are not enforced.
- Convention cases (`-1` miss value, mixed-case Caesar) are included in the primary success rate and reported separately for sensitivity.
- The sandbox shares the Docker Desktop VM kernel, so containment is process/namespace level, not a hardware boundary.
