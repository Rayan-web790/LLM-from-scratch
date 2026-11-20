# Phase 11 — Functional Correctness Results

Generated 2026-11-18T02:19:13+00:00 from the frozen Phase 10 report `phase10_generation_evaluation_20261118_031316.json` (checkpoint step 16260, seed 42). No model was loaded, no generation was re-run, and no checkpoint, tokenizer or dataset file was touched.

Execution was performed exclusively inside the verified Docker sandbox (`reports/phase11_sandbox_security.md`: 16/16 containment probes passed). Candidates with invalid syntax were rejected without execution.

## Headline metrics

| metric | Python | JavaScript |
|---|---|---|
| syntax-valid rate | 84.0% | 22.0% |
| functional task success rate | 0.0% | 0.0% |
| test-case pass rate | 3.3% | 4.3% |
| runtime-error rate | 50.0% | 0.0% |
| timeout rate | 0.0% | 0.0% |
| completion rate | 84.0% | 22.0% |
| edge-case success rate | 7.4% | 6.4% |
| functional success (executed candidates only) | 0.0% | 0.0% |
| success excluding convention cases | 0.0% | 0.0% |
| missing-entry rate | 0.0% | 0.0% |
| resource-limit rate | 0.0% | 0.0% |
| syntax-failure rate | 16.0% | 78.0% |

Rates are over all 50 tasks per language; a task that never compiled is a failure for every execution-based metric, never a silent pass.

## Outcome distribution

| outcome | Python | JavaScript |
|---|---|---|
| Syntax failure (not executed) | 8 | 39 |
| Missing function / entry point | 0 | 0 |
| Runtime exception | 25 | 0 |
| Timeout | 0 | 0 |
| Resource-limit violation | 0 | 0 |
| Incorrect output | 17 | 11 |
| Complete task success | 0 | 0 |

## Python per-task results

| task | entry | outcome | cases passed | detail |
|---|---|---|---|---|
| py_01 | `factorial` | runtime_error | 0/4 | NameError on case n=5: name 'np' is not defined |
| py_02 | `fibonacci` | runtime_error | 0/4 | UnboundLocalError on case n=0: cannot access local variable 'nth' where it is not associated with a value |
| py_03 | `is_palindrome` | syntax_failure | 0/5 | candidate did not parse; not executed |
| py_04 | `merge_sorted` | runtime_error | 0/4 | TypeError on case interleaved: 'int' object is not subscriptable |
| py_05 | `binary_search` | runtime_error | 0/5 | AttributeError on case found middle: 'int' object has no attribute 'startswith' |
| py_06 | `nth_prime` | runtime_error | 0/4 | UnboundLocalError on case n=1: cannot access local variable 'nth_prime' where it is not associated with a value |
| py_07 | `reverse_string` | incorrect_output | 1/4 | 3 case(s) wrong, first: hello got '' want 'olleh' |
| py_08 | `is_anagram` | syntax_failure | 0/4 | candidate did not parse; not executed |
| py_09 | `gcd` | runtime_error | 0/5 | UnboundLocalError on case 12,18: cannot access local variable 'gcd' where it is not associated with a value |
| py_10 | `format_date` | runtime_error | 0/4 | NameError on case 2024-01-05: name 'UTC' is not defined |
| py_11 | `is_prime` | incorrect_output | 1/7 | 6 case(s) wrong, first: 2 got None want True |
| py_12 | `triangular_number` | syntax_failure | 0/4 | candidate did not parse; not executed |
| py_13 | `linear_search` | runtime_error | 0/4 | AttributeError on case last: 'int' object has no attribute 'shape' |
| py_14 | `find_max` | runtime_error | 0/4 | AttributeError on case 1,5,3: 'list' object has no attribute 'max' |
| py_15 | `bubble_sort` | incorrect_output | 0/5 | 5 case(s) wrong, first: 3,1,2 got None want [1, 2, 3] |
| py_16 | `is_even` | incorrect_output | 1/5 | 4 case(s) wrong, first: 4 got None want True |
| py_17 | `sum_list` | incorrect_output | 0/4 | 4 case(s) wrong, first: 1,2,3 got [] want 6 |
| py_18 | `factorial_iter` | runtime_error | 0/4 | AttributeError on case n=5: 'int' object has no attribute 'shape' |
| py_19 | `my_len` | runtime_error | 0/4 | IndexError on case empty: string index out of range |
| py_20 | `is_leap_year` | syntax_failure | 0/5 | candidate did not parse; not executed |
| py_21 | `countdown` | runtime_error | 0/4 | AttributeError on case 5: 'int' object has no attribute 'countdown' |
| py_22 | `safe_factorial` | incorrect_output | 0/5 | 5 case(s) wrong, first: n=5 got 5 want 120 |
| py_23 | `caesar_cipher` | incorrect_output | 0/5 | 5 case(s) wrong, first: abc+1 got 0 want 'bcd' |
| py_24 | `fib_memo` | incorrect_output | 0/4 | 4 case(s) wrong, first: n=0 got {} want 0 |
| py_25 | `get_evens` | incorrect_output | 0/4 | 4 case(s) wrong, first: 1,2,3,4 got [[1, 2, 3, 4]] want [2, 4] |
| py_26 | `get_odds` | runtime_error | 1/4 | TypeError on case 1,2,3,4: 'int' object is not subscriptable |
| py_27 | `filter_positives` | syntax_failure | 0/4 | candidate did not parse; not executed |
| py_28 | `string_lengths` | runtime_error | 0/4 | AttributeError on case a,bb,ccc: 'list' object has no attribute 'shape' |
| py_29 | `is_sorted` | syntax_failure | 0/5 | candidate did not parse; not executed |
| py_30 | `fib_recursive` | runtime_error | 0/4 | UnboundLocalError on case n=0: cannot access local variable 'nth' where it is not associated with a value |
| py_31 | `remove_duplicates` | incorrect_output | 0/4 | 4 case(s) wrong, first: 1,2,2,3,1 got None want [1, 2, 3] |
| py_32 | `list_intersection` | runtime_error | 0/4 | AttributeError on case 1,2,3 & 2,3,4: 'list' object has no attribute 'intersection' |
| py_33 | `list_union` | runtime_error | 0/4 | AttributeError on case disjoint: 'list' object has no attribute 'intersection' |
| py_34 | `flatten` | incorrect_output | 0/4 | 4 case(s) wrong, first: nested got [] want [1, 2, 3, 4, 5] |
| py_35 | `binary_search_simple` | runtime_error | 0/3 | AttributeError on case present: 'int' object has no attribute 'startswith' |
| py_36 | `running_total` | incorrect_output | 0/4 | 4 case(s) wrong, first: 1,2,3 got None want [1, 3, 6] |
| py_37 | `selection_sort` | incorrect_output | 2/4 | 2 case(s) wrong, first: 64,25,12 got [64, 25, 12] want [12, 25, 64] |
| py_38 | `caesar_wrap` | incorrect_output | 0/5 | 5 case(s) wrong, first: xyz+3 got ['xyz'] want 'abc' |
| py_39 | `same_word` | syntax_failure | 0/4 | candidate did not parse; not executed |
| py_40 | `my_abs` | runtime_error | 0/5 | AttributeError on case -5: 'int' object has no attribute 'startswith' |
| py_41 | `power` | runtime_error | 0/4 | RecursionError on case 2^10: maximum recursion depth exceeded |
| py_42 | `is_digits` | syntax_failure | 0/4 | candidate did not parse; not executed |
| py_43 | `min_of_three` | incorrect_output | 0/4 | 4 case(s) wrong, first: 1,2,3 got 0 want 1 |
| py_44 | `c_to_f` | runtime_error | 0/4 | AttributeError on case 0C: 'int' object has no attribute 'get_f' |
| py_45 | `is_positive` | incorrect_output | 0/3 | 3 case(s) wrong, first: 5 got 5 want True |
| py_46 | `nth_even` | runtime_error | 0/3 | UnboundLocalError on case n=1: cannot access local variable 'nth' where it is not associated with a value |
| py_47 | `my_mod` | runtime_error | 0/3 | AttributeError on case 10%3: 'int' object has no attribute 'modulus' |
| py_48 | `is_blank` | runtime_error | 0/4 | AttributeError on case empty: 'str' object has no attribute 'is_blank' |
| py_49 | `square` | incorrect_output | 1/4 | 3 case(s) wrong, first: 4 got 1125899906842624 want 16 |
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
| js_07 | `reverseString` | incorrect_output | 2/4 | 2 case(s) wrong, first: hello got "hello" want "olleh" |
| js_08 | `isAnagram` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_09 | `gcd` | syntax_failure | 0/5 | candidate did not parse; not executed |
| js_10 | `formatDate` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_11 | `isPrime` | syntax_failure | 0/7 | candidate did not parse; not executed |
| js_12 | `triangularNumber` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_13 | `linearSearch` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_14 | `findMax` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_15 | `bubbleSort` | syntax_failure | 0/5 | candidate did not parse; not executed |
| js_16 | `isEven` | syntax_failure | 0/5 | candidate did not parse; not executed |
| js_17 | `sumArray` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_18 | `factorialIter` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_19 | `myLen` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_20 | `isLeapYear` | syntax_failure | 0/5 | candidate did not parse; not executed |
| js_21 | `countdown` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_22 | `safeFactorial` | incorrect_output | 0/5 | 5 case(s) wrong, first: n=5 got [1,4,3,1,2] want 120 |
| js_23 | `caesarCipher` | syntax_failure | 0/5 | candidate did not parse; not executed |
| js_24 | `fibMemo` | incorrect_output | 0/4 | 4 case(s) wrong, first: n=0 got {} want 0 |
| js_25 | `getEvens` | incorrect_output | 2/4 | 2 case(s) wrong, first: 1,2,3,4 got [1,2,3,4] want [2,4] |
| js_26 | `getOdds` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_27 | `filterPositives` | incorrect_output | 2/4 | 2 case(s) wrong, first: -2,0,3,5 got [-2,0,3,5] want [3,5] |
| js_28 | `stringLengths` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_29 | `isSorted` | syntax_failure | 0/5 | candidate did not parse; not executed |
| js_30 | `fibRecursive` | incorrect_output | 0/4 | 4 case(s) wrong, first: n=0 got [] want 0 |
| js_31 | `removeDuplicates` | incorrect_output | 1/4 | 3 case(s) wrong, first: 1,2,2,3,1 got [] want [1,2,3] |
| js_32 | `listIntersection` | incorrect_output | 1/4 | 3 case(s) wrong, first: 1,2,3 & 2,3,4 got [1,2,3] want [2,3] |
| js_33 | `listUnion` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_34 | `flatten` | incorrect_output | 1/4 | 3 case(s) wrong, first: nested got [1,[2,[3,4]],5] want [1,2,3,4,5] |
| js_35 | `includesSorted` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_36 | `runningTotal` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_37 | `selectionSort` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_38 | `caesarWrap` | syntax_failure | 0/5 | candidate did not parse; not executed |
| js_39 | `sameWord` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_40 | `myAbs` | incorrect_output | 0/5 | 5 case(s) wrong, first: -5 got [] want 5 |
| js_41 | `power` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_42 | `isDigits` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_43 | `minOfThree` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_44 | `cToF` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_45 | `isPositive` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_46 | `nthEven` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_47 | `myMod` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_48 | `isBlank` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_49 | `square` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_50 | `fibIter` | syntax_failure | 0/4 | candidate did not parse; not executed |

## Limitations

- Behaviour is judged only through the documented specification; style constraints that are not observable (`without len()`, `without %`, recursiveness) are not enforced.
- Convention cases (`-1` miss value, mixed-case Caesar) are included in the primary success rate and reported separately for sensitivity.
- The sandbox shares the Docker Desktop VM kernel, so containment is process/namespace level, not a hardware boundary.
