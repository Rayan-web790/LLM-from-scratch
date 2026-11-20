# Phase 11 — Functional Correctness Results

Generated 2026-11-18T14:10:58+00:00 from the frozen Phase 10 report `phase14_exp_b_generation.json` (checkpoint step 16260, seed 42). No model was loaded, no generation was re-run, and no checkpoint, tokenizer or dataset file was touched.

Execution was performed exclusively inside the verified Docker sandbox (`reports/phase11_sandbox_security.md`: 16/16 containment probes passed). Candidates with invalid syntax were rejected without execution.

## Headline metrics

| metric | Python | JavaScript |
|---|---|---|
| syntax-valid rate | 78.0% | 36.0% |
| functional task success rate | 4.0% | 0.0% |
| test-case pass rate | 11.0% | 1.2% |
| runtime-error rate | 34.0% | 2.0% |
| timeout rate | 0.0% | 0.0% |
| completion rate | 78.0% | 36.0% |
| edge-case success rate | 15.9% | 3.2% |
| functional success (executed candidates only) | 5.1% | 0.0% |
| success excluding convention cases | 4.0% | 0.0% |
| missing-entry rate | 0.0% | 0.0% |
| resource-limit rate | 0.0% | 0.0% |
| syntax-failure rate | 22.0% | 64.0% |

Rates are over all 50 tasks per language; a task that never compiled is a failure for every execution-based metric, never a silent pass.

## Outcome distribution

| outcome | Python | JavaScript |
|---|---|---|
| Syntax failure (not executed) | 11 | 32 |
| Missing function / entry point | 0 | 0 |
| Runtime exception | 17 | 1 |
| Timeout | 0 | 0 |
| Resource-limit violation | 0 | 0 |
| Incorrect output | 20 | 17 |
| Complete task success | 2 | 0 |

## Python per-task results

| task | entry | outcome | cases passed | detail |
|---|---|---|---|---|
| py_v2_01 | `sumSquaredValues` | success | 3/3 |  |
| py_v2_02 | `count_positives` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3 got [1, -2, 3, 0, 1] want 3 |
| py_v2_03 | `count_negatives` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3 got [1, -2, 3, -4, -3] want 3 |
| py_v2_04 | `max_abs` | runtime_error | 0/4 | UnboundLocalError on case basicV3: cannot access local variable 'num_s' where it is not associated with a value |
| py_v2_05 | `min_abs` | runtime_error | 0/4 | UnboundLocalError on case basicV3: cannot access local variable 'num_s' where it is not associated with a value |
| py_v2_06 | `all_positive` | incorrect_output | 2/3 | 1 case(s) wrong, first: allV3 got False want True |
| py_v2_07 | `any_negative` | incorrect_output | 0/3 | 3 case(s) wrong, first: yesV3 got None want True |
| py_v2_08 | `count_zeros` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_09 | `sum_first_n` | incorrect_output | 0/4 | 4 case(s) wrong, first: basicV3 got None want 3 |
| py_v2_10 | `sum_last_n` | incorrect_output | 0/4 | 4 case(s) wrong, first: basicV3 got None want 9 |
| py_v2_11 | `mean_rounded` | runtime_error | 0/3 | TypeError on case basicV3: '<' not supported between instances of 'list' and 'int' |
| py_v2_12 | `median_simple` | runtime_error | 0/3 | NameError on case oddV3: name 'num_of_middles' is not defined |
| py_v2_13 | `range_of` | incorrect_output | 0/4 | 4 case(s) wrong, first: basicV3 got [1, 5, 3, 4] want 4 |
| py_v2_14 | `normalize_minmax` | runtime_error | 0/3 | UnboundLocalError on case basicV3: cannot access local variable 'num_s' where it is not associated with a value |
| py_v2_15 | `reverse_words` | syntax_failure | 0/4 | candidate did not parse; not executed |
| py_v2_16 | `count_words` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_17 | `longest_word` | incorrect_output | 1/3 | 2 case(s) wrong, first: basicV3 got 'a bb ccc!' want 'ccc!' |
| py_v2_18 | `shortest_word` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_19 | `words_sorted` | runtime_error | 0/3 | AttributeError on case basicV3: 'str' object has no attribute 'keys' |
| py_v2_20 | `remove_punctuation` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_21 | `is_substring` | success | 3/3 |  |
| py_v2_22 | `count_substring` | runtime_error | 0/3 | AttributeError on case basicV3: 'str' object has no attribute 'count_length' |
| py_v2_23 | `replaceAllOccurrences` | incorrect_output | 3/4 | 1 case(s) wrong, first: basicV3 got 'aXbXc!' want 'aYbYc!' |
| py_v2_24 | `starts_with_vowel` | incorrect_output | 0/3 | 3 case(s) wrong, first: yesV3 got None want True |
| py_v2_25 | `ends_with_digit` | incorrect_output | 0/3 | 3 case(s) wrong, first: yesV3 got None want False |
| py_v2_26 | `alternating_case` | syntax_failure | 0/4 | candidate did not parse; not executed |
| py_v2_27 | `collapseWhitespaceRuns` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_28 | `truncateText` | incorrect_output | 1/4 | 3 case(s) wrong, first: longV3 got 'abcdef!' want 'abc...' |
| py_v2_29 | `pad_left` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_30 | `checkPrimality` | syntax_failure | 0/4 | candidate did not parse; not executed |
| py_v2_31 | `next_prime` | incorrect_output | 2/4 | 2 case(s) wrong, first: primeV3 got 8 want 11 |
| py_v2_32 | `prime_count_upto` | runtime_error | 0/4 | AttributeError on case basicV3: 'int' object has no attribute 'ndim' |
| py_v2_33 | `gcd_two` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3 got 18 want 1 |
| py_v2_34 | `lcm_two` | incorrect_output | 1/3 | 2 case(s) wrong, first: basicV3 got 6 want 30 |
| py_v2_35 | `iterativeFactorialValue` | incorrect_output | 2/4 | 2 case(s) wrong, first: basicV3 got 6 want 720 |
| py_v2_36 | `raiseToPower` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3 got None want 59049 |
| py_v2_37 | `sqrt_floor` | runtime_error | 0/4 | NameError on case basicV3: name 'np' is not defined |
| py_v2_38 | `abs_diff` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3 got 4 want 3 |
| py_v2_39 | `signOfNumber` | runtime_error | 0/3 | AttributeError on case posV3: 'int' object has no attribute 'split' |
| py_v2_40 | `clamp_list` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3 got [3] want [3, 5, 10, 12, 12] |
| py_v2_41 | `filter_positive` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_42 | `filter_negative` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_43 | `square_list` | runtime_error | 0/3 | TypeError on case basicV3: 'list' object cannot be interpreted as an integer |
| py_v2_44 | `cube_list` | runtime_error | 0/3 | UnboundLocalError on case basicV3: cannot access local variable 'num_elements' where it is not associated with a value |
| py_v2_45 | `double_list` | runtime_error | 0/3 | NameError on case basicV3: name 'num_elements' is not defined |
| py_v2_46 | `negate_list` | runtime_error | 0/3 | UnboundLocalError on case basicV3: cannot access local variable 'num_elements' where it is not associated with a value |
| py_v2_47 | `sort_ascending` | runtime_error | 0/3 | NameError on case basicV3: name 'num_rows' is not defined |
| py_v2_48 | `sort_descending` | runtime_error | 0/3 | NameError on case basicV3: name 'num_descending' is not defined |
| py_v2_49 | `unique_sorted` | runtime_error | 0/3 | NameError on case basicV3: name 'num_unique_values' is not defined |
| py_v2_50 | `product_of_list` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3 got [2, 3, 4, 5] want 120 |

## Javascript per-task results

| task | entry | outcome | cases passed | detail |
|---|---|---|---|---|
| js_v2_01 | `sumSquaredValues` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_02 | `countPositive` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got 6 want 4 |
| js_v2_03 | `countNegative` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got 6 want 4 |
| js_v2_04 | `maxAbsValue` | incorrect_output | 0/4 | 4 case(s) wrong, first: basicV3V3 got [-3,1,2,3,5] want 5 |
| js_v2_05 | `minAbsValue` | incorrect_output | 0/4 | 4 case(s) wrong, first: basicV3V3 got [-3,1,2,3,5] want 1 |
| js_v2_06 | `allPositiveValues` | incorrect_output | 0/3 | 3 case(s) wrong, first: allV3V3 got 1 want true |
| js_v2_07 | `anyNegativeValue` | incorrect_output | 0/3 | 3 case(s) wrong, first: yesV3V3 got 1 want true |
| js_v2_08 | `zeroCount` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got 0 want 2 |
| js_v2_09 | `sumFirstN` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_10 | `sumLastN` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_11 | `roundedMean` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_12 | `simpleMedian` | incorrect_output | 0/3 | 3 case(s) wrong, first: oddV3V3 got [1,2,3,4,6] want 3 |
| js_v2_13 | `valueRange` | incorrect_output | 0/4 | 4 case(s) wrong, first: basicV3V3 got [1,5,3,4,6] want 5 |
| js_v2_14 | `minMaxNormalize` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_15 | `reverseWordOrder` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_16 | `countWordsInText` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_17 | `longestWord` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_18 | `shortestWord` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_19 | `sortedWords` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_20 | `stripPunctuation` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_21 | `isSubstring` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_22 | `countSubstring` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_23 | `replaceAllOccurrences` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_24 | `startsWithVowel` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_25 | `endsWithDigit` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_26 | `alternateCase` | incorrect_output | 1/4 | 3 case(s) wrong, first: basicV3V3 got "!?!" want "HeLlO!?!" |
| js_v2_27 | `collapseWhitespaceRuns` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_28 | `truncateText` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_29 | `padStringLeft` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_30 | `isPrimeNumber` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_31 | `nextPrimeNumber` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_32 | `primeCount` | runtime_error | 0/4 | ReferenceError on case basicV3V3: BigInteger is not defined |
| js_v2_33 | `gcdPair` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_34 | `lcmPair` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_35 | `iterativeFactorial` | incorrect_output | 0/4 | 4 case(s) wrong, first: basicV3V3 got [0,1,2,3,4,5,6,7] want 40320 |
| js_v2_36 | `powerOf` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_37 | `floorSqrt` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_38 | `absoluteDifference` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_39 | `signOf` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_40 | `clampArray` | incorrect_output | 1/3 | 2 case(s) wrong, first: basicV3V3 got [false,false,false,false,false,false] want [3,5,10,12,12,12] |
| js_v2_41 | `positiveFilter` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_42 | `negativeFilter` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got -1 want [-1,-3] |
| js_v2_43 | `squareArray` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got 2 want [1,4,9,16,36] |
| js_v2_44 | `cubeArray` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got [] want [1,8,27,64,216] |
| js_v2_45 | `doubleArray` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got 1 want [2,4,6,8,12] |
| js_v2_46 | `negateArray` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_47 | `ascendingSort` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_48 | `descendingSort` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_49 | `uniqueSorted` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_50 | `productList` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got [2,3,4,5,7] want 840 |

## Limitations

- Behaviour is judged only through the documented specification; style constraints that are not observable (`without len()`, `without %`, recursiveness) are not enforced.
- Convention cases (`-1` miss value, mixed-case Caesar) are included in the primary success rate and reported separately for sensitivity.
- The sandbox shares the Docker Desktop VM kernel, so containment is process/namespace level, not a hardware boundary.
