# Phase 11 — Functional Correctness Results

Generated 2026-11-18T14:12:57+00:00 from the frozen Phase 10 report `phase14_exp_c_generation.json` (checkpoint step 16260, seed 42). No model was loaded, no generation was re-run, and no checkpoint, tokenizer or dataset file was touched.

Execution was performed exclusively inside the verified Docker sandbox (`reports/phase11_sandbox_security.md`: 16/16 containment probes passed). Candidates with invalid syntax were rejected without execution.

## Headline metrics

| metric | Python | JavaScript |
|---|---|---|
| syntax-valid rate | 80.0% | 46.0% |
| functional task success rate | 6.0% | 2.0% |
| test-case pass rate | 16.5% | 7.3% |
| runtime-error rate | 38.0% | 0.0% |
| timeout rate | 0.0% | 0.0% |
| completion rate | 80.0% | 46.0% |
| edge-case success rate | 22.2% | 11.1% |
| functional success (executed candidates only) | 7.5% | 4.3% |
| success excluding convention cases | 6.0% | 2.0% |
| missing-entry rate | 0.0% | 0.0% |
| resource-limit rate | 0.0% | 0.0% |
| syntax-failure rate | 20.0% | 54.0% |

Rates are over all 50 tasks per language; a task that never compiled is a failure for every execution-based metric, never a silent pass.

## Outcome distribution

| outcome | Python | JavaScript |
|---|---|---|
| Syntax failure (not executed) | 10 | 27 |
| Missing function / entry point | 0 | 0 |
| Runtime exception | 19 | 0 |
| Timeout | 0 | 0 |
| Resource-limit violation | 0 | 0 |
| Incorrect output | 18 | 22 |
| Complete task success | 3 | 1 |

## Python per-task results

| task | entry | outcome | cases passed | detail |
|---|---|---|---|---|
| py_v2_01 | `sumSquaredValues` | success | 3/3 |  |
| py_v2_02 | `count_positives` | incorrect_output | 2/3 | 1 case(s) wrong, first: basicV3 got 0 want 3 |
| py_v2_03 | `count_negatives` | incorrect_output | 2/3 | 1 case(s) wrong, first: basicV3 got 0 want 3 |
| py_v2_04 | `max_abs` | incorrect_output | 0/4 | 4 case(s) wrong, first: basicV3 got [-3, 1, 2, 3] want 3 |
| py_v2_05 | `min_abs` | incorrect_output | 0/4 | 4 case(s) wrong, first: basicV3 got [-3, 1, 2, 3] want 1 |
| py_v2_06 | `all_positive` | incorrect_output | 2/3 | 1 case(s) wrong, first: allV3 got False want True |
| py_v2_07 | `any_negative` | incorrect_output | 1/3 | 2 case(s) wrong, first: noV3 got True want False |
| py_v2_08 | `count_zeros` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_09 | `sum_first_n` | runtime_error | 0/4 | NameError on case basicV3: name 'num' is not defined |
| py_v2_10 | `sum_last_n` | runtime_error | 0/4 | UnboundLocalError on case basicV3: cannot access local variable 'num_elements' where it is not associated with a value |
| py_v2_11 | `mean_rounded` | runtime_error | 0/3 | TypeError on case basicV3: '<' not supported between instances of 'list' and 'int' |
| py_v2_12 | `median_simple` | runtime_error | 0/3 | NameError on case oddV3: name 'np' is not defined |
| py_v2_13 | `range_of` | runtime_error | 0/4 | NameError on case basicV3: name 'num_s' is not defined |
| py_v2_14 | `normalize_minmax` | incorrect_output | 1/3 | 2 case(s) wrong, first: basicV3 got [2] want [0.0, 0.3333333333333333, 0.6666666666666666, 1.0] |
| py_v2_15 | `reverse_words` | syntax_failure | 0/4 | candidate did not parse; not executed |
| py_v2_16 | `count_words` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_17 | `longest_word` | incorrect_output | 1/3 | 2 case(s) wrong, first: basicV3 got 'a bb ccc!' want 'ccc!' |
| py_v2_18 | `shortest_word` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_19 | `words_sorted` | runtime_error | 0/3 | AttributeError on case basicV3: 'str' object has no attribute 'sort' |
| py_v2_20 | `remove_punctuation` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_21 | `is_substring` | success | 3/3 |  |
| py_v2_22 | `count_substring` | incorrect_output | 2/3 | 1 case(s) wrong, first: basicV3 got 0 want 3 |
| py_v2_23 | `replaceAllOccurrences` | incorrect_output | 0/4 | 4 case(s) wrong, first: basicV3 got [0, 1, 2, 3, 4, 5] want 'aYbYc!' |
| py_v2_24 | `starts_with_vowel` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_25 | `ends_with_digit` | success | 3/3 |  |
| py_v2_26 | `alternating_case` | runtime_error | 0/4 | AttributeError on case basicV3: 'str' object has no attribute 'lowercase' |
| py_v2_27 | `collapseWhitespaceRuns` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_28 | `truncateText` | incorrect_output | 1/4 | 3 case(s) wrong, first: longV3 got 'abcdef!' want 'abc...' |
| py_v2_29 | `pad_left` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_30 | `checkPrimality` | incorrect_output | 2/4 | 2 case(s) wrong, first: twoV3 got False want True |
| py_v2_31 | `next_prime` | incorrect_output | 0/4 | 4 case(s) wrong, first: basicV3 got 0 want 11 |
| py_v2_32 | `prime_count_upto` | runtime_error | 0/4 | AttributeError on case basicV3: 'int' object has no attribute 'issub' |
| py_v2_33 | `gcd_two` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_34 | `lcm_two` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3 got 0 want 30 |
| py_v2_35 | `iterativeFactorialValue` | incorrect_output | 2/4 | 2 case(s) wrong, first: basicV3 got 6 want 720 |
| py_v2_36 | `raiseToPower` | runtime_error | 0/3 | ValueError on case zero expV3: invalid exp |
| py_v2_37 | `sqrt_floor` | runtime_error | 0/4 | NameError on case basicV3: name 'np' is not defined |
| py_v2_38 | `abs_diff` | runtime_error | 0/3 | NameError on case basicV3: name 'np' is not defined |
| py_v2_39 | `signOfNumber` | runtime_error | 0/3 | AttributeError on case posV3: 'int' object has no attribute 'lower' |
| py_v2_40 | `clamp_list` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3 got [3, 12] want [3, 5, 10, 12, 12] |
| py_v2_41 | `filter_positive` | runtime_error | 0/3 | NameError on case basicV3: name 'num_elements' is not defined |
| py_v2_42 | `filter_negative` | runtime_error | 0/3 | UnboundLocalError on case basicV3: cannot access local variable 'num_elements' where it is not associated with a value |
| py_v2_43 | `square_list` | runtime_error | 0/3 | TypeError on case basicV3: 'list' object cannot be interpreted as an integer |
| py_v2_44 | `cube_list` | runtime_error | 0/3 | UnboundLocalError on case basicV3: cannot access local variable 'num_classes' where it is not associated with a value |
| py_v2_45 | `double_list` | incorrect_output | 1/3 | 2 case(s) wrong, first: basicV3 got [2] want [2, 4, 6, 8] |
| py_v2_46 | `negate_list` | syntax_failure | 0/3 | candidate did not parse; not executed |
| py_v2_47 | `sort_ascending` | runtime_error | 0/3 | NameError on case basicV3: name 'num_nums' is not defined |
| py_v2_48 | `sort_descending` | runtime_error | 0/3 | NameError on case basicV3: name 'num_count' is not defined |
| py_v2_49 | `unique_sorted` | incorrect_output | 1/3 | 2 case(s) wrong, first: basicV3 got [1] want [1, 2, 3] |
| py_v2_50 | `product_of_list` | runtime_error | 0/3 | NameError on case basicV3: name 'num_elements' is not defined |

## Javascript per-task results

| task | entry | outcome | cases passed | detail |
|---|---|---|---|---|
| js_v2_01 | `sumSquaredValues` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got 16 want 66 |
| js_v2_02 | `countPositive` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got false want 4 |
| js_v2_03 | `countNegative` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got false want 4 |
| js_v2_04 | `maxAbsValue` | incorrect_output | 0/4 | 4 case(s) wrong, first: basicV3V3 got null want 5 |
| js_v2_05 | `minAbsValue` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_06 | `allPositiveValues` | incorrect_output | 2/3 | 1 case(s) wrong, first: allV3V3 got false want true |
| js_v2_07 | `anyNegativeValue` | incorrect_output | 2/3 | 1 case(s) wrong, first: yesV3V3 got false want true |
| js_v2_08 | `zeroCount` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got false want 2 |
| js_v2_09 | `sumFirstN` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_10 | `sumLastN` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_11 | `roundedMean` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got [1,2,3,4,6] want 3.2 |
| js_v2_12 | `simpleMedian` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_13 | `valueRange` | incorrect_output | 0/4 | 4 case(s) wrong, first: basicV3V3 got [] want 5 |
| js_v2_14 | `minMaxNormalize` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_15 | `reverseWordOrder` | incorrect_output | 0/4 | 4 case(s) wrong, first: basicV3V3 got ["h","e","l","l","o"," ","w","o","r","l","d","!","?","!"] want "world!?! hello" |
| js_v2_16 | `countWordsInText` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_17 | `longestWord` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_18 | `shortestWord` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_19 | `sortedWords` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got ["b","a","n","a","n","a"," ","a","p","p","l","e"," ","c","h","e","r","r","y","!","?","!"] want ["apple","banana" |
| js_v2_20 | `stripPunctuation` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_21 | `isSubstring` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_22 | `countSubstring` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_23 | `replaceAllOccurrences` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_24 | `startsWithVowel` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_25 | `endsWithDigit` | success | 3/3 |  |
| js_v2_26 | `alternateCase` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_27 | `collapseWhitespaceRuns` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_28 | `truncateText` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_29 | `padStringLeft` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_30 | `isPrimeNumber` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_31 | `nextPrimeNumber` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_32 | `primeCount` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_33 | `gcdPair` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_34 | `lcmPair` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_35 | `iterativeFactorial` | syntax_failure | 0/4 | candidate did not parse; not executed |
| js_v2_36 | `powerOf` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_37 | `floorSqrt` | incorrect_output | 0/4 | 4 case(s) wrong, first: basicV3V3 got 3.605551275463989 want 3 |
| js_v2_38 | `absoluteDifference` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_39 | `signOf` | syntax_failure | 0/3 | candidate did not parse; not executed |
| js_v2_40 | `clampArray` | incorrect_output | 1/3 | 2 case(s) wrong, first: basicV3V3 got [1,5,10,15,16,18] want [3,5,10,12,12,12] |
| js_v2_41 | `positiveFilter` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got 0 want [2,4,5,7] |
| js_v2_42 | `negativeFilter` | incorrect_output | 2/3 | 1 case(s) wrong, first: basicV3V3 got [] want [-1,-3] |
| js_v2_43 | `squareArray` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got [1,2,3,4,6] want [1,4,9,16,36] |
| js_v2_44 | `cubeArray` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got [1,2,3,4,6] want [1,8,27,64,216] |
| js_v2_45 | `doubleArray` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got [] want [2,4,6,8,12] |
| js_v2_46 | `negateArray` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got [] want [-1,2,-3,-4,-6] |
| js_v2_47 | `ascendingSort` | incorrect_output | 1/3 | 2 case(s) wrong, first: basicV3V3 got [3,1,2,3,5] want [1,2,3,3,5] |
| js_v2_48 | `descendingSort` | incorrect_output | 0/3 | 3 case(s) wrong, first: basicV3V3 got 1 want [5,3,3,2,1] |
| js_v2_49 | `uniqueSorted` | incorrect_output | 1/3 | 2 case(s) wrong, first: basicV3V3 got [3,1,2,1,2,4] want [1,2,3,4] |
| js_v2_50 | `productList` | syntax_failure | 0/3 | candidate did not parse; not executed |

## Limitations

- Behaviour is judged only through the documented specification; style constraints that are not observable (`without len()`, `without %`, recursiveness) are not enforced.
- Convention cases (`-1` miss value, mixed-case Caesar) are included in the primary success rate and reported separately for sensitivity.
- The sandbox shares the Docker Desktop VM kernel, so containment is process/namespace level, not a hardware boundary.
