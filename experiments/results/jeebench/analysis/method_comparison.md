# Phase 3 method comparison (consistent failures, JEEBench (math))

Phase-3 version `jee_extra_v1_2026-11-02`. Each table uses only the failures on which every listed method has a result. *Wins / losses* = failures this method solved and the baseline did not / the reverse (McNemar exact p). Tokens = mean solver output tokens per failure (judge excluded).

## qwen/qwen-2.5-7b-instruct (176 of 176 failures)

| Method | recovered | tokens | vs Retry x3 (compute-matched) | vs Retry x3 + checks + guard |
|---|---|---|---|---|
| Retry (1 try, v4 control) | 25/176 (14.2%) | 1116 | 19 / 16 (p=0.736) | 15 / 15 (p=1.000) |
| ESV v4 | 25/176 (14.2%) | 3834 | 21 / 18 (p=0.749) | 20 / 20 (p=1.000) |
| Retry x3 (compute-matched) | 22/176 (12.5%) | 2882 |  | 15 / 18 (p=0.728) |
| Retry x3 + checks + guard | 25/176 (14.2%) | 3899 | 18 / 15 (p=0.728) |  |
| ESV v2, 1 strategy | 17/176 (9.7%) | 1913 | 13 / 18 (p=0.473) | 12 / 20 (p=0.215) |
| ESV v2 | 28/176 (15.9%) | 4798 | 24 / 18 (p=0.441) | 16 / 13 (p=0.711) |

## mistralai/mistral-nemo (258 of 258 failures)

| Method | recovered | tokens | vs Retry x3 (compute-matched) | vs Retry x3 + checks + guard |
|---|---|---|---|---|
| Retry (1 try, v4 control) | 20/258 (7.8%) | 859 | 18 / 14 (p=0.597) | 16 / 21 (p=0.511) |
| ESV v4 | 23/258 (8.9%) | 3202 | 21 / 14 (p=0.311) | 17 / 19 (p=0.868) |
| Retry x3 (compute-matched) | 16/258 (6.2%) | 2056 |  | 12 / 21 (p=0.163) |
| Retry x3 + checks + guard | 25/258 (9.7%) | 2455 | 21 / 12 (p=0.163) |  |
| ESV v2, 1 strategy | 17/258 (6.6%) | 1527 | 13 / 12 (p=1.000) | 13 / 21 (p=0.229) |
| ESV v2 | 17/258 (6.6%) | 3775 | 13 / 12 (p=1.000) | 11 / 19 (p=0.200) |

## google/gemma-3-4b-it (81 of 191 failures)

| Method | recovered | tokens | vs Retry x3 (compute-matched) | vs Retry x3 + checks + guard |
|---|---|---|---|---|
| Retry (1 try, v4 control) | 5/81 (6.2%) | 2227 | 5 / 7 (p=0.774) | 4 / 4 (p=1.000) |
| ESV v4 | 6/81 (7.4%) | 7769 | 4 / 5 (p=1.000) | 5 / 4 (p=1.000) |
| Retry x3 (compute-matched) | 7/81 (8.6%) | 5717 |  | 5 / 3 (p=0.727) |
| Retry x3 + checks + guard | 5/81 (6.2%) | 6544 | 3 / 5 (p=0.727) |  |
| ESV v2, 1 strategy | 7/81 (8.6%) | 3308 | 5 / 5 (p=1.000) | 5 / 3 (p=0.727) |
| ESV v2 | 8/81 (9.9%) | 8376 | 7 / 6 (p=1.000) | 6 / 3 (p=0.508) |

## qwen/qwen-2.5-72b-instruct (0 of 0 failures)

| Method | recovered | tokens | vs Retry x3 (compute-matched) | vs Retry x3 + checks + guard |
|---|---|---|---|---|
| Retry (1 try, v4 control) | - | 0 | 0 / 0 | 0 / 0 |
| ESV v4 | - | 0 | 0 / 0 | 0 / 0 |
| Retry x3 (compute-matched) | - | 0 |  | 0 / 0 |
| Retry x3 + checks + guard | - | 0 | 0 / 0 |  |
| ESV v2, 1 strategy | - | 0 | 0 / 0 | 0 / 0 |
| ESV v2 | - | 0 | 0 / 0 | 0 / 0 |

## mistralai/mistral-small-3.2-24b-instruct (0 of 0 failures)

| Method | recovered | tokens | vs Retry x3 (compute-matched) | vs Retry x3 + checks + guard |
|---|---|---|---|---|
| Retry (1 try, v4 control) | - | 0 | 0 / 0 | 0 / 0 |
| ESV v4 | - | 0 | 0 / 0 | 0 / 0 |
| Retry x3 (compute-matched) | - | 0 |  | 0 / 0 |
| Retry x3 + checks + guard | - | 0 | 0 / 0 |  |
| ESV v2, 1 strategy | - | 0 | 0 / 0 | 0 / 0 |
| ESV v2 | - | 0 | 0 / 0 | 0 / 0 |

## meta-llama/llama-3.3-70b-instruct (0 of 0 failures)

| Method | recovered | tokens | vs Retry x3 (compute-matched) | vs Retry x3 + checks + guard |
|---|---|---|---|---|
| Retry (1 try, v4 control) | - | 0 | 0 / 0 | 0 / 0 |
| ESV v4 | - | 0 | 0 / 0 | 0 / 0 |
| Retry x3 (compute-matched) | - | 0 |  | 0 / 0 |
| Retry x3 + checks + guard | - | 0 | 0 / 0 |  |
| ESV v2, 1 strategy | - | 0 | 0 / 0 | 0 / 0 |
| ESV v2 | - | 0 | 0 / 0 | 0 / 0 |

## Pooled (515 failures)

| Method | recovered | tokens | vs Retry x3 (compute-matched) | vs Retry x3 + checks + guard |
|---|---|---|---|---|
| Retry (1 try, v4 control) | 50/515 (9.7%) | 1162 | 42 / 37 (p=0.653) | 35 / 40 (p=0.644) |
| ESV v4 | 54/515 (10.5%) | 4137 | 46 / 37 (p=0.380) | 42 / 43 (p=1.000) |
| Retry x3 (compute-matched) | 45/515 (8.7%) | 2914 |  | 32 / 42 (p=0.295) |
| Retry x3 + checks + guard | 55/515 (10.7%) | 3592 | 42 / 32 (p=0.295) |  |
| ESV v2, 1 strategy | 41/515 (8.0%) | 1939 | 31 / 35 (p=0.712) | 30 / 44 (p=0.130) |
| ESV v2 | 53/515 (10.3%) | 4848 | 44 / 36 (p=0.434) | 33 / 35 (p=0.904) |

### Pooled, by answer format (recovered counts)

| answer format | n | Retry (1 try, v4 control) | ESV v4 | Retry x3 (compute-matched) | Retry x3 + checks + guard | ESV v2, 1 strategy | ESV v2 |
|---|---|---|---|---|---|---|---|
| integer | 72 | 9 | 3 | 2 | 6 | 4 | 8 |
| multiple | 222 | 24 | 28 | 17 | 22 | 17 | 25 |
| numeric | 160 | 10 | 12 | 12 | 7 | 7 | 5 |
| single | 61 | 7 | 11 | 14 | 20 | 13 | 15 |

### Pooled, by arm (recovered counts)

| arm | n | Retry (1 try, v4 control) | ESV v4 | Retry x3 (compute-matched) | Retry x3 + checks + guard | ESV v2, 1 strategy | ESV v2 |
|---|---|---|---|---|---|---|---|
| original | 254 | 24 | 28 | 20 | 24 | 16 | 23 |
| perturbed-breaking | 139 | 15 | 12 | 16 | 16 | 11 | 17 |
| perturbed-preserving | 122 | 11 | 14 | 9 | 15 | 14 | 13 |

### ESV v2: stages (pooled)

- recognition said a failure occurred: 306/515 (59.4%)
- diagnosis verdicts: {'CORRECT': 202, 'PARTIAL': 138, 'INCORRECT': 175}
- strategies kept per run: {1: 16, 2: 20, 3: 479}; regenerated for similarity: 74
- strategies judged VALID: 384/1493 (25.7%); runs with at least one VALID: 227/515 (44.1%), of which recovered 27/227 (11.9%)
- some derivation correct (ceiling): 91/515 (17.7%); final repeated known-wrong: 12/515 (2.3%)

### ESV v2, 1 strategy: stages (pooled)

- recognition said a failure occurred: 302/515 (58.6%)
- diagnosis verdicts: {'CORRECT': 190, 'INCORRECT': 192, 'PARTIAL': 133}
- strategies kept per run: {1: 515}; regenerated for similarity: 0
- strategies judged VALID: 98/513 (19.1%); runs with at least one VALID: 98/515 (19.0%), of which recovered 16/98 (16.3%)
- some derivation correct (ceiling): 41/515 (8.0%); final repeated known-wrong: 58/515 (11.3%)
