# Gold-free evaluation: ESV on every instance (JEEBench, small models)

Version `jee_open_v1_2026-11-03`, vote rule `initial_votes_v1_2026-11-03` (the initial answer votes in ESV's and Recheck x3's final vote and wins ties). Every method starts from the CoT majority answer (CoT maj@4) and decides on its own whether to change it; the gold answer is used only to score the final answers. *Fixes* / *breaks* = instances the method turned wrong -> right / right -> wrong relative to CoT maj@4. Tokens = mean solver output tokens per instance beyond the 4 CoT samples (judge excluded). The two controls share ESV's answer checks; Recheck x3 also shares ESV's recognition decision, so ESV - Recheck x3 isolates the diagnosis and strategies.

| Model | instances with open results |
|---|---|
| qwen/qwen-2.5-7b-instruct | 386 / 386 |
| mistralai/mistral-nemo | 386 / 386 |
| google/gemma-3-4b-it | 193 / 386 |

## qwen/qwen-2.5-7b-instruct (386 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | tokens/instance |
|---|---|---|---|---|---|
| Direct QA pass@1 | 20.1% |  |  |  |  |
| Direct QA maj@4 | 19.7% | 35 | 93 | -58 (p=0.000) |  |
| CoT pass@1 | 30.8% |  |  |  |  |
| CoT maj@4 (initial) | 34.7% |  |  |  |  |
| Self-Refine (control) | 33.2% | 13 | 19 | -6 (p=0.377) | 497 |
| Recognition + 3 re-solves (control) | 35.8% | 8 | 4 | +4 (p=0.388) | 1802 |
| ESV | 33.9% | 11 | 14 | -3 (p=0.690) | 2111 |

ESV head to head (ESV-only / other-only, McNemar p): vs CoT maj@4 (initial) 11 / 14 (p=0.690); vs Self-Refine (control) 25 / 22 (p=0.771); vs Recognition + 3 re-solves (control) 12 / 19 (p=0.281)

| Detector | catch rate (flags a wrong initial) | false alarms (flags a right initial) | precision |
|---|---|---|---|
| ESV recognition | 134/252 (53.2%) | 55/134 (41.0%) | 70.9% |
| Self-Refine verdict | 170/252 (67.5%) | 64/134 (47.8%) | 72.6% |

| Method | flagged & wrong: fixed | flagged & right: broken |
|---|---|---|
| ESV | 11/134 (8.2%) | 14/55 (25.5%) |
| Recognition + 3 re-solves (control) | 8/134 (6.0%) | 4/55 (7.3%) |
| Self-Refine (control) | 12/170 (7.1%) | 19/64 (29.7%) |

## mistralai/mistral-nemo (386 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | tokens/instance |
|---|---|---|---|---|---|
| Direct QA pass@1 | 11.7% |  |  |  |  |
| Direct QA maj@4 | 11.9% | 33 | 50 | -17 (p=0.078) |  |
| CoT pass@1 | 13.1% |  |  |  |  |
| CoT maj@4 (initial) | 16.3% |  |  |  |  |
| Self-Refine (control) | 13.2% | 7 | 19 | -12 (p=0.029) | 379 |
| Recognition + 3 re-solves (control) | 15.8% | 4 | 6 | -2 (p=0.754) | 1328 |
| ESV | 15.3% | 5 | 9 | -4 (p=0.424) | 1686 |

ESV head to head (ESV-only / other-only, McNemar p): vs CoT maj@4 (initial) 5 / 9 (p=0.424); vs Self-Refine (control) 21 / 13 (p=0.229); vs Recognition + 3 re-solves (control) 7 / 9 (p=0.804)

| Detector | catch rate (flags a wrong initial) | false alarms (flags a right initial) | precision |
|---|---|---|---|
| ESV recognition | 172/323 (53.3%) | 33/63 (52.4%) | 83.9% |
| Self-Refine verdict | 146/323 (45.2%) | 23/63 (36.5%) | 86.4% |

| Method | flagged & wrong: fixed | flagged & right: broken |
|---|---|---|
| ESV | 5/172 (2.9%) | 9/33 (27.3%) |
| Recognition + 3 re-solves (control) | 4/172 (2.3%) | 6/33 (18.2%) |
| Self-Refine (control) | 5/146 (3.4%) | 16/23 (69.6%) |

## google/gemma-3-4b-it (193 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | tokens/instance |
|---|---|---|---|---|---|
| Direct QA pass@1 | 10.4% |  |  |  |  |
| Direct QA maj@4 | 12.4% | 12 | 47 | -35 (p=0.000) |  |
| CoT pass@1 | 26.4% |  |  |  |  |
| CoT maj@4 (initial) | 30.6% |  |  |  |  |
| Self-Refine (control) | 31.1% | 8 | 7 | +1 (p=1.000) | 1112 |
| Recognition + 3 re-solves (control) | 30.1% | 3 | 4 | -1 (p=1.000) | 3781 |
| ESV | 32.6% | 4 | 0 | +4 (p=0.125) | 4595 |

ESV head to head (ESV-only / other-only, McNemar p): vs CoT maj@4 (initial) 4 / 0 (p=0.125); vs Self-Refine (control) 11 / 8 (p=0.648); vs Recognition + 3 re-solves (control) 7 / 2 (p=0.180)

| Detector | catch rate (flags a wrong initial) | false alarms (flags a right initial) | precision |
|---|---|---|---|
| ESV recognition | 98/134 (73.1%) | 31/59 (52.5%) | 76.0% |
| Self-Refine verdict | 65/134 (48.5%) | 15/59 (25.4%) | 81.2% |

| Method | flagged & wrong: fixed | flagged & right: broken |
|---|---|---|
| ESV | 4/98 (4.1%) | 0/31 (0.0%) |
| Recognition + 3 re-solves (control) | 3/98 (3.1%) | 4/31 (12.9%) |
| Self-Refine (control) | 4/65 (6.2%) | 5/15 (33.3%) |

## Pooled (965 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | tokens/instance |
|---|---|---|---|---|---|
| Direct QA pass@1 | 14.8% |  |  |  |  |
| Direct QA maj@4 | 15.1% | 80 | 190 | -110 (p=0.000) |  |
| CoT pass@1 | 22.8% |  |  |  |  |
| CoT maj@4 (initial) | 26.5% |  |  |  |  |
| Self-Refine (control) | 24.8% | 28 | 45 | -17 (p=0.060) | 573 |
| Recognition + 3 re-solves (control) | 26.6% | 15 | 14 | +1 (p=1.000) | 2008 |
| ESV | 26.2% | 20 | 23 | -3 (p=0.761) | 2438 |

ESV head to head (ESV-only / other-only, McNemar p): vs CoT maj@4 (initial) 20 / 23 (p=0.761); vs Self-Refine (control) 57 / 43 (p=0.193); vs Recognition + 3 re-solves (control) 26 / 30 (p=0.689)

| Detector | catch rate (flags a wrong initial) | false alarms (flags a right initial) | precision |
|---|---|---|---|
| ESV recognition | 404/709 (57.0%) | 119/256 (46.5%) | 77.2% |
| Self-Refine verdict | 381/709 (53.7%) | 102/256 (39.8%) | 78.9% |

| Method | flagged & wrong: fixed | flagged & right: broken |
|---|---|---|
| ESV | 20/404 (5.0%) | 23/119 (19.3%) |
| Recognition + 3 re-solves (control) | 15/404 (3.7%) | 14/119 (11.8%) |
| Self-Refine (control) | 21/381 (5.5%) | 40/102 (39.2%) |

### Pooled, by arm

| group | n | CoT maj@4 | Self-Refine | Recheck x3 | ESV | ESV fixes / breaks |
|---|---|---|---|---|---|---|
| original | 483 | 26.9% | 24.6% | 26.7% | 27.1% | 11 / 10 |
| breaking | 242 | 23.1% | 23.6% | 23.1% | 21.5% | 4 / 8 |
| preserving | 240 | 29.2% | 26.2% | 30.0% | 29.2% | 5 / 5 |

### Pooled, by answer format

| group | n | CoT maj@4 | Self-Refine | Recheck x3 | ESV | ESV fixes / breaks |
|---|---|---|---|---|---|---|
| single | 197 | 41.1% | 36.5% | 41.1% | 39.6% | 9 / 12 |
| multiple | 350 | 17.1% | 15.7% | 17.1% | 18.3% | 5 / 1 |
| integer | 146 | 26.7% | 26.0% | 28.1% | 25.3% | 2 / 4 |
| numeric | 272 | 27.9% | 27.2% | 27.6% | 27.2% | 4 / 6 |

### ESV lifecycle on flagged wrong answers (pooled)

404 instances. Judge verdicts are measurement only.

| ESV stage outcome | n | fixed |
|---|---|---|
| diagnosis CORRECT | 164 | 5/164 (3.0%) |
| diagnosis PARTIAL | 123 | 7/123 (5.7%) |
| diagnosis INCORRECT | 117 | 8/117 (6.8%) |
| some strategy VALID | 204 | 8/204 (3.9%) |
| no strategy VALID | 200 | 12/200 (6.0%) |
| some derivation correct (ceiling) | 91 | 20/91 (22.0%) |

- strategies kept per run: {1: 11, 2: 12, 3: 381}
