# Gold-free evaluation: ESV on every instance (JEEBench (math), large models)

Version `jee_open_v1_2026-11-03`, vote rule `initial_votes_v1_2026-11-03` (the initial answer votes in ESV's and Recheck x3's final vote and wins ties). Every method starts from the CoT majority answer (CoT maj@4) and decides on its own whether to change it; the gold answer is used only to score the final answers. *Fixes* / *breaks* = instances the method turned wrong -> right / right -> wrong relative to CoT maj@4. Tokens = mean solver output tokens per instance beyond the 4 CoT samples (judge excluded). The two controls share ESV's answer checks; Recheck x3 also shares ESV's recognition decision, so ESV - Recheck x3 isolates the diagnosis and strategies.

| Model | instances with open results |
|---|---|
| qwen/qwen-2.5-72b-instruct | 60 / 60 |
| mistralai/mistral-small-3.2-24b-instruct | 60 / 60 |
| meta-llama/llama-3.3-70b-instruct | 60 / 60 |

## qwen/qwen-2.5-72b-instruct (60 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | tokens/instance |
|---|---|---|---|---|---|
| Direct QA pass@1 | 22.1% |  |  |  |  |
| Direct QA maj@4 | 21.7% | 5 | 18 | -13 (p=0.011) |  |
| CoT pass@1 | 38.3% |  |  |  |  |
| CoT maj@4 (initial) | 43.3% |  |  |  |  |
| Self-Refine (control) | 41.7% | 2 | 3 | -1 (p=1.000) | 851 |
| Recognition + 3 re-solves (control) | 43.3% | 0 | 0 | +0 (p=-) | 855 |
| ESV | 43.3% | 0 | 0 | +0 (p=-) | 1336 |

ESV head to head (ESV-only / other-only, McNemar p): vs CoT maj@4 (initial) 0 / 0 (p=-); vs Self-Refine (control) 3 / 2 (p=1.000); vs Recognition + 3 re-solves (control) 0 / 0 (p=-)

| Detector | catch rate (flags a wrong initial) | false alarms (flags a right initial) | precision |
|---|---|---|---|
| ESV recognition | 5/34 (14.7%) | 3/26 (11.5%) | 62.5% |
| Self-Refine verdict | 17/34 (50.0%) | 8/26 (30.8%) | 68.0% |

| Method | flagged & wrong: fixed | flagged & right: broken |
|---|---|---|
| ESV | 0/5 (0.0%) | 0/3 (0.0%) |
| Recognition + 3 re-solves (control) | 0/5 (0.0%) | 0/3 (0.0%) |
| Self-Refine (control) | 2/17 (11.8%) | 3/8 (37.5%) |

## mistralai/mistral-small-3.2-24b-instruct (60 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | tokens/instance |
|---|---|---|---|---|---|
| Direct QA pass@1 | 14.2% |  |  |  |  |
| Direct QA maj@4 | 18.3% | 5 | 20 | -15 (p=0.004) |  |
| CoT pass@1 | 34.6% |  |  |  |  |
| CoT maj@4 (initial) | 43.3% |  |  |  |  |
| Self-Refine (control) | 40.0% | 3 | 5 | -2 (p=0.727) | 917 |
| Recognition + 3 re-solves (control) | 43.3% | 3 | 3 | +0 (p=1.000) | 2063 |
| ESV | 38.3% | 1 | 4 | -3 (p=0.375) | 2487 |

ESV head to head (ESV-only / other-only, McNemar p): vs CoT maj@4 (initial) 1 / 4 (p=0.375); vs Self-Refine (control) 4 / 5 (p=1.000); vs Recognition + 3 re-solves (control) 2 / 5 (p=0.453)

| Detector | catch rate (flags a wrong initial) | false alarms (flags a right initial) | precision |
|---|---|---|---|
| ESV recognition | 18/34 (52.9%) | 10/26 (38.5%) | 64.3% |
| Self-Refine verdict | 23/34 (67.6%) | 16/26 (61.5%) | 59.0% |

| Method | flagged & wrong: fixed | flagged & right: broken |
|---|---|---|
| ESV | 1/18 (5.6%) | 4/10 (40.0%) |
| Recognition + 3 re-solves (control) | 3/18 (16.7%) | 3/10 (30.0%) |
| Self-Refine (control) | 3/23 (13.0%) | 5/16 (31.2%) |

## meta-llama/llama-3.3-70b-instruct (60 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | tokens/instance |
|---|---|---|---|---|---|
| Direct QA pass@1 | 11.7% |  |  |  |  |
| Direct QA maj@4 | 10.0% | 3 | 17 | -14 (p=0.003) |  |
| CoT pass@1 | 29.2% |  |  |  |  |
| CoT maj@4 (initial) | 33.3% |  |  |  |  |
| Self-Refine (control) | 35.0% | 2 | 1 | +1 (p=1.000) | 562 |
| Recognition + 3 re-solves (control) | 33.3% | 0 | 0 | +0 (p=-) | 1765 |
| ESV | 33.3% | 0 | 0 | +0 (p=-) | 1999 |

ESV head to head (ESV-only / other-only, McNemar p): vs CoT maj@4 (initial) 0 / 0 (p=-); vs Self-Refine (control) 1 / 2 (p=1.000); vs Recognition + 3 re-solves (control) 0 / 0 (p=-)

| Detector | catch rate (flags a wrong initial) | false alarms (flags a right initial) | precision |
|---|---|---|---|
| ESV recognition | 11/40 (27.5%) | 4/20 (20.0%) | 73.3% |
| Self-Refine verdict | 26/40 (65.0%) | 7/20 (35.0%) | 78.8% |

| Method | flagged & wrong: fixed | flagged & right: broken |
|---|---|---|
| ESV | 0/11 (0.0%) | 0/4 (0.0%) |
| Recognition + 3 re-solves (control) | 0/11 (0.0%) | 0/4 (0.0%) |
| Self-Refine (control) | 2/26 (7.7%) | 1/7 (14.3%) |

## Pooled (180 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | tokens/instance |
|---|---|---|---|---|---|
| Direct QA pass@1 | 16.0% |  |  |  |  |
| Direct QA maj@4 | 16.7% | 13 | 55 | -42 (p=0.000) |  |
| CoT pass@1 | 34.0% |  |  |  |  |
| CoT maj@4 (initial) | 40.0% |  |  |  |  |
| Self-Refine (control) | 38.9% | 7 | 9 | -2 (p=0.804) | 777 |
| Recognition + 3 re-solves (control) | 40.0% | 3 | 3 | +0 (p=1.000) | 1561 |
| ESV | 38.3% | 1 | 4 | -3 (p=0.375) | 1941 |

ESV head to head (ESV-only / other-only, McNemar p): vs CoT maj@4 (initial) 1 / 4 (p=0.375); vs Self-Refine (control) 8 / 9 (p=1.000); vs Recognition + 3 re-solves (control) 2 / 5 (p=0.453)

| Detector | catch rate (flags a wrong initial) | false alarms (flags a right initial) | precision |
|---|---|---|---|
| ESV recognition | 34/108 (31.5%) | 17/72 (23.6%) | 66.7% |
| Self-Refine verdict | 66/108 (61.1%) | 31/72 (43.1%) | 68.0% |

| Method | flagged & wrong: fixed | flagged & right: broken |
|---|---|---|
| ESV | 1/34 (2.9%) | 4/17 (23.5%) |
| Recognition + 3 re-solves (control) | 3/34 (8.8%) | 3/17 (17.6%) |
| Self-Refine (control) | 7/66 (10.6%) | 9/31 (29.0%) |

### Pooled, by arm

| group | n | CoT maj@4 | Self-Refine | Recheck x3 | ESV | ESV fixes / breaks |
|---|---|---|---|---|---|---|
| original | 90 | 33.3% | 32.2% | 33.3% | 32.2% | 0 / 1 |
| breaking | 45 | 44.4% | 40.0% | 44.4% | 37.8% | 0 / 3 |
| preserving | 45 | 48.9% | 51.1% | 48.9% | 51.1% | 1 / 0 |

### Pooled, by answer format

| group | n | CoT maj@4 | Self-Refine | Recheck x3 | ESV | ESV fixes / breaks |
|---|---|---|---|---|---|---|
| single | 36 | 44.4% | 33.3% | 38.9% | 38.9% | 0 / 2 |
| multiple | 90 | 33.3% | 33.3% | 34.4% | 32.2% | 1 / 2 |
| integer | 24 | 12.5% | 20.8% | 16.7% | 12.5% | 0 / 0 |
| numeric | 30 | 76.7% | 76.7% | 76.7% | 76.7% | 0 / 0 |

### ESV lifecycle on flagged wrong answers (pooled)

34 instances. Judge verdicts are measurement only.

| ESV stage outcome | n | fixed |
|---|---|---|
| diagnosis CORRECT | 24 | 0/24 (0.0%) |
| diagnosis PARTIAL | 5 | 0/5 (0.0%) |
| diagnosis INCORRECT | 5 | 1/5 (20.0%) |
| some strategy VALID | 32 | 1/32 (3.1%) |
| no strategy VALID | 2 | 0/2 (0.0%) |
| some derivation correct (ceiling) | 10 | 1/10 (10.0%) |

- strategies kept per run: {3: 34}
