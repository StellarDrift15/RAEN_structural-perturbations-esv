# ESV v3, stage by stage (JEEBench (math), large models)

Version `jee_esv3_v1_2026-11-03`, decision rule `stages1-3_strict_v1_2026-11-03`. Gold is used only to score. Design and motivation: `esv_bottleneck_analysis.md`.

| Stage | Status |
|---|---|
| 1. Disagreement gate | built, measured below |
| 2-3. Candidate verification + choice | run, measured below |
| 4. Diagnosis-driven repair | not built (decision `none_sound` keeps the initial answer) |

## Stage 1: disagreement gate (no calls)

Unanimous = fewer than 2 distinct readable answers among the 4 CoT samples: ESV keeps the answer. Only the disagreeing instances can change, and only the fixable ones can gain.

| model | instances | unanimous (no call) | unanimous but wrong (lost) | disagreeing (ESV runs) | disagreeing, majority right (can break) | disagreeing, fixable (majority wrong, a candidate right) | disagreeing, no candidate right | CoT maj@4 | CoT pass@4 (ceiling) |
|---|---|---|---|---|---|---|---|---|---|
| qwen/qwen-2.5-72b-instruct | 60 | 15/60 (25.0%) | 5/15 (33.3%) | 45/60 (75.0%) | 16 | 11 | 18 | 43.3% | 61.7% |
| mistralai/mistral-small-3.2-24b-instruct | 60 | 8/60 (13.3%) | 2/8 (25.0%) | 52/60 (86.7%) | 20 | 13 | 19 | 43.3% | 65.0% |
| meta-llama/llama-3.3-70b-instruct | 60 | 13/60 (21.7%) | 8/13 (61.5%) | 47/60 (78.3%) | 15 | 12 | 20 | 33.3% | 53.3% |
| **pooled** | 180 | 36/180 (20.0%) | 15/36 (41.7%) | 144/180 (80.0%) | 51 | 36 | 57 | 40.0% | 60.0% |

### What the wrong answers are

ESV targets the *trap*: the answer the original strategy S yields when applied blindly to a strategy-breaking P'. The premise holds only where that column is a large share of the failures.

| model | wrong CoT maj@4 answers | on originals | on preserving P' | on breaking P': the TRAP answer | on breaking P': other wrong | CoT maj@4 on originals | CoT maj@4 on breaking P' | CoT samples giving the trap answer |
|---|---|---|---|---|---|---|---|---|
| qwen/qwen-2.5-72b-instruct | 34 | 19/34 (55.9%) | 7/34 (20.6%) | 2/34 (5.9%) | 6/34 (17.6%) | 36.7% | 46.7% | 9/60 (15.0%) |
| mistralai/mistral-small-3.2-24b-instruct | 34 | 20/34 (58.8%) | 9/34 (26.5%) | 3/34 (8.8%) | 2/34 (5.9%) | 33.3% | 66.7% | 11/60 (18.3%) |
| meta-llama/llama-3.3-70b-instruct | 40 | 21/40 (52.5%) | 7/40 (17.5%) | 6/40 (15.0%) | 6/40 (15.0%) | 30.0% | 20.0% | 18/60 (30.0%) |
| **pooled** | 108 | 60/108 (55.6%) | 23/108 (21.3%) | 11/108 (10.2%) | 14/108 (13.0%) | 33.3% | 44.4% | 38/180 (21.1%) |

## Stages 2-3: candidate verification and choice

**Go / no-go (fixed before the run): NO-GO.** ESV v3 - CoT maj@4 = -7.2 pp (needs >= +3); ESV v3 - USC = -5.6 pp (needs > 0).

### All instances

**qwen/qwen-2.5-72b-instruct** (60 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 21.7% | 5 | 18 | -13 (p=0.011) |  |
| CoT pass@1 | 38.3% |  |  |  |  |
| CoT maj@4 (initial) | 43.3% |  |  |  |  |
| CoT pass@4 (oracle choice) | 61.7% |  |  |  |  |
| Self-Refine (Exp. 3) | 41.7% | 2 | 3 | -1 (p=1.000) | 851 / 1.00 |
| Recheck x3 (Exp. 3) | 43.3% | 0 | 0 | +0 (p=-) | 855 / 1.40 |
| ESV v2 (Exp. 3) | 43.3% | 0 | 0 | +0 (p=-) | 1336 / 1.70 |
| Universal Self-Consistency | 38.3% | 2 | 5 | -3 (p=0.453) | 7 / 0.75 |
| ESV v3 (stages 1-3) | 31.7% | 2 | 9 | -7 (p=0.065) | 843 / 0.75 |

**mistralai/mistral-small-3.2-24b-instruct** (60 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 18.3% | 5 | 20 | -15 (p=0.004) |  |
| CoT pass@1 | 34.6% |  |  |  |  |
| CoT maj@4 (initial) | 43.3% |  |  |  |  |
| CoT pass@4 (oracle choice) | 65.0% |  |  |  |  |
| Self-Refine (Exp. 3) | 40.0% | 3 | 5 | -2 (p=0.727) | 917 / 1.00 |
| Recheck x3 (Exp. 3) | 43.3% | 3 | 3 | +0 (p=1.000) | 2063 / 2.40 |
| ESV v2 (Exp. 3) | 38.3% | 1 | 4 | -3 (p=0.375) | 2487 / 3.33 |
| Universal Self-Consistency | 40.0% | 3 | 5 | -2 (p=0.727) | 9 / 0.87 |
| ESV v3 (stages 1-3) | 35.0% | 3 | 8 | -5 (p=0.227) | 1034 / 0.87 |

**meta-llama/llama-3.3-70b-instruct** (60 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 10.0% | 3 | 17 | -14 (p=0.003) |  |
| CoT pass@1 | 29.2% |  |  |  |  |
| CoT maj@4 (initial) | 33.3% |  |  |  |  |
| CoT pass@4 (oracle choice) | 53.3% |  |  |  |  |
| Self-Refine (Exp. 3) | 35.0% | 2 | 1 | +1 (p=1.000) | 562 / 1.00 |
| Recheck x3 (Exp. 3) | 33.3% | 0 | 0 | +0 (p=-) | 1765 / 1.75 |
| ESV v2 (Exp. 3) | 33.3% | 0 | 0 | +0 (p=-) | 1999 / 2.25 |
| Universal Self-Consistency | 36.7% | 2 | 0 | +2 (p=0.500) | 9 / 0.78 |
| ESV v3 (stages 1-3) | 31.7% | 4 | 5 | -1 (p=1.000) | 756 / 0.78 |

**Pooled** (180 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 16.7% | 13 | 55 | -42 (p=0.000) |  |
| CoT pass@1 | 34.0% |  |  |  |  |
| CoT maj@4 (initial) | 40.0% |  |  |  |  |
| CoT pass@4 (oracle choice) | 60.0% |  |  |  |  |
| Self-Refine (Exp. 3) | 38.9% | 7 | 9 | -2 (p=0.804) | 777 / 1.00 |
| Recheck x3 (Exp. 3) | 40.0% | 3 | 3 | +0 (p=1.000) | 1561 / 1.85 |
| ESV v2 (Exp. 3) | 38.3% | 1 | 4 | -3 (p=0.375) | 1941 / 2.43 |
| Universal Self-Consistency | 38.3% | 7 | 10 | -3 (p=0.629) | 8 / 0.80 |
| ESV v3 (stages 1-3) | 32.8% | 9 | 22 | -13 (p=0.029) | 878 / 0.80 |

ESV v3 vs USC (ESV-only / USC-only right, McNemar p): 6 / 16 (p=0.052)

### The choice, on disagreeing instances only (pooled)

| Choice among the candidates (144 disagreeing instances) | right |
|---|---|
| majority vote (CoT maj@4) | 51/144 (35.4%) |
| uniform random candidate (chance) | 34.5/144 (24.0%) |
| Universal Self-Consistency | 48/144 (33.3%) |
| ESV v3, strict SOUND (used) | 38/144 (26.4%) |
| ESV v3, verdict-only SOUND (secondary) | 38/144 (26.4%) |
| oracle choice (CoT pass@4) | 87/144 (60.4%) |

### Stages 2-3 as a detector, per candidate (pooled)

| SOUND rule | right candidates called SOUND | wrong candidates called SOUND | precision of SOUND |
|---|---|---|---|
| strict (used) | 38/87 (43.7%) | 112/308 (36.4%) | 25.3% |
| verdict only | 38/87 (43.7%) | 116/308 (37.7%) | 24.7% |

|  | n | some binding MISMATCH | some condition VIOLATED | verdict unreadable | failure types when FLAWED |
|---|---|---|---|---|---|
| right candidates | 87 | 5/87 (5.7%) | 9/87 (10.3%) | 0/87 (0.0%) | COMPUTATION: 23, METHOD: 18, BINDING: 8 |
| wrong candidates | 308 | 26/308 (8.4%) | 55/308 (17.9%) | 0/308 (0.0%) | COMPUTATION: 95, BINDING: 50, METHOD: 47 |

### Decisions (pooled)

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 36 | 58.3% | 58.3% | 0 | 0 | 21 |
| one_sound | 89 | 41.6% | 29.2% | 7 | 18 | 56 |
| several_sound | 29 | 34.5% | 27.6% | 2 | 4 | 20 |
| none_sound | 26 | 15.4% | 15.4% | 0 | 0 | 11 |

`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.

### By arm (pooled)

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 90 | 33.3% | 32.2% | 31.1% | 6 / 8 |
| breaking | 45 | 44.4% | 40.0% | 31.1% | 1 / 7 |
| preserving | 45 | 48.9% | 48.9% | 37.8% | 2 / 7 |
