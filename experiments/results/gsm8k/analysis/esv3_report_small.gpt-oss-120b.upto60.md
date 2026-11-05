# ESV v3, stage by stage (GSM8K, small models, verifier openai/gpt-oss-120b, DEV pairs < 60)

Version `jee_esv3_v1_2026-11-03`, decision rule `stages1-3_strict_v1_2026-11-03`. Gold is used only to score. Design and motivation: `esv_bottleneck_analysis.md`.

| Stage | Status |
|---|---|
| 1. Disagreement gate | built, measured below |
| 2-3. Candidate verification + choice | run, measured below |
| 4. Diagnosis-driven repair | not built (decision `none_sound` keeps the initial answer) |

Excluded from every score: 6 instances whose published gold is wrong or ambiguous (`common.DISPUTED_INSTANCES`): 105_original, 111_original, 189_original, 40_original, 82_original, 95_original.

## Stage 1: disagreement gate (no calls)

Unanimous = fewer than 2 distinct readable answers among the 4 CoT samples: ESV keeps the answer. Only the disagreeing instances can change, and only the fixable ones can gain.

| model | instances | unanimous (no call) | unanimous but wrong (lost) | disagreeing (ESV runs) | disagreeing, majority right (can break) | disagreeing, fixable (majority wrong, a candidate right) | disagreeing, no candidate right | CoT maj@4 | CoT pass@4 (ceiling) |
|---|---|---|---|---|---|---|---|---|---|
| qwen/qwen-2.5-7b-instruct | 119 | 101/119 (84.9%) | 0/101 (0.0%) | 18/119 (15.1%) | 9 | 4 | 5 | 92.4% | 95.8% |
| mistralai/mistral-nemo | 119 | 74/119 (62.2%) | 1/74 (1.4%) | 45/119 (37.8%) | 25 | 12 | 8 | 82.4% | 92.4% |
| google/gemma-3-4b-it | 119 | 96/119 (80.7%) | 3/96 (3.1%) | 23/119 (19.3%) | 12 | 4 | 7 | 88.2% | 91.6% |
| **pooled** | 357 | 271/357 (75.9%) | 4/271 (1.5%) | 86/357 (24.1%) | 46 | 20 | 20 | 87.7% | 93.3% |

### What the wrong answers are

ESV targets the *trap*: the answer the original strategy S yields when applied blindly to a strategy-breaking P'. The premise holds only where that column is a large share of the failures.

| model | wrong CoT maj@4 answers | on originals | on preserving P' | on breaking P': the TRAP answer | on breaking P': other wrong | CoT maj@4 on originals | CoT maj@4 on breaking P' | CoT samples giving the trap answer |
|---|---|---|---|---|---|---|---|---|
| qwen/qwen-2.5-7b-instruct | 9 | 4/9 (44.4%) | 0/9 (0.0%) | 0/9 (0.0%) | 5/9 (55.6%) | 93.2% | 83.3% | 4/120 (3.3%) |
| mistralai/mistral-nemo | 21 | 6/21 (28.6%) | 5/21 (23.8%) | 3/21 (14.3%) | 7/21 (33.3%) | 89.8% | 66.7% | 9/120 (7.5%) |
| google/gemma-3-4b-it | 14 | 6/14 (42.9%) | 3/14 (21.4%) | 1/14 (7.1%) | 4/14 (28.6%) | 89.8% | 83.3% | 7/120 (5.8%) |
| **pooled** | 44 | 16/44 (36.4%) | 8/44 (18.2%) | 4/44 (9.1%) | 16/44 (36.4%) | 91.0% | 77.8% | 20/360 (5.6%) |

## Stages 2-3: candidate verification and choice

**Go / no-go (fixed before the run): GO.** ESV v3 - CoT maj@4 = +4.5 pp (needs >= +3); ESV v3 - USC = +4.5 pp (needs > 0).

### All instances

**qwen/qwen-2.5-7b-instruct** (119 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 20.2% | 0 | 86 | -86 (p=0.000) |  |
| CoT pass@1 | 91.2% |  |  |  |  |
| CoT maj@4 (initial) | 92.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 95.8% |  |  |  |  |
| Universal Self-Consistency | 92.4% | 0 | 0 | +0 (p=-) | 0 / 0.00 |
| ESV v3 (stages 1-3) | 95.8% | 4 | 0 | +4 (p=0.125) | 342 / 0.15 |

**mistralai/mistral-nemo** (119 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 12.6% | 1 | 84 | -83 (p=0.000) |  |
| CoT pass@1 | 78.2% |  |  |  |  |
| CoT maj@4 (initial) | 82.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 92.4% |  |  |  |  |
| Universal Self-Consistency | 82.4% | 0 | 0 | +0 (p=-) | 0 / 0.00 |
| ESV v3 (stages 1-3) | 90.8% | 10 | 0 | +10 (p=0.002) | 786 / 0.38 |

**google/gemma-3-4b-it** (119 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 8.4% | 0 | 95 | -95 (p=0.000) |  |
| CoT pass@1 | 85.7% |  |  |  |  |
| CoT maj@4 (initial) | 88.2% |  |  |  |  |
| CoT pass@4 (oracle choice) | 91.6% |  |  |  |  |
| Universal Self-Consistency | 88.2% | 0 | 0 | +0 (p=-) | 0 / 0.00 |
| ESV v3 (stages 1-3) | 89.9% | 2 | 0 | +2 (p=0.500) | 467 / 0.19 |

**Pooled** (357 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 13.7% | 1 | 265 | -264 (p=0.000) |  |
| CoT pass@1 | 85.0% |  |  |  |  |
| CoT maj@4 (initial) | 87.7% |  |  |  |  |
| CoT pass@4 (oracle choice) | 93.3% |  |  |  |  |
| Universal Self-Consistency | 87.7% | 0 | 0 | +0 (p=-) | 0 / 0.00 |
| ESV v3 (stages 1-3) | 92.2% | 16 | 0 | +16 (p=0.000) | 532 / 0.24 |

ESV v3 vs USC (ESV-only / USC-only right, McNemar p): 16 / 0 (p=0.000)

### The choice, on disagreeing instances only (pooled)

| Choice among the candidates (86 disagreeing instances) | right |
|---|---|
| majority vote (CoT maj@4) | 46/86 (53.5%) |
| uniform random candidate (chance) | 28.2/86 (32.8%) |
| Universal Self-Consistency | 46/86 (53.5%) |
| ESV v3, strict SOUND (used) | 62/86 (72.1%) |
| ESV v3, verdict-only SOUND (secondary) | 62/86 (72.1%) |
| oracle choice (CoT pass@4) | 66/86 (76.7%) |

### Stages 2-3 as a detector, per candidate (pooled)

| SOUND rule | right candidates called SOUND | wrong candidates called SOUND | precision of SOUND |
|---|---|---|---|
| strict (used) | 54/66 (81.8%) | 5/162 (3.1%) | 91.5% |
| verdict only | 54/66 (81.8%) | 5/162 (3.1%) | 91.5% |

|  | n | some binding MISMATCH | some condition VIOLATED | verdict unreadable | failure types when FLAWED |
|---|---|---|---|---|---|
| right candidates | 66 | 7/66 (10.6%) | 3/66 (4.5%) | 0/66 (0.0%) | BINDING: 9, COMPUTATION: 2, METHOD: 1 |
| wrong candidates | 162 | 105/162 (64.8%) | 95/162 (58.6%) | 0/162 (0.0%) | BINDING: 90, COMPUTATION: 42, METHOD: 25 |

### Decisions (pooled)

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 271 | 98.5% | 98.5% | 0 | 0 | 267 |
| one_sound | 59 | 64.4% | 91.5% | 16 | 0 | 55 |
| none_sound | 27 | 29.6% | 29.6% | 0 | 0 | 11 |

`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.

### By arm (pooled)

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 177 | 91.0% | 91.0% | 93.2% | 4 / 0 |
| breaking | 90 | 77.8% | 77.8% | 86.7% | 8 / 0 |
| preserving | 90 | 91.1% | 91.1% | 95.6% | 4 / 0 |
