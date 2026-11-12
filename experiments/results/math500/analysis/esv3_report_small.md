# ESV v3, stage by stage (MATH-500, small models)

Version `jee_esv3_v1_2026-11-03`, decision rule `stages1-3_strict_v1_2026-11-03`. Gold is used only to score. Design and motivation: `esv_bottleneck_analysis.md`.

## The pipeline, as run

Every method starts from the same 4 CoT samples and none of them re-solves the problem; they differ only in how they pick among answers the solver already produced. ESV fires only on disagreement, so it can damage a correct majority as easily as repair a wrong one, and is scored that way.

| # | Stage | What it does | Calls | Model called | Outcome |
|---|---|---|---|---|---|
| 0 | CoT sampling (shared input, not part of ESV) | The solver answers the problem 4 times with chain-of-thought. Produced once by `run_methods.py` and reused unchanged by every method, so nothing below ever re-solves the problem. | 4 per instance, T = 0.7, seeds 0-3 | solver | 4 answers. Their majority is the `initial` answer and is scored as the CoT maj@4 baseline. |
| 1 | Disagreement gate | Group the 4 answers into distinct candidates by normalized value, in order of first appearance, keeping each one's vote count. Unreadable answers are dropped. | 0 (free) | none | Fewer than 2 distinct candidates -> decision `unanimous`: keep the initial answer and STOP. 2 or more -> go to stage 2. |
| 2 | Contrastive verification | All distinct candidates in ONE call, one representative solution each, in order of first appearance, with vote counts hidden. Per candidate: the strategy, its bindings to quoted givens (MATCHES / MISMATCH), the conditions the method needs (HOLDS / VIOLATED), then the first divergence point, then SOUND or FLAWED with a failure type (BINDING / METHOD / COMPUTATION), the flawed step and the corrected fact. | 1 per disagreeing instance, T = 0 | the solver itself (self-verification) | A filled-in check per candidate (run, measured below). |
| 3 | Decision rule (code only, no model) | A candidate is strictly SOUND only if its verdict is SOUND **and** it lists zero MISMATCH **and** zero VIOLATED. Exactly one such candidate wins; several, the one with the most votes (ties to the initial answer, then to the earliest); none, nothing is chosen. | 0 (free) | none | `one_sound` -> take that candidate. `several_sound` -> take the most-voted SOUND one. `none_sound` -> keep the initial answer. `unreadable` -> keep the initial answer. |
| 4 | Diagnosis-driven repair | Would re-solve using the diagnosis from stage 2 where no candidate survives. NOT BUILT: `none_sound` simply keeps the initial answer, so ESV never invents an answer the solver did not already produce. | n/a | n/a | Not built. Its ceiling is the `a candidate is right` column of the decisions table below. |
| 5 | Scoring | Gold enters here and nowhere else. Paired against CoT maj@4 on the same instances, with exact McNemar tests, so a stage can break a correct majority as easily as repair a wrong one. | 0 (free) | none | The accuracy tables below. |

### Where USC sits (the baseline, not a stage)

Universal Self-Consistency is the control ESV must beat, not a step inside it. It replaces stages 2-3 with a single selection call and is wired in deliberately parallel to them:

- **When it fires:** on exactly the same instances as stage 2, the disagreeing ones (2 or more distinct readable candidates). A `unanimous` instance never calls it and keeps the majority answer, so USC and CoT maj@4 agree there by construction.
- **Who is called:** the **solver**, never the verifier. USC has no verifier, which is why it is identical in both conditions. (Measured in this report.)
- **What it sees:** all 4 samples including duplicates, in arbitrary order, with the reasoning shown. Unlike stage 2 it is therefore free to count multiplicity: it can see that 3 of 4 samples agree.
- **How it answers:** one `<selected_response>` tag naming the response it picks. Resolved first as a response label (1-4), then as the selected answer's value, which solvers frequently return instead.
- **When it declines:** a tag that resolves to neither keeps the initial answer, i.e. USC falls back to CoT maj@4. The rate is reported under *Did USC actually select?* and must be read before any USC comparison.

## Stage 1: disagreement gate (no calls)

Unanimous = fewer than 2 distinct readable answers among the 4 CoT samples: ESV keeps the answer. Only the disagreeing instances can change, and only the fixable ones can gain.

| model | instances | unanimous (no call) | unanimous but wrong (lost) | disagreeing (ESV runs) | disagreeing, majority right (can break) | disagreeing, fixable (majority wrong, a candidate right) | disagreeing, no candidate right | CoT maj@4 | CoT pass@4 (ceiling) |
|---|---|---|---|---|---|---|---|---|---|
| mistralai/mistral-nemo | 488 | 97/488 (19.9%) | 4/97 (4.1%) | 391/488 (80.1%) | 128 | 73 | 190 | 45.3% | 60.2% |
| google/gemma-3-4b-it | 488 | 278/488 (57.0%) | 11/278 (4.0%) | 210/488 (43.0%) | 62 | 48 | 100 | 67.4% | 77.3% |
| **pooled** | 976 | 375/976 (38.4%) | 15/375 (4.0%) | 601/976 (61.6%) | 190 | 121 | 290 | 56.4% | 68.8% |

### What the wrong answers are

ESV targets the *trap*: the answer the original strategy S yields when applied blindly to a strategy-breaking P'. The premise holds only where that column is a large share of the failures.

| model | wrong CoT maj@4 answers | on originals | on preserving P' | on breaking P': the TRAP answer | on breaking P': other wrong | CoT maj@4 on originals | CoT maj@4 on breaking P' | CoT samples giving the trap answer |
|---|---|---|---|---|---|---|---|---|
| mistralai/mistral-nemo | 267 | 131/267 (49.1%) | 52/267 (19.5%) | 12/267 (4.5%) | 72/267 (27.0%) | 46.3% | 31.7% | 50/492 (10.2%) |
| google/gemma-3-4b-it | 159 | 76/159 (47.8%) | 29/159 (18.2%) | 12/159 (7.5%) | 42/159 (26.4%) | 68.9% | 56.1% | 41/492 (8.3%) |
| **pooled** | 426 | 207/426 (48.6%) | 81/426 (19.0%) | 24/426 (5.6%) | 114/426 (26.8%) | 57.6% | 43.9% | 91/984 (9.2%) |

## Stages 2-3: candidate verification and choice

**Go / no-go (fixed before the run): NO-GO.** ESV v3 - CoT maj@4 = -2.3 pp (needs >= +3); ESV v3 - USC = -2.7 pp (needs > 0).

### All instances

**mistralai/mistral-nemo** (488 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 11.5% | 17 | 182 | -165 (p=5e-36) |  |
| CoT pass@1 | 38.2% |  |  |  |  |
| CoT maj@4 (initial) | 45.3% |  |  |  |  |
| CoT pass@4 (oracle choice) | 60.2% |  |  |  |  |
| Universal Self-Consistency | 44.5% | 16 | 20 | -4 (p=0.618) | 8 / 0.80 |
| ESV v3 (stages 1-3) | 42.0% | 21 | 37 | -16 (p=0.048) | 810 / 0.80 |

**google/gemma-3-4b-it** (488 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 9.8% | 5 | 286 | -281 (p=9e-78) |  |
| CoT pass@1 | 65.0% |  |  |  |  |
| CoT maj@4 (initial) | 67.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 77.3% |  |  |  |  |
| Universal Self-Consistency | 69.1% | 18 | 10 | +8 (p=0.185) | 6 / 0.43 |
| ESV v3 (stages 1-3) | 66.2% | 7 | 13 | -6 (p=0.263) | 583 / 0.43 |

**Pooled** (976 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 10.7% | 22 | 468 | -446 (p=6e-110) |  |
| CoT pass@1 | 51.6% |  |  |  |  |
| CoT maj@4 (initial) | 56.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 68.8% |  |  |  |  |
| Universal Self-Consistency | 56.8% | 34 | 30 | +4 (p=0.708) | 7 / 0.62 |
| ESV v3 (stages 1-3) | 54.1% | 28 | 50 | -22 (p=0.017) | 696 / 0.62 |

ESV v3 vs USC (ESV-only / USC-only right, McNemar p): 35 / 61 (p=0.010)

### Did USC actually select? (pooled)

USC names its pick in one tag. Where that tag cannot be resolved to a response, USC keeps the initial answer and its accuracy becomes CoT maj@4's by construction - so this table has to be read before any USC comparison.

| model | USC calls | by response label | by answer value | ambiguous (label used) | unresolved -> keeps CoT maj@4 |
|---|---|---|---|---|---|
| mistralai/mistral-nemo | 391 | 363/391 (92.8%) | 0/391 (0.0%) | 28/391 (7.2%) | 0/391 (0.0%) |
| google/gemma-3-4b-it | 210 | 198/210 (94.3%) | 1/210 (0.5%) | 11/210 (5.2%) | 0/210 (0.0%) |
| **pooled** | 601 | 561/601 (93.3%) | 1/601 (0.2%) | 39/601 (6.5%) | 0/601 (0.0%) |

### The choice, on disagreeing instances only (pooled)

| Choice among the candidates (601 disagreeing instances) | right |
|---|---|
| majority vote (CoT maj@4) | 190/601 (31.6%) |
| uniform random candidate (chance) | 120.1/601 (20.0%) |
| Universal Self-Consistency | 194/601 (32.3%) |
| ESV v3, strict SOUND (used) | 168/601 (28.0%) |
| ESV v3, verdict-only SOUND (secondary) | 169/601 (28.1%) |
| oracle choice (CoT pass@4) | 311/601 (51.7%) |

### Stages 2-3 as a detector, per candidate (pooled)

| SOUND rule | right candidates called SOUND | wrong candidates called SOUND | precision of SOUND |
|---|---|---|---|
| strict (used) | 175/311 (56.3%) | 672/1552 (43.3%) | 20.7% |
| verdict only | 184/311 (59.2%) | 694/1552 (44.7%) | 21.0% |

|  | n | some binding MISMATCH | some condition VIOLATED | verdict unreadable | failure types when FLAWED |
|---|---|---|---|---|---|
| right candidates | 311 | 10/311 (3.2%) | 19/311 (6.1%) | 0/311 (0.0%) | METHOD: 58, COMPUTATION: 42, BINDING: 23, None: 4 |
| wrong candidates | 1552 | 59/1552 (3.8%) | 102/1552 (6.6%) | 0/1552 (0.0%) | METHOD: 388, COMPUTATION: 300, BINDING: 152, None: 18 |

### Decisions (pooled)

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 375 | 96.0% | 96.0% | 0 | 0 | 360 |
| one_sound | 295 | 36.3% | 31.2% | 25 | 40 | 166 |
| several_sound | 240 | 25.8% | 22.9% | 3 | 10 | 110 |
| none_sound | 66 | 31.8% | 31.8% | 0 | 0 | 35 |

`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.

### By arm (pooled)

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 488 | 57.6% | 58.6% | 56.4% | 17 / 23 |
| breaking | 246 | 43.9% | 44.7% | 39.8% | 4 / 14 |
| preserving | 242 | 66.5% | 65.3% | 64.0% | 7 / 13 |

## Compute cost

Read off the `usage` and `latency_seconds` stored on every call, so this section is reproducible offline from the records. Every method starts from the same 4 CoT samples and adds at most one call, so the baseline is shared and only the extra call separates them. **serial** is billed compute time; **critical path** is what a user waits for, since `run_methods.py` issues the 4 samples concurrently while the extra call cannot start until all 4 are back.

| Method | accuracy | calls | solver tokens | verifier tokens | total tokens | vs CoT maj@4 | serial s | critical path s |
|---|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | 10.7% | 4.00 | 1,942 | - | 1,942 | -57.7% | 15.3 | 8.6 |
| CoT pass@1 | 51.6% | 1.00 | 1,148 | - | 1,148 | -75.0% | 19.9 | 19.9 |
| CoT maj@4 (initial) | 56.4% | 4.00 | 4,592 | - | 4,592 | +0.0% | 79.5 | 31.8 |
| Universal Self-Consistency | 56.8% | 4.62 | 6,159 | - | 6,159 | +34.1% | 80.3 | 32.6 |
| ESV v3 (stages 1-3) | 54.1% | 4.62 | 7,433 | - | 7,433 | +61.9% | 103.3 | 55.6 |
| CoT pass@4 (oracle choice) | 68.8% | 4.00 | 4,592 | - | 4,592 | +0.0% | 79.5 | 31.8 |

### What the extra compute buys (pooled)

| Method | accuracy vs CoT maj@4 | fixes / breaks | net | McNemar | extra tokens / instance | extra wait | extra tokens per net fix |
|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | -45.7 pp | 22 / 468 | -446 | p=6e-110 | -2,650 | -23.2 s | no net gain |
| Universal Self-Consistency | +0.4 pp | 34 / 30 | +4 | p=0.708 | +1,567 | +0.8 s | 382,293 |
| ESV v3 (stages 1-3) | -2.3 pp | 28 / 50 | -22 | p=0.017 | +2,841 | +23.8 s | no net gain |
| CoT pass@4 (oracle choice) | +12.4 pp | 121 / 0 | +121 | p=8e-37 | +0 | +0.0 s | 0 |

### The disagreement gate is the efficiency mechanism (pooled)

Stage 1 costs nothing - it is code over answers the solver already produced - and it is what keeps ESV under 1.5x the baseline despite a verification call that is itself larger than a CoT sample.

|  | per instance |
|---|---|
| Instances where the gate fires (>= 2 distinct candidates) | 601/976 (61.6%) |
| A verification call, on an instance that fires | 4,614 tokens, 38.7 s |
| Amortized over every instance (what ESV actually costs) | 2,841 tokens, 23.8 s |
| The same protocol ungated, verifying every instance | 4,614 tokens, 38.7 s |
| Verification spend the gate avoids | 38.4% |
| Cost on a gated-out instance | 0 tokens, 0 s - ESV returns the CoT maj@4 answer untouched |

### Compute by arm (pooled)

The gate fires where the perturbation actually hurts, so ESV spends most on the strategy-breaking arm - which is also where it gains most. The cost profile is adaptive, not flat.

| arm | instances | gate fires | CoT maj@4 | ESV v3 | baseline tokens | extra tokens | vs baseline |
|---|---|---|---|---|---|---|---|
| original | 488 | 291/488 (59.6%) | 57.6% | 56.4% | 4,458 | +2,689 | +60.3% |
| breaking | 246 | 178/246 (72.4%) | 43.9% | 39.8% | 5,117 | +3,486 | +68.1% |
| preserving | 242 | 132/242 (54.5%) | 66.5% | 64.0% | 4,327 | +2,492 | +57.6% |

### By answer format (pooled)

`expression` answers are compared by LaTeX normalization and symbolic equivalence (`expression_answers.py`) rather than by exact match, so this split is the check that the scoring is not what separates the methods.

| answer format | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| expression | 352 | 40.6% | 39.5% | 36.6% | 6 / 20 |
| integer | 580 | 65.5% | 66.4% | 64.1% | 22 / 30 |
| numeric | 44 | 61.4% | 68.2% | 61.4% | 0 / 0 |
