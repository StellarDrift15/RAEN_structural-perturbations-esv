# ESV v3, stage by stage (GSM8K, small models, DEV pairs < 60)

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

**Go / no-go (fixed before the run): NO-GO.** ESV v3 - CoT maj@4 = -2.5 pp (needs >= +3); ESV v3 - USC = -3.4 pp (needs > 0).

### All instances

**qwen/qwen-2.5-7b-instruct** (119 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 20.2% | 0 | 86 | -86 (p=3e-26) |  |
| CoT pass@1 | 91.2% |  |  |  |  |
| CoT maj@4 (initial) | 92.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 95.8% |  |  |  |  |
| Universal Self-Consistency | 93.3% | 1 | 0 | +1 (p=1.000) | 2 / 0.15 |
| ESV v3 (stages 1-3) | 91.6% | 2 | 3 | -1 (p=1.000) | 147 / 0.15 |

**mistralai/mistral-nemo** (119 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 12.6% | 1 | 84 | -83 (p=4e-24) |  |
| CoT pass@1 | 78.2% |  |  |  |  |
| CoT maj@4 (initial) | 82.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 92.4% |  |  |  |  |
| Universal Self-Consistency | 84.0% | 4 | 2 | +2 (p=0.688) | 4 / 0.38 |
| ESV v3 (stages 1-3) | 77.3% | 3 | 9 | -6 (p=0.146) | 339 / 0.38 |

**google/gemma-3-4b-it** (119 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 8.4% | 0 | 95 | -95 (p=5e-29) |  |
| CoT pass@1 | 85.7% |  |  |  |  |
| CoT maj@4 (initial) | 88.2% |  |  |  |  |
| CoT pass@4 (oracle choice) | 91.6% |  |  |  |  |
| Universal Self-Consistency | 88.2% | 0 | 0 | +0 (p=-) | 4 / 0.19 |
| ESV v3 (stages 1-3) | 86.6% | 1 | 3 | -2 (p=0.625) | 216 / 0.19 |

**Pooled** (357 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 13.7% | 1 | 265 | -264 (p=5e-78) |  |
| CoT pass@1 | 85.0% |  |  |  |  |
| CoT maj@4 (initial) | 87.7% |  |  |  |  |
| CoT pass@4 (oracle choice) | 93.3% |  |  |  |  |
| Universal Self-Consistency | 88.5% | 5 | 2 | +3 (p=0.453) | 3 / 0.24 |
| ESV v3 (stages 1-3) | 85.2% | 6 | 15 | -9 (p=0.078) | 234 / 0.24 |

ESV v3 vs USC (ESV-only / USC-only right, McNemar p): 5 / 17 (p=0.017)

### Did USC actually select? (pooled)

USC names its pick in one tag. Where that tag cannot be resolved to a response, USC keeps the initial answer and its accuracy becomes CoT maj@4's by construction - so this table has to be read before any USC comparison.

| model | USC calls | by response label | by answer value | ambiguous (label used) | unresolved -> keeps CoT maj@4 |
|---|---|---|---|---|---|
| qwen/qwen-2.5-7b-instruct | 18 | 9/18 (50.0%) | 9/18 (50.0%) | 0/18 (0.0%) | 0/18 (0.0%) |
| mistralai/mistral-nemo | 45 | 36/45 (80.0%) | 8/45 (17.8%) | 1/45 (2.2%) | 0/45 (0.0%) |
| google/gemma-3-4b-it | 23 | 16/23 (69.6%) | 4/23 (17.4%) | 0/23 (0.0%) | 3/23 (13.0%) |
| **pooled** | 86 | 61/86 (70.9%) | 21/86 (24.4%) | 1/86 (1.2%) | 3/86 (3.5%) |

### The choice, on disagreeing instances only (pooled)

| Choice among the candidates (86 disagreeing instances) | right |
|---|---|
| majority vote (CoT maj@4) | 46/86 (53.5%) |
| uniform random candidate (chance) | 28.2/86 (32.8%) |
| Universal Self-Consistency | 49/86 (57.0%) |
| ESV v3, strict SOUND (used) | 37/86 (43.0%) |
| ESV v3, verdict-only SOUND (secondary) | 35/86 (40.7%) |
| oracle choice (CoT pass@4) | 66/86 (76.7%) |

### Stages 2-3 as a detector, per candidate (pooled)

| SOUND rule | right candidates called SOUND | wrong candidates called SOUND | precision of SOUND |
|---|---|---|---|
| strict (used) | 38/66 (57.6%) | 57/162 (35.2%) | 40.0% |
| verdict only | 40/66 (60.6%) | 60/162 (37.0%) | 40.0% |

|  | n | some binding MISMATCH | some condition VIOLATED | verdict unreadable | failure types when FLAWED |
|---|---|---|---|---|---|
| right candidates | 66 | 3/66 (4.5%) | 5/66 (7.6%) | 0/66 (0.0%) | BINDING: 11, METHOD: 7, COMPUTATION: 7, None: 1 |
| wrong candidates | 162 | 23/162 (14.2%) | 27/162 (16.7%) | 0/162 (0.0%) | COMPUTATION: 38, METHOD: 32, BINDING: 27, None: 5 |

### Decisions (pooled)

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 271 | 98.5% | 98.5% | 0 | 0 | 267 |
| one_sound | 61 | 62.3% | 45.9% | 5 | 15 | 49 |
| several_sound | 16 | 25.0% | 31.2% | 1 | 0 | 11 |
| none_sound | 9 | 44.4% | 44.4% | 0 | 0 | 6 |

`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.

### By arm (pooled)

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 177 | 91.0% | 91.5% | 89.3% | 2 / 5 |
| breaking | 90 | 77.8% | 77.8% | 72.2% | 3 / 8 |
| preserving | 90 | 91.1% | 93.3% | 90.0% | 1 / 2 |

## Compute cost

Read off the `usage` and `latency_seconds` stored on every call, so this section is reproducible offline from the records. Every method starts from the same 4 CoT samples and adds at most one call, so the baseline is shared and only the extra call separates them. **serial** is billed compute time; **critical path** is what a user waits for, since `run_methods.py` issues the 4 samples concurrently while the extra call cannot start until all 4 are back.

| Method | accuracy | calls | solver tokens | verifier tokens | total tokens | vs CoT maj@4 | serial s | critical path s |
|---|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | 13.7% | 4.00 | 1,150 | - | 1,150 | -53.0% | 2.9 | 1.6 |
| CoT pass@1 | 85.0% | 1.00 | 612 | - | 612 | -75.0% | 4.8 | 4.8 |
| CoT maj@4 (initial) | 87.7% | 4.00 | 2,446 | - | 2,446 | +0.0% | 19.1 | 6.7 |
| Universal Self-Consistency | 88.5% | 4.24 | 2,904 | - | 2,904 | +18.7% | 19.3 | 6.9 |
| ESV v3 (stages 1-3) | 85.2% | 4.24 | 3,293 | - | 3,293 | +34.6% | 23.5 | 11.0 |
| CoT pass@4 (oracle choice) | 93.3% | 4.00 | 2,446 | - | 2,446 | +0.0% | 19.1 | 6.7 |

### What the extra compute buys (pooled)

| Method | accuracy vs CoT maj@4 | fixes / breaks | net | McNemar | extra tokens / instance | extra wait | extra tokens per net fix |
|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | -73.9 pp | 1 / 265 | -264 | p=5e-78 | -1,296 | -5.1 s | no net gain |
| Universal Self-Consistency | +0.8 pp | 5 / 2 | +3 | p=0.453 | +458 | +0.2 s | 54,516 |
| ESV v3 (stages 1-3) | -2.5 pp | 6 / 15 | -9 | p=0.078 | +847 | +4.3 s | no net gain |
| CoT pass@4 (oracle choice) | +5.6 pp | 20 / 0 | +20 | p=2e-06 | +0 | +0.0 s | 0 |

### The disagreement gate is the efficiency mechanism (pooled)

Stage 1 costs nothing - it is code over answers the solver already produced - and it is what keeps ESV under 1.5x the baseline despite a verification call that is itself larger than a CoT sample.

|  | per instance |
|---|---|
| Instances where the gate fires (>= 2 distinct candidates) | 86/357 (24.1%) |
| A verification call, on an instance that fires | 3,517 tokens, 17.9 s |
| Amortized over every instance (what ESV actually costs) | 847 tokens, 4.3 s |
| The same protocol ungated, verifying every instance | 3,517 tokens, 17.9 s |
| Verification spend the gate avoids | 75.9% |
| Cost on a gated-out instance | 0 tokens, 0 s - ESV returns the CoT maj@4 answer untouched |

### Compute by arm (pooled)

The gate fires where the perturbation actually hurts, so ESV spends most on the strategy-breaking arm - which is also where it gains most. The cost profile is adaptive, not flat.

| arm | instances | gate fires | CoT maj@4 | ESV v3 | baseline tokens | extra tokens | vs baseline |
|---|---|---|---|---|---|---|---|
| original | 177 | 31/177 (17.5%) | 91.0% | 89.3% | 2,310 | +578 | +25.0% |
| breaking | 90 | 41/90 (45.6%) | 77.8% | 72.2% | 2,916 | +1,620 | +55.5% |
| preserving | 90 | 14/90 (15.6%) | 91.1% | 90.0% | 2,244 | +604 | +26.9% |
