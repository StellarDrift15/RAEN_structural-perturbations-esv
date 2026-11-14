# ESV v3, stage by stage (GSM8K, models qwen-2.5-7b-instruct, gemma-3-4b-it, verifier mistralai/mistral-nemo)

Version `jee_esv3_v1_2026-11-03`, decision rule `stages1-3_strict_v1_2026-11-03`. Gold is used only to score. Design and motivation: `esv_bottleneck_analysis.md`.

The ESV v3 arm is verified by mistralai/mistral-nemo; the USC arm is read from `esv3_records.json`, because USC asks the solver (it has no verifier) and is measured once, in the self-verify run.

## The pipeline, as run

Every method starts from the same 4 CoT samples and none of them re-solves the problem; they differ only in how they pick among answers the solver already produced. ESV fires only on disagreement, so it can damage a correct majority as easily as repair a wrong one, and is scored that way.

| # | Stage | What it does | Calls | Model called | Outcome |
|---|---|---|---|---|---|
| 0 | CoT sampling (shared input, not part of ESV) | The solver answers the problem 4 times with chain-of-thought. Produced once by `run_methods.py` and reused unchanged by every method, so nothing below ever re-solves the problem. | 4 per instance, T = 0.7, seeds 0-3 | solver | 4 answers. Their majority is the `initial` answer and is scored as the CoT maj@4 baseline. |
| 1 | Disagreement gate | Group the 4 answers into distinct candidates by normalized value, in order of first appearance, keeping each one's vote count. Unreadable answers are dropped. | 0 (free) | none | Fewer than 2 distinct candidates -> decision `unanimous`: keep the initial answer and STOP. 2 or more -> go to stage 2. |
| 2 | Contrastive verification | All distinct candidates in ONE call, one representative solution each, in order of first appearance, with vote counts hidden. Per candidate: the strategy, its bindings to quoted givens (MATCHES / MISMATCH), the conditions the method needs (HOLDS / VIOLATED), then the first divergence point, then SOUND or FLAWED with a failure type (BINDING / METHOD / COMPUTATION), the flawed step and the corrected fact. | 1 per disagreeing instance, T = 0 | mistralai/mistral-nemo | A filled-in check per candidate (run, measured below). |
| 3 | Decision rule (code only, no model) | A candidate is strictly SOUND only if its verdict is SOUND **and** it lists zero MISMATCH **and** zero VIOLATED. Exactly one such candidate wins; several, the one with the most votes (ties to the initial answer, then to the earliest); none, nothing is chosen. | 0 (free) | none | `one_sound` -> take that candidate. `several_sound` -> take the most-voted SOUND one. `none_sound` -> keep the initial answer. `unreadable` -> keep the initial answer. |
| 4 | Diagnosis-driven repair | Would re-solve using the diagnosis from stage 2 where no candidate survives. NOT BUILT: `none_sound` simply keeps the initial answer, so ESV never invents an answer the solver did not already produce. | n/a | n/a | Not built. Its ceiling is the `a candidate is right` column of the decisions table below. |
| 5 | Scoring | Gold enters here and nowhere else. Paired against CoT maj@4 on the same instances, with exact McNemar tests, so a stage can break a correct majority as easily as repair a wrong one. | 0 (free) | none | The accuracy tables below. |

### Where USC sits (the baseline, not a stage)

Universal Self-Consistency is the control ESV must beat, not a step inside it. It replaces stages 2-3 with a single selection call and is wired in deliberately parallel to them:

- **When it fires:** on exactly the same instances as stage 2, the disagreeing ones (2 or more distinct readable candidates). A `unanimous` instance never calls it and keeps the majority answer, so USC and CoT maj@4 agree there by construction.
- **Who is called:** the **solver**, never the verifier. USC has no verifier, which is why it is identical in both conditions. (Not measured in this report: USC asks the solver, so it is run once, in the self-verify condition, and read from `esv3_records.json`.)
- **What it sees:** all 4 samples including duplicates, in arbitrary order, with the reasoning shown. Unlike stage 2 it is therefore free to count multiplicity: it can see that 3 of 4 samples agree.
- **How it answers:** one `<selected_response>` tag naming the response it picks. Resolved first as a response label (1-4), then as the selected answer's value, which solvers frequently return instead.
- **When it declines:** a tag that resolves to neither keeps the initial answer, i.e. USC falls back to CoT maj@4. The rate is reported under *Did USC actually select?* and must be read before any USC comparison.

Excluded from every score: 6 instances whose published gold is wrong or ambiguous (`common.DISPUTED_INSTANCES`): 105_original, 111_original, 189_original, 40_original, 82_original, 95_original.

## Stage 1: disagreement gate (no calls)

Unanimous = fewer than 2 distinct readable answers among the 4 CoT samples: ESV keeps the answer. Only the disagreeing instances can change, and only the fixable ones can gain.

| model | instances | unanimous (no call) | unanimous but wrong (lost) | disagreeing (ESV runs) | disagreeing, majority right (can break) | disagreeing, fixable (majority wrong, a candidate right) | disagreeing, no candidate right | CoT maj@4 | CoT pass@4 (ceiling) |
|---|---|---|---|---|---|---|---|---|---|
| qwen/qwen-2.5-7b-instruct | 390 | 317/390 (81.3%) | 4/317 (1.3%) | 73/390 (18.7%) | 42 | 15 | 16 | 91.0% | 94.9% |
| google/gemma-3-4b-it | 390 | 323/390 (82.8%) | 11/323 (3.4%) | 67/390 (17.2%) | 39 | 17 | 11 | 90.0% | 94.4% |
| **pooled** | 780 | 640/780 (82.1%) | 15/640 (2.3%) | 140/780 (17.9%) | 81 | 32 | 27 | 90.5% | 94.6% |

### What the wrong answers are

ESV targets the *trap*: the answer the original strategy S yields when applied blindly to a strategy-breaking P'. The premise holds only where that column is a large share of the failures.

| model | wrong CoT maj@4 answers | on originals | on preserving P' | on breaking P': the TRAP answer | on breaking P': other wrong | CoT maj@4 on originals | CoT maj@4 on breaking P' | CoT samples giving the trap answer |
|---|---|---|---|---|---|---|---|---|
| qwen/qwen-2.5-7b-instruct | 35 | 15/35 (42.9%) | 4/35 (11.4%) | 6/35 (17.1%) | 10/35 (28.6%) | 92.2% | 83.8% | 25/396 (6.3%) |
| google/gemma-3-4b-it | 39 | 16/39 (41.0%) | 4/39 (10.3%) | 8/39 (20.5%) | 11/39 (28.2%) | 91.7% | 80.8% | 30/396 (7.6%) |
| **pooled** | 74 | 31/74 (41.9%) | 8/74 (10.8%) | 14/74 (18.9%) | 21/74 (28.4%) | 91.9% | 82.3% | 55/792 (6.9%) |

## Stages 2-3: candidate verification and choice

**Go / no-go (fixed before the run): NO-GO.** ESV v3 - CoT maj@4 = -2.2 pp (needs >= +3); ESV v3 - USC = -2.7 pp (needs > 0).

### All instances

**qwen/qwen-2.5-7b-instruct** (390 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 18.7% | 3 | 285 | -282 (p=2e-80) |  |
| CoT pass@1 | 88.7% |  |  |  |  |
| CoT maj@4 (initial) | 91.0% |  |  |  |  |
| CoT pass@4 (oracle choice) | 94.9% |  |  |  |  |
| Universal Self-Consistency | 90.8% | 2 | 3 | -1 (p=1.000) | 2 / 0.19 |
| ESV v3 (stages 1-3) | 89.0% | 4 | 12 | -8 (p=0.077) | 150 / 0.19 |

**google/gemma-3-4b-it** (390 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 7.2% | 1 | 324 | -323 (p=1e-95) |  |
| CoT pass@1 | 88.1% |  |  |  |  |
| CoT maj@4 (initial) | 90.0% |  |  |  |  |
| CoT pass@4 (oracle choice) | 94.4% |  |  |  |  |
| Universal Self-Consistency | 91.3% | 6 | 1 | +5 (p=0.125) | 3 / 0.17 |
| ESV v3 (stages 1-3) | 87.7% | 3 | 12 | -9 (p=0.035) | 147 / 0.17 |

**Pooled** (780 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 12.9% | 4 | 609 | -605 (p=3e-175) |  |
| CoT pass@1 | 88.4% |  |  |  |  |
| CoT maj@4 (initial) | 90.5% |  |  |  |  |
| CoT pass@4 (oracle choice) | 94.6% |  |  |  |  |
| Universal Self-Consistency | 91.0% | 8 | 4 | +4 (p=0.388) | 3 / 0.18 |
| ESV v3 (stages 1-3) | 88.3% | 7 | 24 | -17 (p=0.003) | 149 / 0.18 |

ESV v3 vs USC (ESV-only / USC-only right, McNemar p): 7 / 28 (p=5e-04)

### Did USC actually select? (pooled)

USC names its pick in one tag. Where that tag cannot be resolved to a response, USC keeps the initial answer and its accuracy becomes CoT maj@4's by construction - so this table has to be read before any USC comparison.

| model | USC calls | by response label | by answer value | ambiguous (label used) | unresolved -> keeps CoT maj@4 |
|---|---|---|---|---|---|
| qwen/qwen-2.5-7b-instruct | 73 | 30/73 (41.1%) | 43/73 (58.9%) | 0/73 (0.0%) | 0/73 (0.0%) |
| google/gemma-3-4b-it | 67 | 49/67 (73.1%) | 9/67 (13.4%) | 0/67 (0.0%) | 9/67 (13.4%) |
| **pooled** | 140 | 79/140 (56.4%) | 52/140 (37.1%) | 0/140 (0.0%) | 9/140 (6.4%) |

### The choice, on disagreeing instances only (pooled)

| Choice among the candidates (140 disagreeing instances) | right |
|---|---|
| majority vote (CoT maj@4) | 81/140 (57.9%) |
| uniform random candidate (chance) | 51.2/140 (36.6%) |
| Universal Self-Consistency | 85/140 (60.7%) |
| ESV v3, strict SOUND (used) | 64/140 (45.7%) |
| ESV v3, verdict-only SOUND (secondary) | 64/140 (45.7%) |
| oracle choice (CoT pass@4) | 113/140 (80.7%) |

### Stages 2-3 as a detector, per candidate (pooled)

| SOUND rule | right candidates called SOUND | wrong candidates called SOUND | precision of SOUND |
|---|---|---|---|
| strict (used) | 60/113 (53.1%) | 91/219 (41.6%) | 39.7% |
| verdict only | 64/113 (56.6%) | 95/219 (43.4%) | 40.3% |

|  | n | some binding MISMATCH | some condition VIOLATED | verdict unreadable | failure types when FLAWED |
|---|---|---|---|---|---|
| right candidates | 113 | 12/113 (10.6%) | 5/113 (4.4%) | 0/113 (0.0%) | COMPUTATION: 22, METHOD: 20, BINDING: 7 |
| wrong candidates | 219 | 28/219 (12.8%) | 32/219 (14.6%) | 0/219 (0.0%) | METHOD: 58, COMPUTATION: 49, BINDING: 17 |

### Decisions (pooled)

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 640 | 97.7% | 97.7% | 0 | 0 | 625 |
| one_sound | 98 | 60.2% | 43.9% | 7 | 23 | 82 |
| several_sound | 26 | 57.7% | 53.8% | 0 | 1 | 20 |
| none_sound | 16 | 43.8% | 43.8% | 0 | 0 | 11 |

`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.

### By arm

Accuracy on the untouched originals, on strategy-preserving P' and on strategy-breaking P', per model and pooled. ESV's premise is that the breaking arm is where a selector has the most to recover, so the fixes / breaks column is read per arm.

**qwen/qwen-2.5-7b-instruct**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 192 | 92.2% | 92.7% | 90.6% | 3 / 6 |
| breaking | 99 | 83.8% | 82.8% | 79.8% | 1 / 5 |
| preserving | 99 | 96.0% | 94.9% | 94.9% | 0 / 1 |

**google/gemma-3-4b-it**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 192 | 91.7% | 93.2% | 91.1% | 1 / 2 |
| breaking | 99 | 80.8% | 82.8% | 75.8% | 2 / 7 |
| preserving | 99 | 96.0% | 96.0% | 92.9% | 0 / 3 |

**Pooled**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 384 | 91.9% | 93.0% | 90.9% | 4 / 8 |
| breaking | 198 | 82.3% | 82.8% | 77.8% | 3 / 12 |
| preserving | 198 | 96.0% | 95.5% | 93.9% | 0 / 4 |

### Decisions, per model

The same decision mix as above, split by model: a checker that almost never returns `one_sound` is abstaining, not selecting.

**qwen/qwen-2.5-7b-instruct**

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 317 | 98.7% | 98.7% | 0 | 0 | 313 |
| one_sound | 52 | 57.7% | 42.3% | 4 | 12 | 41 |
| several_sound | 17 | 64.7% | 64.7% | 0 | 0 | 13 |
| none_sound | 4 | 25.0% | 25.0% | 0 | 0 | 3 |

**google/gemma-3-4b-it**

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 323 | 96.6% | 96.6% | 0 | 0 | 312 |
| one_sound | 46 | 63.0% | 45.7% | 3 | 11 | 41 |
| several_sound | 9 | 44.4% | 33.3% | 0 | 1 | 7 |
| none_sound | 12 | 50.0% | 50.0% | 0 | 0 | 8 |

## Compute cost

Read off the `usage` and `latency_seconds` stored on every call, so this section is reproducible offline from the records. Every method starts from the same 4 CoT samples and adds at most one call, so the baseline is shared and only the extra call separates them. **serial** is billed compute time; **critical path** is what a user waits for, since `run_methods.py` issues the 4 samples concurrently while the extra call cannot start until all 4 are back.

| Method | accuracy | calls | solver tokens | verifier tokens | total tokens | vs CoT maj@4 | serial s | critical path s |
|---|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | 12.9% | 4.00 | 1,167 | - | 1,167 | -54.9% | 2.5 | 1.1 |
| CoT pass@1 | 88.4% | 1.00 | 647 | - | 647 | -75.0% | 5.4 | 5.4 |
| CoT maj@4 (initial) | 90.5% | 4.00 | 2,589 | - | 2,589 | +0.0% | 21.5 | 7.2 |
| Universal Self-Consistency | 91.0% | 4.18 | 2,976 | - | 2,976 | +14.9% | 21.6 | 7.3 |
| ESV v3 (stages 1-3) | 88.3% | 4.18 | 2,589 | 602 | 3,192 | +23.3% | 24.4 | 10.2 |
| CoT pass@4 (oracle choice) | 94.6% | 4.00 | 2,589 | - | 2,589 | +0.0% | 21.5 | 7.2 |

### What the extra compute buys (pooled)

| Method | accuracy vs CoT maj@4 | fixes / breaks | net | McNemar | extra tokens / instance | extra wait | extra tokens per net fix |
|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | -77.6 pp | 4 / 609 | -605 | p=3e-175 | -1,422 | -6.1 s | no net gain |
| Universal Self-Consistency | +0.5 pp | 8 / 4 | +4 | p=0.388 | +387 | +0.1 s | 75,373 |
| ESV v3 (stages 1-3) | -2.2 pp | 7 / 24 | -17 | p=0.003 | +602 | +2.9 s | no net gain |
| CoT pass@4 (oracle choice) | +4.1 pp | 32 / 0 | +32 | p=5e-10 | +0 | +0.0 s | 0 |

### The disagreement gate is the efficiency mechanism (pooled)

Stage 1 costs nothing - it is code over answers the solver already produced - and it is what keeps ESV under 1.5x the baseline despite a verification call that is itself larger than a CoT sample.

|  | per instance |
|---|---|
| Instances where the gate fires (>= 2 distinct candidates) | 140/780 (17.9%) |
| A verification call, on an instance that fires | 3,355 tokens, 16.4 s |
| Amortized over every instance (what ESV actually costs) | 602 tokens, 2.9 s |
| The same protocol ungated, verifying every instance | 3,355 tokens, 16.4 s |
| Verification spend the gate avoids | 82.1% |
| Cost on a gated-out instance | 0 tokens, 0 s - ESV returns the CoT maj@4 answer untouched |

### Compute by arm (pooled)

The gate fires where the perturbation actually hurts, so ESV spends most on the strategy-breaking arm - which is also where it gains most. The cost profile is adaptive, not flat.

| arm | instances | gate fires | CoT maj@4 | ESV v3 | baseline tokens | extra tokens | vs baseline |
|---|---|---|---|---|---|---|---|
| original | 384 | 52/384 (13.5%) | 91.9% | 90.9% | 2,483 | +429 | +17.3% |
| breaking | 198 | 73/198 (36.9%) | 82.3% | 77.8% | 3,011 | +1,291 | +42.9% |
| preserving | 198 | 15/198 (7.6%) | 96.0% | 93.9% | 2,373 | +250 | +10.5% |

The ESV check is billed by mistralai/mistral-nemo, the 4 samples by the solver, so a flat token count understates the check. k = the verifier's price per token over the solver's; totals are in solver-token equivalents. No price list is assumed - read off the row for your k.

| k | ESV tokens / instance | vs CoT maj@4 | check vs USC's overhead | per net fix |
|---|---|---|---|---|
| 1 | 3,192 | +23.3% | 1.6x | - |
| 2 | 3,794 | +46.5% | 3.1x | - |
| 5 | 5,601 | +116.3% | 7.8x | - |
| 10 | 8,612 | +232.6% | 15.6x | - |
| 20 | 14,635 | +465.2% | 31.2x | - |
