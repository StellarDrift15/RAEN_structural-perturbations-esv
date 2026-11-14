# ESV v3, stage by stage (GSM8K, models qwen-2.5-7b-instruct, mistral-nemo, verifier google/gemma-3-4b-it)

Version `jee_esv3_v1_2026-11-03`, decision rule `stages1-3_strict_v1_2026-11-03`. Gold is used only to score. Design and motivation: `esv_bottleneck_analysis.md`.

The ESV v3 arm is verified by google/gemma-3-4b-it; the USC arm is read from `esv3_records.json`, because USC asks the solver (it has no verifier) and is measured once, in the self-verify run.

## The pipeline, as run

Every method starts from the same 4 CoT samples and none of them re-solves the problem; they differ only in how they pick among answers the solver already produced. ESV fires only on disagreement, so it can damage a correct majority as easily as repair a wrong one, and is scored that way.

| # | Stage | What it does | Calls | Model called | Outcome |
|---|---|---|---|---|---|
| 0 | CoT sampling (shared input, not part of ESV) | The solver answers the problem 4 times with chain-of-thought. Produced once by `run_methods.py` and reused unchanged by every method, so nothing below ever re-solves the problem. | 4 per instance, T = 0.7, seeds 0-3 | solver | 4 answers. Their majority is the `initial` answer and is scored as the CoT maj@4 baseline. |
| 1 | Disagreement gate | Group the 4 answers into distinct candidates by normalized value, in order of first appearance, keeping each one's vote count. Unreadable answers are dropped. | 0 (free) | none | Fewer than 2 distinct candidates -> decision `unanimous`: keep the initial answer and STOP. 2 or more -> go to stage 2. |
| 2 | Contrastive verification | All distinct candidates in ONE call, one representative solution each, in order of first appearance, with vote counts hidden. Per candidate: the strategy, its bindings to quoted givens (MATCHES / MISMATCH), the conditions the method needs (HOLDS / VIOLATED), then the first divergence point, then SOUND or FLAWED with a failure type (BINDING / METHOD / COMPUTATION), the flawed step and the corrected fact. | 1 per disagreeing instance, T = 0 | google/gemma-3-4b-it | A filled-in check per candidate (run, measured below). |
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
| mistralai/mistral-nemo | 383 | 227/383 (59.3%) | 2/227 (0.9%) | 156/383 (40.7%) | 100 | 36 | 20 | 84.9% | 94.3% |
| **pooled** | 773 | 544/773 (70.4%) | 6/544 (1.1%) | 229/773 (29.6%) | 142 | 51 | 36 | 88.0% | 94.6% |

### What the wrong answers are

ESV targets the *trap*: the answer the original strategy S yields when applied blindly to a strategy-breaking P'. The premise holds only where that column is a large share of the failures.

| model | wrong CoT maj@4 answers | on originals | on preserving P' | on breaking P': the TRAP answer | on breaking P': other wrong | CoT maj@4 on originals | CoT maj@4 on breaking P' | CoT samples giving the trap answer |
|---|---|---|---|---|---|---|---|---|
| qwen/qwen-2.5-7b-instruct | 35 | 15/35 (42.9%) | 4/35 (11.4%) | 6/35 (17.1%) | 10/35 (28.6%) | 92.2% | 83.8% | 25/396 (6.3%) |
| mistralai/mistral-nemo | 58 | 22/58 (37.9%) | 10/58 (17.2%) | 11/58 (19.0%) | 15/58 (25.9%) | 88.4% | 72.6% | 34/380 (8.9%) |
| **pooled** | 93 | 37/93 (39.8%) | 14/93 (15.1%) | 17/93 (18.3%) | 25/93 (26.9%) | 90.3% | 78.4% | 59/776 (7.6%) |

## Stages 2-3: candidate verification and choice

**Go / no-go (fixed before the run): NO-GO.** ESV v3 - CoT maj@4 = -2.3 pp (needs >= +3); ESV v3 - USC = -3.0 pp (needs > 0).

### All instances

**qwen/qwen-2.5-7b-instruct** (390 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 18.7% | 3 | 285 | -282 (p=2e-80) |  |
| CoT pass@1 | 88.7% |  |  |  |  |
| CoT maj@4 (initial) | 91.0% |  |  |  |  |
| CoT pass@4 (oracle choice) | 94.9% |  |  |  |  |
| Universal Self-Consistency | 90.8% | 2 | 3 | -1 (p=1.000) | 2 / 0.19 |
| ESV v3 (stages 1-3) | 89.2% | 3 | 10 | -7 (p=0.092) | 197 / 0.19 |

**mistralai/mistral-nemo** (383 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 10.4% | 3 | 288 | -285 (p=2e-81) |  |
| CoT pass@1 | 79.1% |  |  |  |  |
| CoT maj@4 (initial) | 84.9% |  |  |  |  |
| CoT pass@4 (oracle choice) | 94.3% |  |  |  |  |
| Universal Self-Consistency | 86.4% | 11 | 5 | +6 (p=0.210) | 4 / 0.41 |
| ESV v3 (stages 1-3) | 82.0% | 12 | 23 | -11 (p=0.090) | 460 / 0.41 |

**Pooled** (773 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 14.6% | 6 | 573 | -567 (p=5e-161) |  |
| CoT pass@1 | 84.0% |  |  |  |  |
| CoT maj@4 (initial) | 88.0% |  |  |  |  |
| CoT pass@4 (oracle choice) | 94.6% |  |  |  |  |
| Universal Self-Consistency | 88.6% | 13 | 8 | +5 (p=0.383) | 3 / 0.30 |
| ESV v3 (stages 1-3) | 85.6% | 15 | 33 | -18 (p=0.013) | 328 / 0.30 |

ESV v3 vs USC (ESV-only / USC-only right, McNemar p): 14 / 37 (p=0.002)

### Did USC actually select? (pooled)

USC names its pick in one tag. Where that tag cannot be resolved to a response, USC keeps the initial answer and its accuracy becomes CoT maj@4's by construction - so this table has to be read before any USC comparison.

| model | USC calls | by response label | by answer value | ambiguous (label used) | unresolved -> keeps CoT maj@4 |
|---|---|---|---|---|---|
| qwen/qwen-2.5-7b-instruct | 73 | 30/73 (41.1%) | 43/73 (58.9%) | 0/73 (0.0%) | 0/73 (0.0%) |
| mistralai/mistral-nemo | 156 | 105/156 (67.3%) | 48/156 (30.8%) | 3/156 (1.9%) | 0/156 (0.0%) |
| **pooled** | 229 | 135/229 (59.0%) | 91/229 (39.7%) | 3/229 (1.3%) | 0/229 (0.0%) |

### The choice, on disagreeing instances only (pooled)

| Choice among the candidates (229 disagreeing instances) | right |
|---|---|
| majority vote (CoT maj@4) | 142/229 (62.0%) |
| uniform random candidate (chance) | 83.6/229 (36.5%) |
| Universal Self-Consistency | 147/229 (64.2%) |
| ESV v3, strict SOUND (used) | 124/229 (54.1%) |
| ESV v3, verdict-only SOUND (secondary) | 123/229 (53.7%) |
| oracle choice (CoT pass@4) | 193/229 (84.3%) |

### Stages 2-3 as a detector, per candidate (pooled)

| SOUND rule | right candidates called SOUND | wrong candidates called SOUND | precision of SOUND |
|---|---|---|---|
| strict (used) | 114/193 (59.1%) | 108/381 (28.3%) | 51.4% |
| verdict only | 120/193 (62.2%) | 113/381 (29.7%) | 51.5% |

|  | n | some binding MISMATCH | some condition VIOLATED | verdict unreadable | failure types when FLAWED |
|---|---|---|---|---|---|
| right candidates | 193 | 4/193 (2.1%) | 6/193 (3.1%) | 0/193 (0.0%) | COMPUTATION: 39, BINDING: 19, METHOD: 15 |
| wrong candidates | 381 | 18/381 (4.7%) | 15/381 (3.9%) | 0/381 (0.0%) | COMPUTATION: 139, BINDING: 71, METHOD: 58 |

### Decisions (pooled)

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 544 | 98.9% | 98.9% | 0 | 0 | 538 |
| one_sound | 168 | 67.9% | 56.5% | 14 | 33 | 144 |
| several_sound | 26 | 30.8% | 34.6% | 1 | 0 | 22 |
| none_sound | 35 | 57.1% | 57.1% | 0 | 0 | 27 |

`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.

### By arm

Accuracy on the untouched originals, on strategy-preserving P' and on strategy-breaking P', per model and pooled. ESV's premise is that the breaking arm is where a selector has the most to recover, so the fixes / breaks column is read per arm.

**qwen/qwen-2.5-7b-instruct**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 192 | 92.2% | 92.7% | 90.6% | 1 / 4 |
| breaking | 99 | 83.8% | 82.8% | 79.8% | 1 / 5 |
| preserving | 99 | 96.0% | 94.9% | 96.0% | 1 / 1 |

**mistralai/mistral-nemo**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 190 | 88.4% | 88.9% | 87.9% | 6 / 7 |
| breaking | 95 | 72.6% | 73.7% | 65.3% | 3 / 10 |
| preserving | 98 | 89.8% | 93.9% | 86.7% | 3 / 6 |

**Pooled**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 382 | 90.3% | 90.8% | 89.3% | 7 / 11 |
| breaking | 194 | 78.4% | 78.4% | 72.7% | 4 / 15 |
| preserving | 197 | 92.9% | 94.4% | 91.4% | 4 / 7 |

### Decisions, per model

The same decision mix as above, split by model: a checker that almost never returns `one_sound` is abstaining, not selecting.

**qwen/qwen-2.5-7b-instruct**

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 317 | 98.7% | 98.7% | 0 | 0 | 313 |
| one_sound | 59 | 62.7% | 50.8% | 3 | 10 | 48 |
| several_sound | 4 | 50.0% | 50.0% | 0 | 0 | 3 |
| none_sound | 10 | 30.0% | 30.0% | 0 | 0 | 6 |

**mistralai/mistral-nemo**

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 227 | 99.1% | 99.1% | 0 | 0 | 225 |
| one_sound | 109 | 70.6% | 59.6% | 11 | 23 | 96 |
| several_sound | 22 | 27.3% | 31.8% | 1 | 0 | 19 |
| none_sound | 25 | 68.0% | 68.0% | 0 | 0 | 21 |

## Compute cost

Read off the `usage` and `latency_seconds` stored on every call, so this section is reproducible offline from the records. Every method starts from the same 4 CoT samples and adds at most one call, so the baseline is shared and only the extra call separates them. **serial** is billed compute time; **critical path** is what a user waits for, since `run_methods.py` issues the 4 samples concurrently while the extra call cannot start until all 4 are back.

| Method | accuracy | calls | solver tokens | verifier tokens | total tokens | vs CoT maj@4 | serial s | critical path s |
|---|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | 14.6% | 4.00 | 1,024 | - | 1,024 | -48.6% | 1.8 | 0.6 |
| CoT pass@1 | 84.0% | 1.00 | 498 | - | 498 | -75.0% | 3.3 | 3.3 |
| CoT maj@4 (initial) | 88.0% | 4.00 | 1,992 | - | 1,992 | +0.0% | 13.4 | 4.3 |
| Universal Self-Consistency | 88.6% | 4.30 | 2,360 | - | 2,360 | +18.4% | 13.6 | 4.5 |
| ESV v3 (stages 1-3) | 85.6% | 4.30 | 1,992 | 887 | 2,879 | +44.5% | 34.1 | 25.0 |
| CoT pass@4 (oracle choice) | 94.6% | 4.00 | 1,992 | - | 1,992 | +0.0% | 13.4 | 4.3 |

### What the extra compute buys (pooled)

| Method | accuracy vs CoT maj@4 | fixes / breaks | net | McNemar | extra tokens / instance | extra wait | extra tokens per net fix |
|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | -73.4 pp | 6 / 573 | -567 | p=5e-161 | -968 | -3.7 s | no net gain |
| Universal Self-Consistency | +0.6 pp | 13 / 8 | +5 | p=0.383 | +368 | +0.2 s | 56,823 |
| ESV v3 (stages 1-3) | -2.3 pp | 15 / 33 | -18 | p=0.013 | +887 | +20.7 s | no net gain |
| CoT pass@4 (oracle choice) | +6.6 pp | 51 / 0 | +51 | p=9e-16 | +0 | +0.0 s | 0 |

### The disagreement gate is the efficiency mechanism (pooled)

Stage 1 costs nothing - it is code over answers the solver already produced - and it is what keeps ESV under 1.5x the baseline despite a verification call that is itself larger than a CoT sample.

|  | per instance |
|---|---|
| Instances where the gate fires (>= 2 distinct candidates) | 229/773 (29.6%) |
| A verification call, on an instance that fires | 2,995 tokens, 70.0 s |
| Amortized over every instance (what ESV actually costs) | 887 tokens, 20.7 s |
| The same protocol ungated, verifying every instance | 2,995 tokens, 70.0 s |
| Verification spend the gate avoids | 70.4% |
| Cost on a gated-out instance | 0 tokens, 0 s - ESV returns the CoT maj@4 answer untouched |

### Compute by arm (pooled)

The gate fires where the perturbation actually hurts, so ESV spends most on the strategy-breaking arm - which is also where it gains most. The cost profile is adaptive, not flat.

| arm | instances | gate fires | CoT maj@4 | ESV v3 | baseline tokens | extra tokens | vs baseline |
|---|---|---|---|---|---|---|---|
| original | 382 | 90/382 (23.6%) | 90.3% | 89.3% | 1,918 | +665 | +34.7% |
| breaking | 194 | 95/194 (49.0%) | 78.4% | 72.7% | 2,259 | +1,606 | +71.1% |
| preserving | 197 | 44/197 (22.3%) | 92.9% | 91.4% | 1,874 | +611 | +32.6% |

The ESV check is billed by google/gemma-3-4b-it, the 4 samples by the solver, so a flat token count understates the check. k = the verifier's price per token over the solver's; totals are in solver-token equivalents. No price list is assumed - read off the row for your k.

| k | ESV tokens / instance | vs CoT maj@4 | check vs USC's overhead | per net fix |
|---|---|---|---|---|
| 1 | 2,879 | +44.5% | 2.4x | - |
| 2 | 3,767 | +89.1% | 4.8x | - |
| 5 | 6,429 | +222.7% | 12.1x | - |
| 10 | 10,865 | +445.4% | 24.1x | - |
| 20 | 19,738 | +890.8% | 48.3x | - |
