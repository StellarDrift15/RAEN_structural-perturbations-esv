# ESV v3, stage by stage (GSM8K, models mistral-nemo, gemma-3-4b-it, verifier openai/gpt-oss-120b)

Version `jee_esv3_v1_2026-11-03`, decision rule `stages1-3_strict_v1_2026-11-03`. Gold is used only to score. Design and motivation: `esv_bottleneck_analysis.md`.

The ESV v3 arm is verified by openai/gpt-oss-120b; the USC arm is read from `esv3_records.json`, because USC asks the solver (it has no verifier) and is measured once, in the self-verify run.

## The pipeline, as run

Every method starts from the same 4 CoT samples and none of them re-solves the problem; they differ only in how they pick among answers the solver already produced. ESV fires only on disagreement, so it can damage a correct majority as easily as repair a wrong one, and is scored that way.

| # | Stage | What it does | Calls | Model called | Outcome |
|---|---|---|---|---|---|
| 0 | CoT sampling (shared input, not part of ESV) | The solver answers the problem 4 times with chain-of-thought. Produced once by `run_methods.py` and reused unchanged by every method, so nothing below ever re-solves the problem. | 4 per instance, T = 0.7, seeds 0-3 | solver | 4 answers. Their majority is the `initial` answer and is scored as the CoT maj@4 baseline. |
| 1 | Disagreement gate | Group the 4 answers into distinct candidates by normalized value, in order of first appearance, keeping each one's vote count. Unreadable answers are dropped. | 0 (free) | none | Fewer than 2 distinct candidates -> decision `unanimous`: keep the initial answer and STOP. 2 or more -> go to stage 2. |
| 2 | Contrastive verification | All distinct candidates in ONE call, one representative solution each, in order of first appearance, with vote counts hidden. Per candidate: the strategy, its bindings to quoted givens (MATCHES / MISMATCH), the conditions the method needs (HOLDS / VIOLATED), then the first divergence point, then SOUND or FLAWED with a failure type (BINDING / METHOD / COMPUTATION), the flawed step and the corrected fact. | 1 per disagreeing instance, T = 0 | openai/gpt-oss-120b | A filled-in check per candidate (run, measured below). |
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
| mistralai/mistral-nemo | 390 | 227/390 (58.2%) | 2/227 (0.9%) | 163/390 (41.8%) | 104 | 39 | 20 | 84.4% | 94.4% |
| google/gemma-3-4b-it | 390 | 323/390 (82.8%) | 11/323 (3.4%) | 67/390 (17.2%) | 39 | 17 | 11 | 90.0% | 94.4% |
| **pooled** | 780 | 550/780 (70.5%) | 13/550 (2.4%) | 230/780 (29.5%) | 143 | 56 | 31 | 87.2% | 94.4% |

### What the wrong answers are

ESV targets the *trap*: the answer the original strategy S yields when applied blindly to a strategy-breaking P'. The premise holds only where that column is a large share of the failures.

| model | wrong CoT maj@4 answers | on originals | on preserving P' | on breaking P': the TRAP answer | on breaking P': other wrong | CoT maj@4 on originals | CoT maj@4 on breaking P' | CoT samples giving the trap answer |
|---|---|---|---|---|---|---|---|---|
| mistralai/mistral-nemo | 61 | 23/61 (37.7%) | 11/61 (18.0%) | 11/61 (18.0%) | 16/61 (26.2%) | 88.0% | 72.7% | 35/396 (8.8%) |
| google/gemma-3-4b-it | 39 | 16/39 (41.0%) | 4/39 (10.3%) | 8/39 (20.5%) | 11/39 (28.2%) | 91.7% | 80.8% | 30/396 (7.6%) |
| **pooled** | 100 | 39/100 (39.0%) | 15/100 (15.0%) | 19/100 (19.0%) | 27/100 (27.0%) | 89.8% | 76.8% | 65/792 (8.2%) |

## Stages 2-3: candidate verification and choice

**Go / no-go (fixed before the run): GO.** ESV v3 - CoT maj@4 = +5.8 pp (needs >= +3); ESV v3 - USC = +4.4 pp (needs > 0).

### All instances

**mistralai/mistral-nemo** (390 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 10.5% | 3 | 291 | -288 (p=3e-82) |  |
| CoT pass@1 | 78.7% |  |  |  |  |
| CoT maj@4 (initial) | 84.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 94.4% |  |  |  |  |
| Universal Self-Consistency | 85.9% | 11 | 5 | +6 (p=0.210) | 4 / 0.42 |
| Judge-select (same checker, no ESV form) | 93.3% | 36 | 1 | +35 (p=6e-10) | 291 / 0.42 |
| ESV v3 (stages 1-3) | 92.8% | 35 | 2 | +33 (p=1e-08) | 824 / 0.42 |

**google/gemma-3-4b-it** (390 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 7.2% | 1 | 324 | -323 (p=1e-95) |  |
| CoT pass@1 | 88.1% |  |  |  |  |
| CoT maj@4 (initial) | 90.0% |  |  |  |  |
| CoT pass@4 (oracle choice) | 94.4% |  |  |  |  |
| Universal Self-Consistency | 91.3% | 6 | 1 | +5 (p=0.125) | 3 / 0.17 |
| Judge-select (same checker, no ESV form) | 94.4% | 17 | 0 | +17 (p=2e-05) | 128 / 0.17 |
| ESV v3 (stages 1-3) | 93.1% | 12 | 0 | +12 (p=5e-04) | 357 / 0.17 |

**Pooled** (780 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 8.8% | 4 | 615 | -611 (p=6e-177) |  |
| CoT pass@1 | 83.4% |  |  |  |  |
| CoT maj@4 (initial) | 87.2% |  |  |  |  |
| CoT pass@4 (oracle choice) | 94.4% |  |  |  |  |
| Universal Self-Consistency | 88.6% | 17 | 6 | +11 (p=0.035) | 4 / 0.29 |
| Judge-select (same checker, no ESV form) | 93.8% | 53 | 1 | +52 (p=6e-15) | 209 / 0.29 |
| ESV v3 (stages 1-3) | 92.9% | 47 | 2 | +45 (p=4e-12) | 590 / 0.29 |

ESV v3 vs USC (ESV-only / USC-only right, McNemar p): 37 / 3 (p=2e-08)

### Did USC actually select? (pooled)

USC names its pick in one tag. Where that tag cannot be resolved to a response, USC keeps the initial answer and its accuracy becomes CoT maj@4's by construction - so this table has to be read before any USC comparison.

| model | USC calls | by response label | by answer value | ambiguous (label used) | unresolved -> keeps CoT maj@4 |
|---|---|---|---|---|---|
| mistralai/mistral-nemo | 163 | 111/163 (68.1%) | 49/163 (30.1%) | 3/163 (1.8%) | 0/163 (0.0%) |
| google/gemma-3-4b-it | 67 | 49/67 (73.1%) | 9/67 (13.4%) | 0/67 (0.0%) | 9/67 (13.4%) |
| **pooled** | 230 | 160/230 (69.6%) | 58/230 (25.2%) | 3/230 (1.3%) | 9/230 (3.9%) |

### The choice, on disagreeing instances only (pooled)

| Choice among the candidates (230 disagreeing instances) | right |
|---|---|
| majority vote (CoT maj@4) | 143/230 (62.2%) |
| uniform random candidate (chance) | 86.7/230 (37.7%) |
| Universal Self-Consistency | 154/230 (67.0%) |
| ESV v3, strict SOUND (used) | 188/230 (81.7%) |
| ESV v3, verdict-only SOUND (secondary) | 188/230 (81.7%) |
| oracle choice (CoT pass@4) | 199/230 (86.5%) |

### Stages 2-3 as a detector, per candidate (pooled)

| SOUND rule | right candidates called SOUND | wrong candidates called SOUND | precision of SOUND |
|---|---|---|---|
| strict (used) | 171/199 (85.9%) | 6/376 (1.6%) | 96.6% |
| verdict only | 171/199 (85.9%) | 6/376 (1.6%) | 96.6% |

|  | n | some binding MISMATCH | some condition VIOLATED | verdict unreadable | failure types when FLAWED |
|---|---|---|---|---|---|
| right candidates | 199 | 17/199 (8.5%) | 10/199 (5.0%) | 0/199 (0.0%) | BINDING: 20, COMPUTATION: 4, METHOD: 4 |
| wrong candidates | 376 | 252/376 (67.0%) | 187/376 (49.7%) | 0/376 (0.0%) | BINDING: 222, COMPUTATION: 103, METHOD: 45 |

### Decisions (pooled)

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 550 | 97.6% | 97.6% | 0 | 0 | 537 |
| one_sound | 177 | 71.2% | 96.6% | 47 | 2 | 174 |
| none_sound | 53 | 32.1% | 32.1% | 0 | 0 | 25 |

`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.

### By arm

Accuracy on the untouched originals, on strategy-preserving P' and on strategy-breaking P', per model and pooled. ESV's premise is that the breaking arm is where a selector has the most to recover, so the fixes / breaks column is read per arm.

**mistralai/mistral-nemo**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 192 | 88.0% | 88.5% | 93.8% | 12 / 1 |
| breaking | 99 | 72.7% | 73.7% | 85.9% | 13 / 0 |
| preserving | 99 | 88.9% | 92.9% | 98.0% | 10 / 1 |

**google/gemma-3-4b-it**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 192 | 91.7% | 93.2% | 94.3% | 5 / 0 |
| breaking | 99 | 80.8% | 82.8% | 87.9% | 7 / 0 |
| preserving | 99 | 96.0% | 96.0% | 96.0% | 0 / 0 |

**Pooled**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 384 | 89.8% | 90.9% | 94.0% | 17 / 1 |
| breaking | 198 | 76.8% | 78.3% | 86.9% | 20 / 0 |
| preserving | 198 | 92.4% | 94.4% | 97.0% | 10 / 1 |

### Decisions, per model

The same decision mix as above, split by model: a checker that almost never returns `one_sound` is abstaining, not selecting.

**mistralai/mistral-nemo**

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 227 | 99.1% | 99.1% | 0 | 0 | 225 |
| one_sound | 130 | 72.3% | 97.7% | 35 | 2 | 130 |
| none_sound | 33 | 30.3% | 30.3% | 0 | 0 | 13 |

**google/gemma-3-4b-it**

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 323 | 96.6% | 96.6% | 0 | 0 | 312 |
| one_sound | 47 | 68.1% | 93.6% | 12 | 0 | 44 |
| none_sound | 20 | 35.0% | 35.0% | 0 | 0 | 12 |

## The judge-select control: is it the structure, or the checker?

Judge-select asks the **same checker** to read the **same candidates** in the same order with the same vote counts hidden, reason freely in its own words, and name one solution or NONE. Everything is held fixed except ESV's form (bindings / conditions / divergence / verdict per candidate). It keeps the licence to reject every candidate, so ESV gets no free win from its `none_sound` branch, and it is given room to reason, so this is structured reasoning versus unstructured reasoning rather than versus no thinking at all.

**ESV v3 vs judge-select, paired over all 780 instances: 0 / 7 (McNemar p=0.016).**

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 8.8% | 4 | 615 | -611 (p=6e-177) |  |
| CoT pass@1 | 83.4% |  |  |  |  |
| CoT maj@4 (initial) | 87.2% |  |  |  |  |
| CoT pass@4 (oracle choice) | 94.4% |  |  |  |  |
| Universal Self-Consistency | 88.6% | 17 | 6 | +11 (p=0.035) | 4 / 0.29 |
| Judge-select (same checker, no ESV form) | 93.8% | 53 | 1 | +52 (p=6e-15) | 209 / 0.29 |
| ESV v3 (stages 1-3) | 92.9% | 47 | 2 | +45 (p=4e-12) | 590 / 0.29 |

### Where the two differ (disagreeing instances only)

Stratified by what ESV decided, because the hypothesis is specific: the form should matter least where one candidate is clearly sound, and most where none survives and judge-select has to commit anyway.

| ESV's decision | n | CoT maj@4 | judge-select | ESV v3 | oracle | ESV-only / judge-only right |
|---|---|---|---|---|---|---|
| one_sound | 177 | 71.2% | 97.7% | 96.6% | 98.3% | 0 / 2 (p=0.500) |
| none_sound | 53 | 32.1% | 41.5% | 32.1% | 47.2% | 0 / 5 (p=0.062) |

### Selection quality on the disagreeing instances

|  | correct |
|---|---|
| CoT maj@4 (initial) | 62.2% |
| Universal Self-Consistency | 67.0% |
| Judge-select (same checker, no ESV form) | 84.8% |
| ESV v3 (stages 1-3) | 81.7% |
| CoT pass@4 (oracle choice) | 86.5% |

### What judge-select actually returned

`none` is a judgement that no candidate holds up - the counterpart of ESV's `none_sound`. `unresolved` is a tag that named nothing usable; both keep the initial answer, but only the first is a decision, so they are reported apart.

| selection | instances | correct |
|---|---|---|
| label | 197/230 (85.7%) | 97.0% |
| none | 31/230 (13.5%) | 6.5% |
| ambiguous | 2/230 (0.9%) | 100.0% |

## Compute cost

Read off the `usage` and `latency_seconds` stored on every call, so this section is reproducible offline from the records. Every method starts from the same 4 CoT samples and adds at most one call, so the baseline is shared and only the extra call separates them. **serial** is billed compute time; **critical path** is what a user waits for, since `run_methods.py` issues the 4 samples concurrently while the extra call cannot start until all 4 are back.

| Method | accuracy | calls | solver tokens | verifier tokens | total tokens | vs CoT maj@4 | serial s | critical path s |
|---|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | 8.8% | 4.00 | 1,095 | - | 1,095 | -57.1% | 2.9 | 1.3 |
| CoT pass@1 | 83.4% | 1.00 | 639 | - | 639 | -75.0% | 5.7 | 5.7 |
| CoT maj@4 (initial) | 87.2% | 4.00 | 2,556 | - | 2,556 | +0.0% | 22.7 | 8.0 |
| Universal Self-Consistency | 88.6% | 4.29 | 3,055 | - | 3,055 | +19.5% | 23.0 | 8.3 |
| Judge-select (same checker, no ESV form) | 93.8% | 4.29 | 2,556 | 575 | 3,131 | +22.5% | 27.0 | 12.2 |
| ESV v3 (stages 1-3) | 92.9% | 4.29 | 2,556 | 1,189 | 3,745 | +46.5% | 31.1 | 16.3 |
| CoT pass@4 (oracle choice) | 94.4% | 4.00 | 2,556 | - | 2,556 | +0.0% | 22.7 | 8.0 |

### What the extra compute buys (pooled)

| Method | accuracy vs CoT maj@4 | fixes / breaks | net | McNemar | extra tokens / instance | extra wait | extra tokens per net fix |
|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | -78.3 pp | 4 / 615 | -611 | p=6e-177 | -1,460 | -6.7 s | no net gain |
| Universal Self-Consistency | +1.4 pp | 17 / 6 | +11 | p=0.035 | +499 | +0.3 s | 35,391 |
| Judge-select (same checker, no ESV form) | +6.7 pp | 53 / 1 | +52 | p=6e-15 | +575 | +4.3 s | 8,626 |
| ESV v3 (stages 1-3) | +5.8 pp | 47 / 2 | +45 | p=4e-12 | +1,189 | +8.3 s | 20,611 |
| CoT pass@4 (oracle choice) | +7.2 pp | 56 / 0 | +56 | p=3e-17 | +0 | +0.0 s | 0 |

### The disagreement gate is the efficiency mechanism (pooled)

Stage 1 costs nothing - it is code over answers the solver already produced - and it is what keeps ESV under 1.5x the baseline despite a verification call that is itself larger than a CoT sample.

|  | per instance |
|---|---|
| Instances where the gate fires (>= 2 distinct candidates) | 230/780 (29.5%) |
| A verification call, on an instance that fires | 4,033 tokens, 28.3 s |
| Amortized over every instance (what ESV actually costs) | 1,189 tokens, 8.3 s |
| The same protocol ungated, verifying every instance | 4,033 tokens, 28.3 s |
| Verification spend the gate avoids | 70.5% |
| Cost on a gated-out instance | 0 tokens, 0 s - ESV returns the CoT maj@4 answer untouched |

### Compute by arm (pooled)

The gate fires where the perturbation actually hurts, so ESV spends most on the strategy-breaking arm - which is also where it gains most. The cost profile is adaptive, not flat.

| arm | instances | gate fires | CoT maj@4 | ESV v3 | baseline tokens | extra tokens | vs baseline |
|---|---|---|---|---|---|---|---|
| original | 384 | 90/384 (23.4%) | 89.8% | 94.0% | 2,435 | +875 | +35.9% |
| breaking | 198 | 96/198 (48.5%) | 76.8% | 86.9% | 3,027 | +2,204 | +72.8% |
| preserving | 198 | 44/198 (22.2%) | 92.4% | 97.0% | 2,318 | +783 | +33.8% |

The ESV check is billed by openai/gpt-oss-120b, the 4 samples by the solver, so a flat token count understates the check. k = the verifier's price per token over the solver's; totals are in solver-token equivalents. No price list is assumed - read off the row for your k.

| k | ESV tokens / instance | vs CoT maj@4 | check vs USC's overhead | per net fix |
|---|---|---|---|---|
| 1 | 3,745 | +46.5% | 2.4x | 20,611 |
| 2 | 4,934 | +93.1% | 4.8x | 41,223 |
| 5 | 8,501 | +232.6% | 11.9x | 103,056 |
| 10 | 14,447 | +465.3% | 23.8x | 206,113 |
| 20 | 26,338 | +930.5% | 47.7x | 412,225 |

Break-even against USC: ESV v3 costs less per net fix than USC for any k below **1.7**. Above it USC is the cheaper way to buy a fix - but USC's total gain is capped at +11 net fixes against ESV's +45, so the two are not interchangeable at any price: more USC cannot reach ESV's accuracy.
