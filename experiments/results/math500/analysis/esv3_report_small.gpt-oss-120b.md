# ESV v3, stage by stage (MATH-500, small models, verifier openai/gpt-oss-120b)

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

**Go / no-go (fixed before the run): GO.** ESV v3 - CoT maj@4 = +5.9 pp (needs >= +3); ESV v3 - USC = +5.5 pp (needs > 0).

### All instances

**mistralai/mistral-nemo** (488 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 11.5% | 17 | 182 | -165 (p=5e-36) |  |
| CoT pass@1 | 38.2% |  |  |  |  |
| CoT maj@4 (initial) | 45.3% |  |  |  |  |
| CoT pass@4 (oracle choice) | 60.2% |  |  |  |  |
| Universal Self-Consistency | 44.5% | 16 | 20 | -4 (p=0.618) | 8 / 0.80 |
| Judge-select (same checker, no ESV form) | 56.6% | 58 | 3 | +55 (p=3e-14) | 1061 / 0.80 |
| ESV v3 (stages 1-3) | 51.6% | 32 | 1 | +31 (p=8e-09) | 2309 / 0.80 |

**google/gemma-3-4b-it** (488 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 9.8% | 5 | 286 | -281 (p=9e-78) |  |
| CoT pass@1 | 65.0% |  |  |  |  |
| CoT maj@4 (initial) | 67.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 77.3% |  |  |  |  |
| Universal Self-Consistency | 69.1% | 18 | 10 | +8 (p=0.185) | 6 / 0.43 |
| Judge-select (same checker, no ESV form) | 75.4% | 41 | 2 | +39 (p=2e-10) | 728 / 0.43 |
| ESV v3 (stages 1-3) | 73.0% | 27 | 0 | +27 (p=1e-08) | 1219 / 0.43 |

**Pooled** (976 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 10.7% | 22 | 468 | -446 (p=6e-110) |  |
| CoT pass@1 | 51.6% |  |  |  |  |
| CoT maj@4 (initial) | 56.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 68.8% |  |  |  |  |
| Universal Self-Consistency | 56.8% | 34 | 30 | +4 (p=0.708) | 7 / 0.62 |
| Judge-select (same checker, no ESV form) | 66.0% | 99 | 5 | +94 (p=1e-23) | 895 / 0.62 |
| ESV v3 (stages 1-3) | 62.3% | 59 | 1 | +58 (p=1e-16) | 1764 / 0.62 |

ESV v3 vs USC (ESV-only / USC-only right, McNemar p): 71 / 17 (p=5e-09)

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
| ESV v3, strict SOUND (used) | 248/601 (41.3%) |
| ESV v3, verdict-only SOUND (secondary) | 248/601 (41.3%) |
| oracle choice (CoT pass@4) | 311/601 (51.7%) |

### Stages 2-3 as a detector, per candidate (pooled)

| SOUND rule | right candidates called SOUND | wrong candidates called SOUND | precision of SOUND |
|---|---|---|---|
| strict (used) | 189/311 (60.8%) | 80/1552 (5.2%) | 70.3% |
| verdict only | 191/311 (61.4%) | 80/1552 (5.2%) | 70.5% |

|  | n | some binding MISMATCH | some condition VIOLATED | verdict unreadable | failure types when FLAWED |
|---|---|---|---|---|---|
| right candidates | 311 | 57/311 (18.3%) | 66/311 (21.2%) | 1/311 (0.3%) | BINDING: 45, METHOD: 39, COMPUTATION: 35 |
| wrong candidates | 1552 | 842/1552 (54.3%) | 947/1552 (61.0%) | 3/1552 (0.2%) | BINDING: 613, COMPUTATION: 463, METHOD: 393 |

### Decisions (pooled)

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 375 | 96.0% | 96.0% | 0 | 0 | 360 |
| one_sound | 187 | 59.4% | 89.3% | 57 | 1 | 171 |
| several_sound | 34 | 38.2% | 44.1% | 2 | 0 | 23 |
| none_sound | 379 | 17.4% | 17.4% | 0 | 0 | 116 |
| unreadable | 1 | 0.0% | 0.0% | 0 | 0 | 1 |

`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.

### By arm (pooled)

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 488 | 57.6% | 58.6% | 64.3% | 33 / 0 |
| breaking | 246 | 43.9% | 44.7% | 50.0% | 16 / 1 |
| preserving | 242 | 66.5% | 65.3% | 70.7% | 10 / 0 |

## The judge-select control: is it the structure, or the checker?

Judge-select asks the **same checker** to read the **same candidates** in the same order with the same vote counts hidden, reason freely in its own words, and name one solution or NONE. Everything is held fixed except ESV's form (bindings / conditions / divergence / verdict per candidate). It keeps the licence to reject every candidate, so ESV gets no free win from its `none_sound` branch, and it is given room to reason, so this is structured reasoning versus unstructured reasoning rather than versus no thinking at all.

**ESV v3 vs judge-select, paired over all 976 instances: 4 / 40 (McNemar p=2e-08).**

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 10.7% | 22 | 468 | -446 (p=6e-110) |  |
| CoT pass@1 | 51.6% |  |  |  |  |
| CoT maj@4 (initial) | 56.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 68.8% |  |  |  |  |
| Universal Self-Consistency | 56.8% | 34 | 30 | +4 (p=0.708) | 7 / 0.62 |
| Judge-select (same checker, no ESV form) | 66.0% | 99 | 5 | +94 (p=1e-23) | 895 / 0.62 |
| ESV v3 (stages 1-3) | 62.3% | 59 | 1 | +58 (p=1e-16) | 1764 / 0.62 |

### Where the two differ (disagreeing instances only)

Stratified by what ESV decided, because the hypothesis is specific: the form should matter least where one candidate is clearly sound, and most where none survives and judge-select has to commit anyway.

| ESV's decision | n | CoT maj@4 | judge-select | ESV v3 | oracle | ESV-only / judge-only right |
|---|---|---|---|---|---|---|
| one_sound | 187 | 59.4% | 89.3% | 89.3% | 91.4% | 0 / 0 (p=-) |
| several_sound | 34 | 38.2% | 47.1% | 44.1% | 67.6% | 2 / 3 (p=1.000) |
| none_sound | 379 | 17.4% | 26.4% | 17.4% | 30.6% | 2 / 36 (p=5e-09) |
| unreadable | 1 | 0.0% | 100.0% | 0.0% | 100.0% | 0 / 1 (p=1.000) |

### Selection quality on the disagreeing instances

|  | correct |
|---|---|
| CoT maj@4 (initial) | 31.6% |
| Universal Self-Consistency | 32.3% |
| Judge-select (same checker, no ESV form) | 47.3% |
| ESV v3 (stages 1-3) | 41.3% |
| CoT pass@4 (oracle choice) | 51.7% |

### What judge-select actually returned

`none` is a judgement that no candidate holds up - the counterpart of ESV's `none_sound`. `unresolved` is a tag that named nothing usable; both keep the initial answer, but only the first is a decision, so they are reported apart.

| selection | instances | correct |
|---|---|---|
| none | 299/601 (49.8%) | 4.0% |
| label | 278/601 (46.3%) | 89.6% |
| ambiguous | 24/601 (4.0%) | 95.8% |

## Compute cost

Read off the `usage` and `latency_seconds` stored on every call, so this section is reproducible offline from the records. Every method starts from the same 4 CoT samples and adds at most one call, so the baseline is shared and only the extra call separates them. **serial** is billed compute time; **critical path** is what a user waits for, since `run_methods.py` issues the 4 samples concurrently while the extra call cannot start until all 4 are back.

| Method | accuracy | calls | solver tokens | verifier tokens | total tokens | vs CoT maj@4 | serial s | critical path s |
|---|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | 10.7% | 4.00 | 1,942 | - | 1,942 | -57.7% | 15.3 | 8.6 |
| CoT pass@1 | 51.6% | 1.00 | 1,148 | - | 1,148 | -75.0% | 19.9 | 19.9 |
| CoT maj@4 (initial) | 56.4% | 4.00 | 4,592 | - | 4,592 | +0.0% | 79.5 | 31.8 |
| Universal Self-Consistency | 56.8% | 4.62 | 6,159 | - | 6,159 | +34.1% | 80.3 | 32.6 |
| Judge-select (same checker, no ESV form) | 66.0% | 4.62 | 4,592 | 2,234 | 6,826 | +48.7% | 92.4 | 44.7 |
| ESV v3 (stages 1-3) | 62.3% | 4.62 | 4,592 | 3,807 | 8,399 | +82.9% | 120.6 | 72.9 |
| CoT pass@4 (oracle choice) | 68.8% | 4.00 | 4,592 | - | 4,592 | +0.0% | 79.5 | 31.8 |

### What the extra compute buys (pooled)

| Method | accuracy vs CoT maj@4 | fixes / breaks | net | McNemar | extra tokens / instance | extra wait | extra tokens per net fix |
|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | -45.7 pp | 22 / 468 | -446 | p=6e-110 | -2,650 | -23.2 s | no net gain |
| Universal Self-Consistency | +0.4 pp | 34 / 30 | +4 | p=0.708 | +1,567 | +0.8 s | 382,293 |
| Judge-select (same checker, no ESV form) | +9.6 pp | 99 / 5 | +94 | p=1e-23 | +2,234 | +12.9 s | 23,197 |
| ESV v3 (stages 1-3) | +5.9 pp | 59 / 1 | +58 | p=1e-16 | +3,807 | +41.1 s | 64,060 |
| CoT pass@4 (oracle choice) | +12.4 pp | 121 / 0 | +121 | p=8e-37 | +0 | +0.0 s | 0 |

### The disagreement gate is the efficiency mechanism (pooled)

Stage 1 costs nothing - it is code over answers the solver already produced - and it is what keeps ESV under 1.5x the baseline despite a verification call that is itself larger than a CoT sample.

|  | per instance |
|---|---|
| Instances where the gate fires (>= 2 distinct candidates) | 601/976 (61.6%) |
| A verification call, on an instance that fires | 6,182 tokens, 66.7 s |
| Amortized over every instance (what ESV actually costs) | 3,807 tokens, 41.1 s |
| The same protocol ungated, verifying every instance | 6,182 tokens, 66.7 s |
| Verification spend the gate avoids | 38.4% |
| Cost on a gated-out instance | 0 tokens, 0 s - ESV returns the CoT maj@4 answer untouched |

### Compute by arm (pooled)

The gate fires where the perturbation actually hurts, so ESV spends most on the strategy-breaking arm - which is also where it gains most. The cost profile is adaptive, not flat.

| arm | instances | gate fires | CoT maj@4 | ESV v3 | baseline tokens | extra tokens | vs baseline |
|---|---|---|---|---|---|---|---|
| original | 488 | 291/488 (59.6%) | 57.6% | 64.3% | 4,458 | +3,600 | +80.8% |
| breaking | 246 | 178/246 (72.4%) | 43.9% | 50.0% | 5,117 | +4,883 | +95.4% |
| preserving | 242 | 132/242 (54.5%) | 66.5% | 70.7% | 4,327 | +3,129 | +72.3% |

The ESV check is billed by openai/gpt-oss-120b, the 4 samples by the solver, so a flat token count understates the check. k = the verifier's price per token over the solver's; totals are in solver-token equivalents. No price list is assumed - read off the row for your k.

| k | ESV tokens / instance | vs CoT maj@4 | check vs USC's overhead | per net fix |
|---|---|---|---|---|
| 1 | 8,399 | +82.9% | 2.4x | 64,060 |
| 2 | 12,206 | +165.8% | 4.9x | 128,119 |
| 5 | 23,626 | +414.5% | 12.1x | 320,298 |
| 10 | 42,660 | +829.0% | 24.3x | 640,596 |
| 20 | 80,728 | +1658.1% | 48.6x | 1,281,191 |

Break-even against USC: ESV v3 costs less per net fix than USC for any k below **6.0**. Above it USC is the cheaper way to buy a fix - but USC's total gain is capped at +4 net fixes against ESV's +58, so the two are not interchangeable at any price: more USC cannot reach ESV's accuracy.

### By answer format (pooled)

`expression` answers are compared by LaTeX normalization and symbolic equivalence (`expression_answers.py`) rather than by exact match, so this split is the check that the scoring is not what separates the methods.

| answer format | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| expression | 352 | 40.6% | 39.5% | 46.0% | 20 / 1 |
| integer | 580 | 65.5% | 66.4% | 71.9% | 37 / 0 |
| numeric | 44 | 61.4% | 68.2% | 65.9% | 2 / 0 |
