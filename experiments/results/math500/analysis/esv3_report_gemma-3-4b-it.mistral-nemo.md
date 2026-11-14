# ESV v3, stage by stage (MATH-500, models gemma-3-4b-it, verifier mistralai/mistral-nemo)

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

## Stage 1: disagreement gate (no calls)

Unanimous = fewer than 2 distinct readable answers among the 4 CoT samples: ESV keeps the answer. Only the disagreeing instances can change, and only the fixable ones can gain.

| model | instances | unanimous (no call) | unanimous but wrong (lost) | disagreeing (ESV runs) | disagreeing, majority right (can break) | disagreeing, fixable (majority wrong, a candidate right) | disagreeing, no candidate right | CoT maj@4 | CoT pass@4 (ceiling) |
|---|---|---|---|---|---|---|---|---|---|
| google/gemma-3-4b-it | 488 | 278/488 (57.0%) | 11/278 (4.0%) | 210/488 (43.0%) | 62 | 48 | 100 | 67.4% | 77.3% |
| **pooled** | 488 | 278/488 (57.0%) | 11/278 (4.0%) | 210/488 (43.0%) | 62 | 48 | 100 | 67.4% | 77.3% |

### What the wrong answers are

ESV targets the *trap*: the answer the original strategy S yields when applied blindly to a strategy-breaking P'. The premise holds only where that column is a large share of the failures.

| model | wrong CoT maj@4 answers | on originals | on preserving P' | on breaking P': the TRAP answer | on breaking P': other wrong | CoT maj@4 on originals | CoT maj@4 on breaking P' | CoT samples giving the trap answer |
|---|---|---|---|---|---|---|---|---|
| google/gemma-3-4b-it | 159 | 76/159 (47.8%) | 29/159 (18.2%) | 12/159 (7.5%) | 42/159 (26.4%) | 68.9% | 56.1% | 41/492 (8.3%) |
| **pooled** | 159 | 76/159 (47.8%) | 29/159 (18.2%) | 12/159 (7.5%) | 42/159 (26.4%) | 68.9% | 56.1% | 41/492 (8.3%) |

## Stages 2-3: candidate verification and choice

**Go / no-go (fixed before the run): NO-GO.** ESV v3 - CoT maj@4 = -1.6 pp (needs >= +3); ESV v3 - USC = -3.3 pp (needs > 0).

### All instances

**google/gemma-3-4b-it** (488 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 9.8% | 5 | 286 | -281 (p=9e-78) |  |
| CoT pass@1 | 65.0% |  |  |  |  |
| CoT maj@4 (initial) | 67.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 77.3% |  |  |  |  |
| Universal Self-Consistency | 69.1% | 18 | 10 | +8 (p=0.185) | 6 / 0.43 |
| ESV v3 (stages 1-3) | 65.8% | 12 | 20 | -8 (p=0.215) | 370 / 0.43 |

**Pooled** (488 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 9.8% | 5 | 286 | -281 (p=9e-78) |  |
| CoT pass@1 | 65.0% |  |  |  |  |
| CoT maj@4 (initial) | 67.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 77.3% |  |  |  |  |
| Universal Self-Consistency | 69.1% | 18 | 10 | +8 (p=0.185) | 6 / 0.43 |
| ESV v3 (stages 1-3) | 65.8% | 12 | 20 | -8 (p=0.215) | 370 / 0.43 |

ESV v3 vs USC (ESV-only / USC-only right, McNemar p): 16 / 32 (p=0.029)

### Did USC actually select? (pooled)

USC names its pick in one tag. Where that tag cannot be resolved to a response, USC keeps the initial answer and its accuracy becomes CoT maj@4's by construction - so this table has to be read before any USC comparison.

| model | USC calls | by response label | by answer value | ambiguous (label used) | unresolved -> keeps CoT maj@4 |
|---|---|---|---|---|---|
| google/gemma-3-4b-it | 210 | 198/210 (94.3%) | 1/210 (0.5%) | 11/210 (5.2%) | 0/210 (0.0%) |
| **pooled** | 210 | 198/210 (94.3%) | 1/210 (0.5%) | 11/210 (5.2%) | 0/210 (0.0%) |

### The choice, on disagreeing instances only (pooled)

| Choice among the candidates (210 disagreeing instances) | right |
|---|---|
| majority vote (CoT maj@4) | 62/210 (29.5%) |
| uniform random candidate (chance) | 43.7/210 (20.8%) |
| Universal Self-Consistency | 70/210 (33.3%) |
| ESV v3, strict SOUND (used) | 54/210 (25.7%) |
| ESV v3, verdict-only SOUND (secondary) | 54/210 (25.7%) |
| oracle choice (CoT pass@4) | 110/210 (52.4%) |

### Stages 2-3 as a detector, per candidate (pooled)

| SOUND rule | right candidates called SOUND | wrong candidates called SOUND | precision of SOUND |
|---|---|---|---|
| strict (used) | 60/110 (54.5%) | 218/491 (44.4%) | 21.6% |
| verdict only | 62/110 (56.4%) | 222/491 (45.2%) | 21.8% |

|  | n | some binding MISMATCH | some condition VIOLATED | verdict unreadable | failure types when FLAWED |
|---|---|---|---|---|---|
| right candidates | 110 | 5/110 (4.5%) | 4/110 (3.6%) | 0/110 (0.0%) | METHOD: 24, COMPUTATION: 21, BINDING: 3 |
| wrong candidates | 491 | 30/491 (6.1%) | 28/491 (5.7%) | 0/491 (0.0%) | METHOD: 131, COMPUTATION: 87, BINDING: 51 |

### Decisions (pooled)

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 278 | 96.0% | 96.0% | 0 | 0 | 267 |
| one_sound | 101 | 39.6% | 31.7% | 10 | 18 | 59 |
| several_sound | 77 | 14.3% | 14.3% | 2 | 2 | 37 |
| none_sound | 32 | 34.4% | 34.4% | 0 | 0 | 14 |

`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.

### By arm

Accuracy on the untouched originals, on strategy-preserving P' and on strategy-breaking P', per model and pooled. ESV's premise is that the breaking arm is where a selector has the most to recover, so the fixes / breaks column is read per arm.

**google/gemma-3-4b-it**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 244 | 68.9% | 69.7% | 67.6% | 7 / 10 |
| breaking | 123 | 56.1% | 60.2% | 56.9% | 4 / 3 |
| preserving | 121 | 76.0% | 76.9% | 71.1% | 1 / 7 |

**Pooled**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 244 | 68.9% | 69.7% | 67.6% | 7 / 10 |
| breaking | 123 | 56.1% | 60.2% | 56.9% | 4 / 3 |
| preserving | 121 | 76.0% | 76.9% | 71.1% | 1 / 7 |

### Decisions, per model

The same decision mix as above, split by model: a checker that almost never returns `one_sound` is abstaining, not selecting.

**google/gemma-3-4b-it**

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 278 | 96.0% | 96.0% | 0 | 0 | 267 |
| one_sound | 101 | 39.6% | 31.7% | 10 | 18 | 59 |
| several_sound | 77 | 14.3% | 14.3% | 2 | 2 | 37 |
| none_sound | 32 | 34.4% | 34.4% | 0 | 0 | 14 |

## Compute cost

Read off the `usage` and `latency_seconds` stored on every call, so this section is reproducible offline from the records. Every method starts from the same 4 CoT samples and adds at most one call, so the baseline is shared and only the extra call separates them. **serial** is billed compute time; **critical path** is what a user waits for, since `run_methods.py` issues the 4 samples concurrently while the extra call cannot start until all 4 are back.

| Method | accuracy | calls | solver tokens | verifier tokens | total tokens | vs CoT maj@4 | serial s | critical path s |
|---|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | 9.8% | 4.00 | 2,798 | - | 2,798 | -57.7% | 28.5 | 16.5 |
| CoT pass@1 | 65.0% | 1.00 | 1,653 | - | 1,653 | -75.0% | 35.6 | 35.6 |
| CoT maj@4 (initial) | 67.4% | 4.00 | 6,614 | - | 6,614 | +0.0% | 142.3 | 57.2 |
| Universal Self-Consistency | 69.1% | 4.43 | 8,432 | - | 8,432 | +27.5% | 142.8 | 57.7 |
| ESV v3 (stages 1-3) | 65.8% | 4.43 | 6,614 | 2,206 | 8,820 | +33.4% | 147.9 | 62.8 |
| CoT pass@4 (oracle choice) | 77.3% | 4.00 | 6,614 | - | 6,614 | +0.0% | 142.3 | 57.2 |

### What the extra compute buys (pooled)

| Method | accuracy vs CoT maj@4 | fixes / breaks | net | McNemar | extra tokens / instance | extra wait | extra tokens per net fix |
|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | -57.6 pp | 5 / 286 | -281 | p=9e-78 | -3,816 | -40.6 s | no net gain |
| Universal Self-Consistency | +1.6 pp | 18 / 10 | +8 | p=0.185 | +1,818 | +0.5 s | 110,893 |
| ESV v3 (stages 1-3) | -1.6 pp | 12 / 20 | -8 | p=0.215 | +2,206 | +5.6 s | no net gain |
| CoT pass@4 (oracle choice) | +9.8 pp | 48 / 0 | +48 | p=7e-15 | +0 | +0.0 s | 0 |

### The disagreement gate is the efficiency mechanism (pooled)

Stage 1 costs nothing - it is code over answers the solver already produced - and it is what keeps ESV under 1.5x the baseline despite a verification call that is itself larger than a CoT sample.

|  | per instance |
|---|---|
| Instances where the gate fires (>= 2 distinct candidates) | 210/488 (43.0%) |
| A verification call, on an instance that fires | 5,127 tokens, 13.0 s |
| Amortized over every instance (what ESV actually costs) | 2,206 tokens, 5.6 s |
| The same protocol ungated, verifying every instance | 5,127 tokens, 13.0 s |
| Verification spend the gate avoids | 57.0% |
| Cost on a gated-out instance | 0 tokens, 0 s - ESV returns the CoT maj@4 answer untouched |

### Compute by arm (pooled)

The gate fires where the perturbation actually hurts, so ESV spends most on the strategy-breaking arm - which is also where it gains most. The cost profile is adaptive, not flat.

| arm | instances | gate fires | CoT maj@4 | ESV v3 | baseline tokens | extra tokens | vs baseline |
|---|---|---|---|---|---|---|---|
| original | 244 | 103/244 (42.2%) | 68.9% | 67.6% | 6,309 | +2,051 | +32.5% |
| breaking | 123 | 65/123 (52.8%) | 56.1% | 56.9% | 7,616 | +2,907 | +38.2% |
| preserving | 121 | 42/121 (34.7%) | 76.0% | 71.1% | 6,209 | +1,807 | +29.1% |

The ESV check is billed by mistralai/mistral-nemo, the 4 samples by the solver, so a flat token count understates the check. k = the verifier's price per token over the solver's; totals are in solver-token equivalents. No price list is assumed - read off the row for your k.

| k | ESV tokens / instance | vs CoT maj@4 | check vs USC's overhead | per net fix |
|---|---|---|---|---|
| 1 | 8,820 | +33.4% | 1.2x | - |
| 2 | 11,026 | +66.7% | 2.4x | - |
| 5 | 17,645 | +166.8% | 6.1x | - |
| 10 | 28,677 | +333.6% | 12.1x | - |
| 20 | 50,741 | +667.2% | 24.3x | - |

### By answer format (pooled)

`expression` answers are compared by LaTeX normalization and symbolic equivalence (`expression_answers.py`) rather than by exact match, so this split is the check that the scoring is not what separates the methods.

| answer format | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| expression | 176 | 42.6% | 43.8% | 41.5% | 6 / 8 |
| integer | 290 | 81.4% | 82.8% | 79.3% | 6 / 12 |
| numeric | 22 | 81.8% | 90.9% | 81.8% | 0 / 0 |
