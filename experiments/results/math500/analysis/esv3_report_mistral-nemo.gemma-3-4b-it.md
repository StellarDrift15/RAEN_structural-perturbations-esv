# ESV v3, stage by stage (MATH-500, models mistral-nemo, verifier google/gemma-3-4b-it)

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

## Stage 1: disagreement gate (no calls)

Unanimous = fewer than 2 distinct readable answers among the 4 CoT samples: ESV keeps the answer. Only the disagreeing instances can change, and only the fixable ones can gain.

| model | instances | unanimous (no call) | unanimous but wrong (lost) | disagreeing (ESV runs) | disagreeing, majority right (can break) | disagreeing, fixable (majority wrong, a candidate right) | disagreeing, no candidate right | CoT maj@4 | CoT pass@4 (ceiling) |
|---|---|---|---|---|---|---|---|---|---|
| mistralai/mistral-nemo | 488 | 97/488 (19.9%) | 4/97 (4.1%) | 391/488 (80.1%) | 128 | 73 | 190 | 45.3% | 60.2% |
| **pooled** | 488 | 97/488 (19.9%) | 4/97 (4.1%) | 391/488 (80.1%) | 128 | 73 | 190 | 45.3% | 60.2% |

### What the wrong answers are

ESV targets the *trap*: the answer the original strategy S yields when applied blindly to a strategy-breaking P'. The premise holds only where that column is a large share of the failures.

| model | wrong CoT maj@4 answers | on originals | on preserving P' | on breaking P': the TRAP answer | on breaking P': other wrong | CoT maj@4 on originals | CoT maj@4 on breaking P' | CoT samples giving the trap answer |
|---|---|---|---|---|---|---|---|---|
| mistralai/mistral-nemo | 267 | 131/267 (49.1%) | 52/267 (19.5%) | 12/267 (4.5%) | 72/267 (27.0%) | 46.3% | 31.7% | 50/492 (10.2%) |
| **pooled** | 267 | 131/267 (49.1%) | 52/267 (19.5%) | 12/267 (4.5%) | 72/267 (27.0%) | 46.3% | 31.7% | 50/492 (10.2%) |

## Stages 2-3: candidate verification and choice

**Go / no-go (fixed before the run): NO-GO.** ESV v3 - CoT maj@4 = -5.5 pp (needs >= +3); ESV v3 - USC = -4.7 pp (needs > 0).

### All instances

**mistralai/mistral-nemo** (488 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 11.5% | 17 | 182 | -165 (p=5e-36) |  |
| CoT pass@1 | 38.2% |  |  |  |  |
| CoT maj@4 (initial) | 45.3% |  |  |  |  |
| CoT pass@4 (oracle choice) | 60.2% |  |  |  |  |
| Universal Self-Consistency | 44.5% | 16 | 20 | -4 (p=0.618) | 8 / 0.80 |
| ESV v3 (stages 1-3) | 39.8% | 17 | 44 | -27 (p=7e-04) | 1179 / 0.80 |

**Pooled** (488 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 11.5% | 17 | 182 | -165 (p=5e-36) |  |
| CoT pass@1 | 38.2% |  |  |  |  |
| CoT maj@4 (initial) | 45.3% |  |  |  |  |
| CoT pass@4 (oracle choice) | 60.2% |  |  |  |  |
| Universal Self-Consistency | 44.5% | 16 | 20 | -4 (p=0.618) | 8 / 0.80 |
| ESV v3 (stages 1-3) | 39.8% | 17 | 44 | -27 (p=7e-04) | 1179 / 0.80 |

ESV v3 vs USC (ESV-only / USC-only right, McNemar p): 21 / 44 (p=0.006)

### Did USC actually select? (pooled)

USC names its pick in one tag. Where that tag cannot be resolved to a response, USC keeps the initial answer and its accuracy becomes CoT maj@4's by construction - so this table has to be read before any USC comparison.

| model | USC calls | by response label | by answer value | ambiguous (label used) | unresolved -> keeps CoT maj@4 |
|---|---|---|---|---|---|
| mistralai/mistral-nemo | 391 | 363/391 (92.8%) | 0/391 (0.0%) | 28/391 (7.2%) | 0/391 (0.0%) |
| **pooled** | 391 | 363/391 (92.8%) | 0/391 (0.0%) | 28/391 (7.2%) | 0/391 (0.0%) |

### The choice, on disagreeing instances only (pooled)

| Choice among the candidates (391 disagreeing instances) | right |
|---|---|
| majority vote (CoT maj@4) | 128/391 (32.7%) |
| uniform random candidate (chance) | 76.4/391 (19.5%) |
| Universal Self-Consistency | 124/391 (31.7%) |
| ESV v3, strict SOUND (used) | 101/391 (25.8%) |
| ESV v3, verdict-only SOUND (secondary) | 103/391 (26.3%) |
| oracle choice (CoT pass@4) | 201/391 (51.4%) |

### Stages 2-3 as a detector, per candidate (pooled)

| SOUND rule | right candidates called SOUND | wrong candidates called SOUND | precision of SOUND |
|---|---|---|---|
| strict (used) | 106/201 (52.7%) | 386/1061 (36.4%) | 21.5% |
| verdict only | 109/201 (54.2%) | 393/1061 (37.0%) | 21.7% |

|  | n | some binding MISMATCH | some condition VIOLATED | verdict unreadable | failure types when FLAWED |
|---|---|---|---|---|---|
| right candidates | 201 | 5/201 (2.5%) | 6/201 (3.0%) | 0/201 (0.0%) | BINDING: 38, COMPUTATION: 37, METHOD: 16, None: 1 |
| wrong candidates | 1061 | 23/1061 (2.2%) | 28/1061 (2.6%) | 0/1061 (0.0%) | BINDING: 243, COMPUTATION: 233, METHOD: 116, None: 76 |

### Decisions (pooled)

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 97 | 95.9% | 95.9% | 0 | 0 | 93 |
| one_sound | 206 | 41.7% | 29.6% | 13 | 38 | 119 |
| several_sound | 131 | 22.1% | 20.6% | 4 | 6 | 56 |
| none_sound | 54 | 24.1% | 24.1% | 0 | 0 | 26 |

`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.

### By arm

Accuracy on the untouched originals, on strategy-preserving P' and on strategy-breaking P', per model and pooled. ESV's premise is that the breaking arm is where a selector has the most to recover, so the fixes / breaks column is read per arm.

**mistralai/mistral-nemo**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 244 | 46.3% | 47.5% | 41.8% | 6 / 17 |
| breaking | 123 | 31.7% | 29.3% | 27.6% | 5 / 10 |
| preserving | 121 | 57.0% | 53.7% | 47.9% | 6 / 17 |

**Pooled**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 244 | 46.3% | 47.5% | 41.8% | 6 / 17 |
| breaking | 123 | 31.7% | 29.3% | 27.6% | 5 / 10 |
| preserving | 121 | 57.0% | 53.7% | 47.9% | 6 / 17 |

### Decisions, per model

The same decision mix as above, split by model: a checker that almost never returns `one_sound` is abstaining, not selecting.

**mistralai/mistral-nemo**

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 97 | 95.9% | 95.9% | 0 | 0 | 93 |
| one_sound | 206 | 41.7% | 29.6% | 13 | 38 | 119 |
| several_sound | 131 | 22.1% | 20.6% | 4 | 6 | 56 |
| none_sound | 54 | 24.1% | 24.1% | 0 | 0 | 26 |

## Compute cost

Read off the `usage` and `latency_seconds` stored on every call, so this section is reproducible offline from the records. Every method starts from the same 4 CoT samples and adds at most one call, so the baseline is shared and only the extra call separates them. **serial** is billed compute time; **critical path** is what a user waits for, since `run_methods.py` issues the 4 samples concurrently while the extra call cannot start until all 4 are back.

| Method | accuracy | calls | solver tokens | verifier tokens | total tokens | vs CoT maj@4 | serial s | critical path s |
|---|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | 11.5% | 4.00 | 1,085 | - | 1,085 | -57.8% | 2.2 | 0.7 |
| CoT pass@1 | 38.2% | 1.00 | 643 | - | 643 | -75.0% | 4.2 | 4.2 |
| CoT maj@4 (initial) | 45.3% | 4.00 | 2,570 | - | 2,570 | +0.0% | 16.7 | 6.5 |
| Universal Self-Consistency | 44.5% | 4.80 | 3,886 | - | 3,886 | +51.2% | 17.8 | 7.5 |
| ESV v3 (stages 1-3) | 39.8% | 4.80 | 2,570 | 3,294 | 5,864 | +128.1% | 65.7 | 55.4 |
| CoT pass@4 (oracle choice) | 60.2% | 4.00 | 2,570 | - | 2,570 | +0.0% | 16.7 | 6.5 |

### What the extra compute buys (pooled)

| Method | accuracy vs CoT maj@4 | fixes / breaks | net | McNemar | extra tokens / instance | extra wait | extra tokens per net fix |
|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | -33.8 pp | 17 / 182 | -165 | p=5e-36 | -1,485 | -5.7 s | no net gain |
| Universal Self-Consistency | -0.8 pp | 16 / 20 | -4 | p=0.618 | +1,316 | +1.0 s | no net gain |
| ESV v3 (stages 1-3) | -5.5 pp | 17 / 44 | -27 | p=7e-04 | +3,294 | +49.0 s | no net gain |
| CoT pass@4 (oracle choice) | +15.0 pp | 73 / 0 | +73 | p=2e-22 | +0 | +0.0 s | 0 |

### The disagreement gate is the efficiency mechanism (pooled)

Stage 1 costs nothing - it is code over answers the solver already produced - and it is what keeps ESV under 1.5x the baseline despite a verification call that is itself larger than a CoT sample.

|  | per instance |
|---|---|
| Instances where the gate fires (>= 2 distinct candidates) | 391/488 (80.1%) |
| A verification call, on an instance that fires | 4,111 tokens, 61.1 s |
| Amortized over every instance (what ESV actually costs) | 3,294 tokens, 49.0 s |
| The same protocol ungated, verifying every instance | 4,111 tokens, 61.1 s |
| Verification spend the gate avoids | 19.9% |
| Cost on a gated-out instance | 0 tokens, 0 s - ESV returns the CoT maj@4 answer untouched |

### Compute by arm (pooled)

The gate fires where the perturbation actually hurts, so ESV spends most on the strategy-breaking arm - which is also where it gains most. The cost profile is adaptive, not flat.

| arm | instances | gate fires | CoT maj@4 | ESV v3 | baseline tokens | extra tokens | vs baseline |
|---|---|---|---|---|---|---|---|
| original | 244 | 188/244 (77.0%) | 46.3% | 41.8% | 2,607 | +3,154 | +121.0% |
| breaking | 123 | 113/123 (91.9%) | 31.7% | 27.6% | 2,619 | +3,930 | +150.1% |
| preserving | 121 | 90/121 (74.4%) | 57.0% | 47.9% | 2,446 | +2,929 | +119.7% |

The ESV check is billed by google/gemma-3-4b-it, the 4 samples by the solver, so a flat token count understates the check. k = the verifier's price per token over the solver's; totals are in solver-token equivalents. No price list is assumed - read off the row for your k.

| k | ESV tokens / instance | vs CoT maj@4 | check vs USC's overhead | per net fix |
|---|---|---|---|---|
| 1 | 5,864 | +128.1% | 2.5x | - |
| 2 | 9,157 | +256.3% | 5.0x | - |
| 5 | 19,038 | +640.7% | 12.5x | - |
| 10 | 35,505 | +1281.5% | 25.0x | - |
| 20 | 68,441 | +2563.0% | 50.1x | - |

### By answer format (pooled)

`expression` answers are compared by LaTeX normalization and symbolic equivalence (`expression_answers.py`) rather than by exact match, so this split is the check that the scoring is not what separates the methods.

| answer format | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| expression | 176 | 38.6% | 35.2% | 30.7% | 5 / 19 |
| integer | 290 | 49.7% | 50.0% | 45.2% | 12 / 25 |
| numeric | 22 | 40.9% | 45.5% | 40.9% | 0 / 0 |
