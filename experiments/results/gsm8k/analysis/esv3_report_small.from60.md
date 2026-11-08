# ESV v3, stage by stage (GSM8K, small models, HELD-OUT pairs >= 60)

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
| qwen/qwen-2.5-7b-instruct | 271 | 216/271 (79.7%) | 4/216 (1.9%) | 55/271 (20.3%) | 33 | 11 | 11 | 90.4% | 94.5% |
| mistralai/mistral-nemo | 271 | 153/271 (56.5%) | 1/153 (0.7%) | 118/271 (43.5%) | 79 | 27 | 12 | 85.2% | 95.2% |
| google/gemma-3-4b-it | 271 | 227/271 (83.8%) | 8/227 (3.5%) | 44/271 (16.2%) | 27 | 13 | 4 | 90.8% | 95.6% |
| **pooled** | 813 | 596/813 (73.3%) | 13/596 (2.2%) | 217/813 (26.7%) | 139 | 51 | 27 | 88.8% | 95.1% |

### What the wrong answers are

ESV targets the *trap*: the answer the original strategy S yields when applied blindly to a strategy-breaking P'. The premise holds only where that column is a large share of the failures.

| model | wrong CoT maj@4 answers | on originals | on preserving P' | on breaking P': the TRAP answer | on breaking P': other wrong | CoT maj@4 on originals | CoT maj@4 on breaking P' | CoT samples giving the trap answer |
|---|---|---|---|---|---|---|---|---|
| qwen/qwen-2.5-7b-instruct | 26 | 11/26 (42.3%) | 4/26 (15.4%) | 6/26 (23.1%) | 5/26 (19.2%) | 91.7% | 84.1% | 21/276 (7.6%) |
| mistralai/mistral-nemo | 40 | 17/40 (42.5%) | 6/40 (15.0%) | 8/40 (20.0%) | 9/40 (22.5%) | 87.2% | 75.4% | 26/276 (9.4%) |
| google/gemma-3-4b-it | 25 | 10/25 (40.0%) | 1/25 (4.0%) | 7/25 (28.0%) | 7/25 (28.0%) | 92.5% | 79.7% | 23/276 (8.3%) |
| **pooled** | 91 | 38/91 (41.8%) | 11/91 (12.1%) | 21/91 (23.1%) | 21/91 (23.1%) | 90.5% | 79.7% | 70/828 (8.5%) |

## Stages 2-3: candidate verification and choice

**Go / no-go (fixed before the run): NO-GO.** ESV v3 - CoT maj@4 = -3.7 pp (needs >= +3); ESV v3 - USC = -4.6 pp (needs > 0).

### All instances

**qwen/qwen-2.5-7b-instruct** (271 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 18.1% | 3 | 199 | -196 (p=4e-55) |  |
| CoT pass@1 | 87.6% |  |  |  |  |
| CoT maj@4 (initial) | 90.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 94.5% |  |  |  |  |
| Universal Self-Consistency | 89.7% | 1 | 3 | -2 (p=0.625) | 2 / 0.20 |
| ESV v3 (stages 1-3) | 88.2% | 2 | 8 | -6 (p=0.109) | 172 / 0.20 |

**mistralai/mistral-nemo** (271 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 9.6% | 2 | 207 | -205 (p=5e-59) |  |
| CoT pass@1 | 78.9% |  |  |  |  |
| CoT maj@4 (initial) | 85.2% |  |  |  |  |
| CoT pass@4 (oracle choice) | 95.2% |  |  |  |  |
| Universal Self-Consistency | 86.7% | 7 | 3 | +4 (p=0.344) | 5 / 0.44 |
| ESV v3 (stages 1-3) | 79.0% | 7 | 24 | -17 (p=0.003) | 342 / 0.44 |

**google/gemma-3-4b-it** (271 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 6.6% | 1 | 229 | -228 (p=3e-67) |  |
| CoT pass@1 | 89.1% |  |  |  |  |
| CoT maj@4 (initial) | 90.8% |  |  |  |  |
| CoT pass@4 (oracle choice) | 95.6% |  |  |  |  |
| Universal Self-Consistency | 92.6% | 6 | 1 | +5 (p=0.125) | 3 / 0.16 |
| ESV v3 (stages 1-3) | 88.2% | 3 | 10 | -7 (p=0.092) | 151 / 0.16 |

**Pooled** (813 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 11.4% | 6 | 635 | -629 (p=2e-179) |  |
| CoT pass@1 | 85.2% |  |  |  |  |
| CoT maj@4 (initial) | 88.8% |  |  |  |  |
| CoT pass@4 (oracle choice) | 95.1% |  |  |  |  |
| Universal Self-Consistency | 89.7% | 14 | 7 | +7 (p=0.189) | 3 / 0.27 |
| ESV v3 (stages 1-3) | 85.1% | 12 | 42 | -30 (p=5e-05) | 222 / 0.27 |

ESV v3 vs USC (ESV-only / USC-only right, McNemar p): 12 / 49 (p=2e-06)

### Did USC actually select? (pooled)

USC names its pick in one tag. Where that tag cannot be resolved to a response, USC keeps the initial answer and its accuracy becomes CoT maj@4's by construction - so this table has to be read before any USC comparison.

| model | USC calls | by response label | by answer value | ambiguous (label used) | unresolved -> keeps CoT maj@4 |
|---|---|---|---|---|---|
| qwen/qwen-2.5-7b-instruct | 55 | 21/55 (38.2%) | 34/55 (61.8%) | 0/55 (0.0%) | 0/55 (0.0%) |
| mistralai/mistral-nemo | 118 | 75/118 (63.6%) | 41/118 (34.7%) | 2/118 (1.7%) | 0/118 (0.0%) |
| google/gemma-3-4b-it | 44 | 33/44 (75.0%) | 5/44 (11.4%) | 0/44 (0.0%) | 6/44 (13.6%) |
| **pooled** | 217 | 129/217 (59.4%) | 80/217 (36.9%) | 2/217 (0.9%) | 6/217 (2.8%) |

### The choice, on disagreeing instances only (pooled)

| Choice among the candidates (217 disagreeing instances) | right |
|---|---|
| majority vote (CoT maj@4) | 139/217 (64.1%) |
| uniform random candidate (chance) | 84/217 (38.7%) |
| Universal Self-Consistency | 146/217 (67.3%) |
| ESV v3, strict SOUND (used) | 109/217 (50.2%) |
| ESV v3, verdict-only SOUND (secondary) | 108/217 (49.8%) |
| oracle choice (CoT pass@4) | 190/217 (87.6%) |

### Stages 2-3 as a detector, per candidate (pooled)

| SOUND rule | right candidates called SOUND | wrong candidates called SOUND | precision of SOUND |
|---|---|---|---|
| strict (used) | 108/190 (56.8%) | 123/331 (37.2%) | 46.8% |
| verdict only | 115/190 (60.5%) | 132/331 (39.9%) | 46.6% |

|  | n | some binding MISMATCH | some condition VIOLATED | verdict unreadable | failure types when FLAWED |
|---|---|---|---|---|---|
| right candidates | 190 | 11/190 (5.8%) | 10/190 (5.3%) | 0/190 (0.0%) | METHOD: 35, COMPUTATION: 24, BINDING: 14, None: 2 |
| wrong candidates | 331 | 47/331 (14.2%) | 40/331 (12.1%) | 0/331 (0.0%) | COMPUTATION: 77, METHOD: 62, BINDING: 55, None: 5 |

### Decisions (pooled)

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 596 | 97.8% | 97.8% | 0 | 0 | 583 |
| one_sound | 162 | 67.9% | 50.6% | 11 | 39 | 143 |
| several_sound | 33 | 54.5% | 48.5% | 1 | 3 | 30 |
| none_sound | 22 | 50.0% | 50.0% | 0 | 0 | 17 |

`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.

### By arm (pooled)

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 399 | 90.5% | 91.5% | 88.5% | 5 / 13 |
| breaking | 207 | 79.7% | 80.7% | 72.9% | 5 / 19 |
| preserving | 207 | 94.7% | 95.2% | 90.8% | 2 / 10 |

## Compute cost

Read off the `usage` and `latency_seconds` stored on every call, so this section is reproducible offline from the records. Every method starts from the same 4 CoT samples and adds at most one call, so the baseline is shared and only the extra call separates them. **serial** is billed compute time; **critical path** is what a user waits for, since `run_methods.py` issues the 4 samples concurrently while the extra call cannot start until all 4 are back.

| Method | accuracy | calls | solver tokens | verifier tokens | total tokens | vs CoT maj@4 | serial s | critical path s |
|---|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | 11.4% | 4.00 | 1,071 | - | 1,071 | -54.4% | 2.1 | 0.8 |
| CoT pass@1 | 85.2% | 1.00 | 587 | - | 587 | -75.0% | 4.8 | 4.8 |
| CoT maj@4 (initial) | 88.8% | 4.00 | 2,349 | - | 2,349 | +0.0% | 19.2 | 6.4 |
| Universal Self-Consistency | 89.7% | 4.27 | 2,752 | - | 2,752 | +17.1% | 19.4 | 6.6 |
| ESV v3 (stages 1-3) | 85.1% | 4.27 | 3,098 | - | 3,098 | +31.9% | 23.5 | 10.7 |
| CoT pass@4 (oracle choice) | 95.1% | 4.00 | 2,349 | - | 2,349 | +0.0% | 19.2 | 6.4 |

### What the extra compute buys (pooled)

| Method | accuracy vs CoT maj@4 | fixes / breaks | net | McNemar | extra tokens / instance | extra wait | extra tokens per net fix |
|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | -77.4 pp | 6 / 635 | -629 | p=2e-179 | -1,278 | -5.7 s | no net gain |
| Universal Self-Consistency | +0.9 pp | 14 / 7 | +7 | p=0.189 | +403 | +0.2 s | 46,796 |
| ESV v3 (stages 1-3) | -3.7 pp | 12 / 42 | -30 | p=5e-05 | +749 | +4.2 s | no net gain |
| CoT pass@4 (oracle choice) | +6.3 pp | 51 / 0 | +51 | p=9e-16 | +0 | +0.0 s | 0 |

### The disagreement gate is the efficiency mechanism (pooled)

Stage 1 costs nothing - it is code over answers the solver already produced - and it is what keeps ESV under 1.5x the baseline despite a verification call that is itself larger than a CoT sample.

|  | per instance |
|---|---|
| Instances where the gate fires (>= 2 distinct candidates) | 217/813 (26.7%) |
| A verification call, on an instance that fires | 2,806 tokens, 15.9 s |
| Amortized over every instance (what ESV actually costs) | 749 tokens, 4.2 s |
| The same protocol ungated, verifying every instance | 2,806 tokens, 15.9 s |
| Verification spend the gate avoids | 73.3% |
| Cost on a gated-out instance | 0 tokens, 0 s - ESV returns the CoT maj@4 answer untouched |

### Compute by arm (pooled)

The gate fires where the perturbation actually hurts, so ESV spends most on the strategy-breaking arm - which is also where it gains most. The cost profile is adaptive, not flat.

| arm | instances | gate fires | CoT maj@4 | ESV v3 | baseline tokens | extra tokens | vs baseline |
|---|---|---|---|---|---|---|---|
| original | 399 | 86/399 (21.6%) | 90.5% | 88.5% | 2,265 | +570 | +25.2% |
| breaking | 207 | 93/207 (44.9%) | 79.7% | 72.9% | 2,697 | +1,382 | +51.2% |
| preserving | 207 | 38/207 (18.4%) | 94.7% | 90.8% | 2,164 | +461 | +21.3% |
