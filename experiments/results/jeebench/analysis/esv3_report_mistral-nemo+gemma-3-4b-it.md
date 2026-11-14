# ESV v3, stage by stage (JEEBench (math), models mistral-nemo, gemma-3-4b-it)

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
| mistralai/mistral-nemo | 386 | 16/386 (4.1%) | 10/16 (62.5%) | 370/386 (95.9%) | 57 | 65 | 248 | 16.3% | 33.2% |
| google/gemma-3-4b-it | 386 | 62/386 (16.1%) | 26/62 (41.9%) | 324/386 (83.9%) | 86 | 73 | 165 | 31.6% | 50.5% |
| **pooled** | 772 | 78/772 (10.1%) | 36/78 (46.2%) | 694/772 (89.9%) | 143 | 138 | 413 | 24.0% | 41.8% |

### What the wrong answers are

ESV targets the *trap*: the answer the original strategy S yields when applied blindly to a strategy-breaking P'. The premise holds only where that column is a large share of the failures.

| model | wrong CoT maj@4 answers | on originals | on preserving P' | on breaking P': the TRAP answer | on breaking P': other wrong | CoT maj@4 on originals | CoT maj@4 on breaking P' | CoT samples giving the trap answer |
|---|---|---|---|---|---|---|---|---|
| mistralai/mistral-nemo | 323 | 161/323 (49.8%) | 84/323 (26.0%) | 11/323 (3.4%) | 67/323 (20.7%) | 16.6% | 19.6% | 42/388 (10.8%) |
| google/gemma-3-4b-it | 264 | 134/264 (50.8%) | 62/264 (23.5%) | 16/264 (6.1%) | 52/264 (19.7%) | 30.6% | 29.9% | 53/388 (13.7%) |
| **pooled** | 587 | 295/587 (50.3%) | 146/587 (24.9%) | 27/587 (4.6%) | 119/587 (20.3%) | 23.6% | 24.7% | 95/776 (12.2%) |

## Stages 2-3: candidate verification and choice

**Go / no-go (fixed before the run): NO-GO.** ESV v3 - CoT maj@4 = -2.2 pp (needs >= +3); ESV v3 - USC = -0.3 pp (needs > 0).

### All instances

**mistralai/mistral-nemo** (386 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 11.9% | 33 | 50 | -17 (p=0.078) |  |
| CoT pass@1 | 13.1% |  |  |  |  |
| CoT maj@4 (initial) | 16.3% |  |  |  |  |
| CoT pass@4 (oracle choice) | 33.2% |  |  |  |  |
| Universal Self-Consistency | 14.2% | 17 | 25 | -8 (p=0.280) | 10 / 0.96 |
| ESV v3 (stages 1-3) | 15.0% | 19 | 24 | -5 (p=0.542) | 1028 / 0.96 |

**google/gemma-3-4b-it** (386 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 11.9% | 24 | 100 | -76 (p=3e-12) |  |
| CoT pass@1 | 28.0% |  |  |  |  |
| CoT maj@4 (initial) | 31.6% |  |  |  |  |
| CoT pass@4 (oracle choice) | 50.5% |  |  |  |  |
| Universal Self-Consistency | 29.8% | 15 | 22 | -7 (p=0.324) | 16 / 0.84 |
| ESV v3 (stages 1-3) | 28.5% | 14 | 26 | -12 (p=0.081) | 1191 / 0.84 |

**Pooled** (772 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 11.9% | 57 | 150 | -93 (p=8e-11) |  |
| CoT pass@1 | 20.5% |  |  |  |  |
| CoT maj@4 (initial) | 24.0% |  |  |  |  |
| CoT pass@4 (oracle choice) | 41.8% |  |  |  |  |
| Universal Self-Consistency | 22.0% | 32 | 47 | -15 (p=0.115) | 13 / 0.90 |
| ESV v3 (stages 1-3) | 21.8% | 33 | 50 | -17 (p=0.078) | 1109 / 0.90 |

ESV v3 vs USC (ESV-only / USC-only right, McNemar p): 45 / 47 (p=0.917)

### Did USC actually select? (pooled)

USC names its pick in one tag. Where that tag cannot be resolved to a response, USC keeps the initial answer and its accuracy becomes CoT maj@4's by construction - so this table has to be read before any USC comparison.

| model | USC calls | by response label | by answer value | ambiguous (label used) | unresolved -> keeps CoT maj@4 |
|---|---|---|---|---|---|
| mistralai/mistral-nemo | 370 | 353/370 (95.4%) | 0/370 (0.0%) | 17/370 (4.6%) | 0/370 (0.0%) |
| google/gemma-3-4b-it | 324 | 163/324 (50.3%) | 130/324 (40.1%) | 11/324 (3.4%) | 20/324 (6.2%) |
| **pooled** | 694 | 516/694 (74.4%) | 130/694 (18.7%) | 28/694 (4.0%) | 20/694 (2.9%) |

### The choice, on disagreeing instances only (pooled)

| Choice among the candidates (694 disagreeing instances) | right |
|---|---|
| majority vote (CoT maj@4) | 143/694 (20.6%) |
| uniform random candidate (chance) | 108.2/694 (15.6%) |
| Universal Self-Consistency | 128/694 (18.4%) |
| ESV v3, strict SOUND (used) | 126/694 (18.2%) |
| ESV v3, verdict-only SOUND (secondary) | 122/694 (17.6%) |
| oracle choice (CoT pass@4) | 281/694 (40.5%) |

### Stages 2-3 as a detector, per candidate (pooled)

| SOUND rule | right candidates called SOUND | wrong candidates called SOUND | precision of SOUND |
|---|---|---|---|
| strict (used) | 125/280 (44.6%) | 748/1760 (42.5%) | 14.3% |
| verdict only | 133/280 (47.5%) | 780/1760 (44.3%) | 14.6% |

|  | n | some binding MISMATCH | some condition VIOLATED | verdict unreadable | failure types when FLAWED |
|---|---|---|---|---|---|
| right candidates | 280 | 6/280 (2.1%) | 20/280 (7.1%) | 0/280 (0.0%) | METHOD: 56, COMPUTATION: 53, BINDING: 29, None: 9 |
| wrong candidates | 1760 | 37/1760 (2.1%) | 137/1760 (7.8%) | 5/1760 (0.3%) | METHOD: 427, COMPUTATION: 330, BINDING: 171, None: 47 |

### Decisions (pooled)

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 78 | 53.8% | 53.8% | 0 | 0 | 42 |
| one_sound | 331 | 22.4% | 16.6% | 27 | 46 | 139 |
| several_sound | 235 | 16.6% | 17.4% | 6 | 4 | 89 |
| none_sound | 127 | 23.6% | 23.6% | 0 | 0 | 53 |
| unreadable | 1 | 0.0% | 0.0% | 0 | 0 | 0 |

`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.

### By arm

Accuracy on the untouched originals, on strategy-preserving P' and on strategy-breaking P', per model and pooled. ESV's premise is that the breaking arm is where a selector has the most to recover, so the fixes / breaks column is read per arm.

**mistralai/mistral-nemo**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 193 | 16.6% | 15.5% | 15.0% | 9 / 12 |
| breaking | 97 | 19.6% | 15.5% | 18.6% | 5 / 6 |
| preserving | 96 | 12.5% | 10.4% | 11.5% | 5 / 6 |

**google/gemma-3-4b-it**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 193 | 30.6% | 27.5% | 26.9% | 4 / 11 |
| breaking | 97 | 29.9% | 29.9% | 25.8% | 3 / 7 |
| preserving | 96 | 35.4% | 34.4% | 34.4% | 7 / 8 |

**Pooled**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 386 | 23.6% | 21.5% | 21.0% | 13 / 23 |
| breaking | 194 | 24.7% | 22.7% | 22.2% | 8 / 13 |
| preserving | 192 | 24.0% | 22.4% | 22.9% | 12 / 14 |

### Decisions, per model

The same decision mix as above, split by model: a checker that almost never returns `one_sound` is abstaining, not selecting.

**mistralai/mistral-nemo**

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 16 | 37.5% | 37.5% | 0 | 0 | 6 |
| one_sound | 198 | 17.7% | 14.1% | 15 | 22 | 69 |
| several_sound | 138 | 10.9% | 12.3% | 4 | 2 | 43 |
| none_sound | 34 | 20.6% | 20.6% | 0 | 0 | 10 |

**google/gemma-3-4b-it**

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 62 | 58.1% | 58.1% | 0 | 0 | 36 |
| one_sound | 133 | 29.3% | 20.3% | 12 | 24 | 70 |
| several_sound | 97 | 24.7% | 24.7% | 2 | 2 | 46 |
| none_sound | 93 | 24.7% | 24.7% | 0 | 0 | 43 |
| unreadable | 1 | 0.0% | 0.0% | 0 | 0 | 0 |

## Compute cost

Read off the `usage` and `latency_seconds` stored on every call, so this section is reproducible offline from the records. Every method starts from the same 4 CoT samples and adds at most one call, so the baseline is shared and only the extra call separates them. **serial** is billed compute time; **critical path** is what a user waits for, since `run_methods.py` issues the 4 samples concurrently while the extra call cannot start until all 4 are back.

| Method | accuracy | calls | solver tokens | verifier tokens | total tokens | vs CoT maj@4 | serial s | critical path s |
|---|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | 11.9% | 4.00 | 1,982 | - | 1,982 | -73.7% | 5.6 | 2.6 |
| CoT pass@1 | 20.5% | 1.00 | 1,887 | - | 1,887 | -75.0% | 19.4 | 19.4 |
| CoT maj@4 (initial) | 24.0% | 4.00 | 7,548 | - | 7,548 | +0.0% | 77.7 | 31.4 |
| Universal Self-Consistency | 22.0% | 4.90 | 10,910 | - | 10,910 | +44.5% | 78.6 | 32.2 |
| ESV v3 (stages 1-3) | 21.8% | 4.90 | 12,411 | - | 12,411 | +64.4% | 101.9 | 55.6 |
| CoT pass@4 (oracle choice) | 41.8% | 4.00 | 7,548 | - | 7,548 | +0.0% | 77.7 | 31.4 |

### What the extra compute buys (pooled)

| Method | accuracy vs CoT maj@4 | fixes / breaks | net | McNemar | extra tokens / instance | extra wait | extra tokens per net fix |
|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | -12.0 pp | 57 / 150 | -93 | p=8e-11 | -5,566 | -28.8 s | no net gain |
| Universal Self-Consistency | -1.9 pp | 32 / 47 | -15 | p=0.115 | +3,362 | +0.9 s | no net gain |
| ESV v3 (stages 1-3) | -2.2 pp | 33 / 50 | -17 | p=0.078 | +4,863 | +24.2 s | no net gain |
| CoT pass@4 (oracle choice) | +17.9 pp | 138 / 0 | +138 | p=6e-42 | +0 | +0.0 s | 0 |

### The disagreement gate is the efficiency mechanism (pooled)

Stage 1 costs nothing - it is code over answers the solver already produced - and it is what keeps ESV under 1.5x the baseline despite a verification call that is itself larger than a CoT sample.

|  | per instance |
|---|---|
| Instances where the gate fires (>= 2 distinct candidates) | 694/772 (89.9%) |
| A verification call, on an instance that fires | 5,410 tokens, 26.9 s |
| Amortized over every instance (what ESV actually costs) | 4,863 tokens, 24.2 s |
| The same protocol ungated, verifying every instance | 5,410 tokens, 26.9 s |
| Verification spend the gate avoids | 10.1% |
| Cost on a gated-out instance | 0 tokens, 0 s - ESV returns the CoT maj@4 answer untouched |

### Compute by arm (pooled)

The gate fires where the perturbation actually hurts, so ESV spends most on the strategy-breaking arm - which is also where it gains most. The cost profile is adaptive, not flat.

| arm | instances | gate fires | CoT maj@4 | ESV v3 | baseline tokens | extra tokens | vs baseline |
|---|---|---|---|---|---|---|---|
| original | 386 | 352/386 (91.2%) | 23.6% | 21.0% | 7,675 | +4,918 | +64.1% |
| breaking | 194 | 172/194 (88.7%) | 24.7% | 22.2% | 7,524 | +4,760 | +63.3% |
| preserving | 192 | 170/192 (88.5%) | 24.0% | 22.9% | 7,318 | +4,857 | +66.4% |

### By answer format (pooled)

`expression` answers are compared by LaTeX normalization and symbolic equivalence (`expression_answers.py`) rather than by exact match, so this split is the check that the scoring is not what separates the methods.

| answer format | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| integer | 116 | 25.9% | 25.9% | 25.0% | 4 / 5 |
| multiple | 272 | 15.8% | 15.8% | 15.4% | 11 / 12 |
| numeric | 220 | 23.6% | 20.0% | 20.0% | 3 / 11 |
| single | 164 | 36.6% | 32.3% | 32.3% | 15 / 22 |
