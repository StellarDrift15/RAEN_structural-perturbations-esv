# ESV v3, stage by stage (GSM8K, small models)

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
| qwen/qwen-2.5-7b-instruct | 390 | 317/390 (81.3%) | 4/317 (1.3%) | 73/390 (18.7%) | 42 | 15 | 16 | 91.0% | 94.9% |
| mistralai/mistral-nemo | 390 | 227/390 (58.2%) | 2/227 (0.9%) | 163/390 (41.8%) | 104 | 39 | 20 | 84.4% | 94.4% |
| google/gemma-3-4b-it | 390 | 323/390 (82.8%) | 11/323 (3.4%) | 67/390 (17.2%) | 39 | 17 | 11 | 90.0% | 94.4% |
| **pooled** | 1170 | 867/1170 (74.1%) | 17/867 (2.0%) | 303/1170 (25.9%) | 185 | 71 | 47 | 88.5% | 94.5% |

### What the wrong answers are

ESV targets the *trap*: the answer the original strategy S yields when applied blindly to a strategy-breaking P'. The premise holds only where that column is a large share of the failures.

| model | wrong CoT maj@4 answers | on originals | on preserving P' | on breaking P': the TRAP answer | on breaking P': other wrong | CoT maj@4 on originals | CoT maj@4 on breaking P' | CoT samples giving the trap answer |
|---|---|---|---|---|---|---|---|---|
| qwen/qwen-2.5-7b-instruct | 35 | 15/35 (42.9%) | 4/35 (11.4%) | 6/35 (17.1%) | 10/35 (28.6%) | 92.2% | 83.8% | 25/396 (6.3%) |
| mistralai/mistral-nemo | 61 | 23/61 (37.7%) | 11/61 (18.0%) | 11/61 (18.0%) | 16/61 (26.2%) | 88.0% | 72.7% | 35/396 (8.8%) |
| google/gemma-3-4b-it | 39 | 16/39 (41.0%) | 4/39 (10.3%) | 8/39 (20.5%) | 11/39 (28.2%) | 91.7% | 80.8% | 30/396 (7.6%) |
| **pooled** | 135 | 54/135 (40.0%) | 19/135 (14.1%) | 25/135 (18.5%) | 37/135 (27.4%) | 90.6% | 79.1% | 90/1188 (7.6%) |

## Stages 2-3: candidate verification and choice

**Go / no-go (fixed before the run): NO-GO.** ESV v3 - CoT maj@4 = -3.3 pp (needs >= +3); ESV v3 - USC = -4.2 pp (needs > 0).

### All instances

**qwen/qwen-2.5-7b-instruct** (390 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 18.7% | 3 | 285 | -282 (p=0.000) |  |
| CoT pass@1 | 88.7% |  |  |  |  |
| CoT maj@4 (initial) | 91.0% |  |  |  |  |
| CoT pass@4 (oracle choice) | 94.9% |  |  |  |  |
| Universal Self-Consistency | 90.8% | 2 | 3 | -1 (p=1.000) | 2 / 0.19 |
| ESV v3 (stages 1-3) | 89.2% | 4 | 11 | -7 (p=0.118) | 164 / 0.19 |

**mistralai/mistral-nemo** (390 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 10.5% | 3 | 291 | -288 (p=0.000) |  |
| CoT pass@1 | 78.7% |  |  |  |  |
| CoT maj@4 (initial) | 84.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 94.4% |  |  |  |  |
| Universal Self-Consistency | 85.9% | 11 | 5 | +6 (p=0.210) | 4 / 0.42 |
| ESV v3 (stages 1-3) | 78.5% | 10 | 33 | -23 (p=0.001) | 341 / 0.42 |

**google/gemma-3-4b-it** (390 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 7.2% | 1 | 324 | -323 (p=0.000) |  |
| CoT pass@1 | 88.1% |  |  |  |  |
| CoT maj@4 (initial) | 90.0% |  |  |  |  |
| CoT pass@4 (oracle choice) | 94.4% |  |  |  |  |
| Universal Self-Consistency | 91.3% | 6 | 1 | +5 (p=0.125) | 3 / 0.17 |
| ESV v3 (stages 1-3) | 87.7% | 4 | 13 | -9 (p=0.049) | 171 / 0.17 |

**Pooled** (1170 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 12.1% | 7 | 900 | -893 (p=0.000) |  |
| CoT pass@1 | 85.1% |  |  |  |  |
| CoT maj@4 (initial) | 88.5% |  |  |  |  |
| CoT pass@4 (oracle choice) | 94.5% |  |  |  |  |
| Universal Self-Consistency | 89.3% | 19 | 9 | +10 (p=0.087) | 3 / 0.26 |
| ESV v3 (stages 1-3) | 85.1% | 18 | 57 | -39 (p=0.000) | 226 / 0.26 |

ESV v3 vs USC (ESV-only / USC-only right, McNemar p): 17 / 66 (p=0.000)

### Did USC actually select? (pooled)

USC names its pick in one tag. Where that tag cannot be resolved to a response, USC keeps the initial answer and its accuracy becomes CoT maj@4's by construction - so this table has to be read before any USC comparison.

| model | USC calls | by response label | by answer value | ambiguous (label used) | unresolved -> keeps CoT maj@4 |
|---|---|---|---|---|---|
| qwen/qwen-2.5-7b-instruct | 73 | 30/73 (41.1%) | 43/73 (58.9%) | 0/73 (0.0%) | 0/73 (0.0%) |
| mistralai/mistral-nemo | 163 | 111/163 (68.1%) | 49/163 (30.1%) | 3/163 (1.8%) | 0/163 (0.0%) |
| google/gemma-3-4b-it | 67 | 49/67 (73.1%) | 9/67 (13.4%) | 0/67 (0.0%) | 9/67 (13.4%) |
| **pooled** | 303 | 190/303 (62.7%) | 101/303 (33.3%) | 3/303 (1.0%) | 9/303 (3.0%) |

### The choice, on disagreeing instances only (pooled)

| Choice among the candidates (303 disagreeing instances) | right |
|---|---|
| majority vote (CoT maj@4) | 185/303 (61.1%) |
| uniform random candidate (chance) | 112.2/303 (37.0%) |
| Universal Self-Consistency | 195/303 (64.4%) |
| ESV v3, strict SOUND (used) | 146/303 (48.2%) |
| ESV v3, verdict-only SOUND (secondary) | 143/303 (47.2%) |
| oracle choice (CoT pass@4) | 256/303 (84.5%) |

### Stages 2-3 as a detector, per candidate (pooled)

| SOUND rule | right candidates called SOUND | wrong candidates called SOUND | precision of SOUND |
|---|---|---|---|
| strict (used) | 146/256 (57.0%) | 180/493 (36.5%) | 44.8% |
| verdict only | 155/256 (60.5%) | 192/493 (38.9%) | 44.7% |

|  | n | some binding MISMATCH | some condition VIOLATED | verdict unreadable | failure types when FLAWED |
|---|---|---|---|---|---|
| right candidates | 256 | 14/256 (5.5%) | 15/256 (5.9%) | 0/256 (0.0%) | METHOD: 42, COMPUTATION: 31, BINDING: 25, None: 3 |
| wrong candidates | 493 | 70/493 (14.2%) | 67/493 (13.6%) | 0/493 (0.0%) | COMPUTATION: 115, METHOD: 94, BINDING: 82, None: 10 |

### Decisions (pooled)

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 867 | 98.0% | 98.0% | 0 | 0 | 850 |
| one_sound | 223 | 66.4% | 49.3% | 16 | 54 | 192 |
| several_sound | 49 | 44.9% | 42.9% | 2 | 3 | 41 |
| none_sound | 31 | 48.4% | 48.4% | 0 | 0 | 23 |

`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.

### By arm (pooled)

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 576 | 90.6% | 91.5% | 88.7% | 7 / 18 |
| breaking | 297 | 79.1% | 79.8% | 72.7% | 8 / 27 |
| preserving | 297 | 93.6% | 94.6% | 90.6% | 3 / 12 |
