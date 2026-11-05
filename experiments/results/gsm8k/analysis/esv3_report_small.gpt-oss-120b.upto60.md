# ESV v3, stage by stage (GSM8K, small models, verifier openai/gpt-oss-120b, DEV pairs < 60)

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

**Go / no-go (fixed before the run): GO.** ESV v3 - CoT maj@4 = +4.5 pp (needs >= +3); ESV v3 - USC = +3.6 pp (needs > 0).

### All instances

**qwen/qwen-2.5-7b-instruct** (119 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 20.2% | 0 | 86 | -86 (p=0.000) |  |
| CoT pass@1 | 91.2% |  |  |  |  |
| CoT maj@4 (initial) | 92.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 95.8% |  |  |  |  |
| Universal Self-Consistency | 93.3% | 1 | 0 | +1 (p=1.000) | 2 / 0.15 |
| ESV v3 (stages 1-3) | 95.8% | 4 | 0 | +4 (p=0.125) | 342 / 0.15 |

**mistralai/mistral-nemo** (119 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 12.6% | 1 | 84 | -83 (p=0.000) |  |
| CoT pass@1 | 78.2% |  |  |  |  |
| CoT maj@4 (initial) | 82.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 92.4% |  |  |  |  |
| Universal Self-Consistency | 84.0% | 4 | 2 | +2 (p=0.688) | 4 / 0.38 |
| ESV v3 (stages 1-3) | 90.8% | 10 | 0 | +10 (p=0.002) | 786 / 0.38 |

**google/gemma-3-4b-it** (119 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 8.4% | 0 | 95 | -95 (p=0.000) |  |
| CoT pass@1 | 85.7% |  |  |  |  |
| CoT maj@4 (initial) | 88.2% |  |  |  |  |
| CoT pass@4 (oracle choice) | 91.6% |  |  |  |  |
| Universal Self-Consistency | 88.2% | 0 | 0 | +0 (p=-) | 4 / 0.19 |
| ESV v3 (stages 1-3) | 89.9% | 2 | 0 | +2 (p=0.500) | 467 / 0.19 |

**Pooled** (357 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 13.7% | 1 | 265 | -264 (p=0.000) |  |
| CoT pass@1 | 85.0% |  |  |  |  |
| CoT maj@4 (initial) | 87.7% |  |  |  |  |
| CoT pass@4 (oracle choice) | 93.3% |  |  |  |  |
| Universal Self-Consistency | 88.5% | 5 | 2 | +3 (p=0.453) | 3 / 0.24 |
| ESV v3 (stages 1-3) | 92.2% | 16 | 0 | +16 (p=0.000) | 532 / 0.24 |

ESV v3 vs USC (ESV-only / USC-only right, McNemar p): 13 / 0 (p=0.000)

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
| ESV v3, strict SOUND (used) | 62/86 (72.1%) |
| ESV v3, verdict-only SOUND (secondary) | 62/86 (72.1%) |
| oracle choice (CoT pass@4) | 66/86 (76.7%) |

### Stages 2-3 as a detector, per candidate (pooled)

| SOUND rule | right candidates called SOUND | wrong candidates called SOUND | precision of SOUND |
|---|---|---|---|
| strict (used) | 54/66 (81.8%) | 5/162 (3.1%) | 91.5% |
| verdict only | 54/66 (81.8%) | 5/162 (3.1%) | 91.5% |

|  | n | some binding MISMATCH | some condition VIOLATED | verdict unreadable | failure types when FLAWED |
|---|---|---|---|---|---|
| right candidates | 66 | 7/66 (10.6%) | 3/66 (4.5%) | 0/66 (0.0%) | BINDING: 9, COMPUTATION: 2, METHOD: 1 |
| wrong candidates | 162 | 105/162 (64.8%) | 95/162 (58.6%) | 0/162 (0.0%) | BINDING: 90, COMPUTATION: 42, METHOD: 25 |

### Decisions (pooled)

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 271 | 98.5% | 98.5% | 0 | 0 | 267 |
| one_sound | 59 | 64.4% | 91.5% | 16 | 0 | 55 |
| none_sound | 27 | 29.6% | 29.6% | 0 | 0 | 11 |

`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.

### By arm (pooled)

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 177 | 91.0% | 91.5% | 93.2% | 4 / 0 |
| breaking | 90 | 77.8% | 77.8% | 86.7% | 8 / 0 |
| preserving | 90 | 91.1% | 93.3% | 95.6% | 4 / 0 |
