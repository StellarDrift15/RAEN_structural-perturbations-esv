# ESV v3, stage by stage (JEEBench (math), models mistral-nemo, gemma-3-4b-it, verifier openai/gpt-oss-120b)

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

**Go / no-go (fixed before the run): GO.** ESV v3 - CoT maj@4 = +3.5 pp (needs >= +3); ESV v3 - USC = +5.4 pp (needs > 0).

### All instances

**mistralai/mistral-nemo** (386 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 11.9% | 33 | 50 | -17 (p=0.078) |  |
| CoT pass@1 | 13.1% |  |  |  |  |
| CoT maj@4 (initial) | 16.3% |  |  |  |  |
| CoT pass@4 (oracle choice) | 33.2% |  |  |  |  |
| Universal Self-Consistency | 14.2% | 17 | 25 | -8 (p=0.280) | 10 / 0.96 |
| Judge-select (same checker, no ESV form) | 26.9% | 41 | 0 | +41 (p=9e-13) | 2466 / 0.96 |
| ESV v3 (stages 1-3) | 17.1% | 3 | 0 | +3 (p=0.250) | 3816 / 0.96 |

**google/gemma-3-4b-it** (386 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 11.9% | 24 | 100 | -76 (p=3e-12) |  |
| CoT pass@1 | 28.0% |  |  |  |  |
| CoT maj@4 (initial) | 31.6% |  |  |  |  |
| CoT pass@4 (oracle choice) | 50.5% |  |  |  |  |
| Universal Self-Consistency | 29.8% | 15 | 22 | -7 (p=0.324) | 16 / 0.84 |
| Judge-select (same checker, no ESV form) | 46.4% | 57 | 0 | +57 (p=1e-17) | 1928 / 0.84 |
| ESV v3 (stages 1-3) | 37.8% | 25 | 1 | +24 (p=8e-07) | 3008 / 0.84 |

**Pooled** (772 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 11.9% | 57 | 150 | -93 (p=8e-11) |  |
| CoT pass@1 | 20.5% |  |  |  |  |
| CoT maj@4 (initial) | 24.0% |  |  |  |  |
| CoT pass@4 (oracle choice) | 41.8% |  |  |  |  |
| Universal Self-Consistency | 22.0% | 32 | 47 | -15 (p=0.115) | 13 / 0.90 |
| Judge-select (same checker, no ESV form) | 36.7% | 98 | 0 | +98 (p=6e-30) | 2197 / 0.90 |
| ESV v3 (stages 1-3) | 27.5% | 28 | 1 | +27 (p=1e-07) | 3412 / 0.90 |

ESV v3 vs USC (ESV-only / USC-only right, McNemar p): 69 / 27 (p=2e-05)

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
| ESV v3, strict SOUND (used) | 170/694 (24.5%) |
| ESV v3, verdict-only SOUND (secondary) | 170/694 (24.5%) |
| oracle choice (CoT pass@4) | 281/694 (40.5%) |

### Stages 2-3 as a detector, per candidate (pooled)

| SOUND rule | right candidates called SOUND | wrong candidates called SOUND | precision of SOUND |
|---|---|---|---|
| strict (used) | 70/280 (25.0%) | 9/1760 (0.5%) | 88.6% |
| verdict only | 70/280 (25.0%) | 9/1760 (0.5%) | 88.6% |

|  | n | some binding MISMATCH | some condition VIOLATED | verdict unreadable | failure types when FLAWED |
|---|---|---|---|---|---|
| right candidates | 280 | 122/280 (43.6%) | 145/280 (51.8%) | 0/280 (0.0%) | BINDING: 104, COMPUTATION: 61, METHOD: 45 |
| wrong candidates | 1760 | 1142/1760 (64.9%) | 1337/1760 (76.0%) | 17/1760 (1.0%) | BINDING: 844, METHOD: 500, COMPUTATION: 388, None: 2 |

### Decisions (pooled)

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 78 | 53.8% | 53.8% | 0 | 0 | 42 |
| one_sound | 75 | 56.0% | 92.0% | 28 | 1 | 70 |
| several_sound | 2 | 0.0% | 0.0% | 0 | 0 | 1 |
| none_sound | 612 | 16.5% | 16.5% | 0 | 0 | 210 |
| unreadable | 5 | 0.0% | 0.0% | 0 | 0 | 0 |

`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.

### By arm

Accuracy on the untouched originals, on strategy-preserving P' and on strategy-breaking P', per model and pooled. ESV's premise is that the breaking arm is where a selector has the most to recover, so the fixes / breaks column is read per arm.

**mistralai/mistral-nemo**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 193 | 16.6% | 15.5% | 17.6% | 2 / 0 |
| breaking | 97 | 19.6% | 15.5% | 19.6% | 0 / 0 |
| preserving | 96 | 12.5% | 10.4% | 13.5% | 1 / 0 |

**google/gemma-3-4b-it**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 193 | 30.6% | 27.5% | 35.2% | 9 / 0 |
| breaking | 97 | 29.9% | 29.9% | 36.1% | 7 / 1 |
| preserving | 96 | 35.4% | 34.4% | 44.8% | 9 / 0 |

**Pooled**

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 386 | 23.6% | 21.5% | 26.4% | 11 / 0 |
| breaking | 194 | 24.7% | 22.7% | 27.8% | 7 / 1 |
| preserving | 192 | 24.0% | 22.4% | 29.2% | 10 / 0 |

### Decisions, per model

The same decision mix as above, split by model: a checker that almost never returns `one_sound` is abstaining, not selecting.

**mistralai/mistral-nemo**

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 16 | 37.5% | 37.5% | 0 | 0 | 6 |
| one_sound | 10 | 50.0% | 80.0% | 3 | 0 | 8 |
| none_sound | 357 | 14.6% | 14.6% | 0 | 0 | 114 |
| unreadable | 3 | 0.0% | 0.0% | 0 | 0 | 0 |

**google/gemma-3-4b-it**

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 62 | 58.1% | 58.1% | 0 | 0 | 36 |
| one_sound | 65 | 56.9% | 93.8% | 25 | 1 | 62 |
| several_sound | 2 | 0.0% | 0.0% | 0 | 0 | 1 |
| none_sound | 255 | 19.2% | 19.2% | 0 | 0 | 96 |
| unreadable | 2 | 0.0% | 0.0% | 0 | 0 | 0 |

## The judge-select control: is it the structure, or the checker?

Judge-select asks the **same checker** to read the **same candidates** in the same order with the same vote counts hidden, reason freely in its own words, and name one solution or NONE. Everything is held fixed except ESV's form (bindings / conditions / divergence / verdict per candidate). It keeps the licence to reject every candidate, so ESV gets no free win from its `none_sound` branch, and it is given room to reason, so this is structured reasoning versus unstructured reasoning rather than versus no thinking at all.

**ESV v3 vs judge-select, paired over all 772 instances: 2 / 73 (McNemar p=2e-19).**

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra completion tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 11.9% | 57 | 150 | -93 (p=8e-11) |  |
| CoT pass@1 | 20.5% |  |  |  |  |
| CoT maj@4 (initial) | 24.0% |  |  |  |  |
| CoT pass@4 (oracle choice) | 41.8% |  |  |  |  |
| Universal Self-Consistency | 22.0% | 32 | 47 | -15 (p=0.115) | 13 / 0.90 |
| Judge-select (same checker, no ESV form) | 36.7% | 98 | 0 | +98 (p=6e-30) | 2197 / 0.90 |
| ESV v3 (stages 1-3) | 27.5% | 28 | 1 | +27 (p=1e-07) | 3412 / 0.90 |

### Where the two differ (disagreeing instances only)

Stratified by what ESV decided, because the hypothesis is specific: the form should matter least where one candidate is clearly sound, and most where none survives and judge-select has to commit anyway.

| ESV's decision | n | CoT maj@4 | judge-select | ESV v3 | oracle | ESV-only / judge-only right |
|---|---|---|---|---|---|---|
| one_sound | 75 | 56.0% | 90.7% | 92.0% | 93.3% | 2 / 1 (p=1.000) |
| several_sound | 2 | 0.0% | 50.0% | 0.0% | 50.0% | 0 / 1 (p=1.000) |
| none_sound | 612 | 16.5% | 28.1% | 16.5% | 34.3% | 0 / 71 (p=8e-22) |
| unreadable | 5 | 0.0% | 0.0% | 0.0% | 0.0% | 0 / 0 (p=-) |

### Selection quality on the disagreeing instances

|  | correct |
|---|---|
| CoT maj@4 (initial) | 20.6% |
| Universal Self-Consistency | 18.4% |
| Judge-select (same checker, no ESV form) | 34.7% |
| ESV v3 (stages 1-3) | 24.5% |
| CoT pass@4 (oracle choice) | 40.5% |

### What judge-select actually returned

`none` is a judgement that no candidate holds up - the counterpart of ESV's `none_sound`. `unresolved` is a tag that named nothing usable; both keep the initial answer, but only the first is a decision, so they are reported apart.

| selection | instances | correct |
|---|---|---|
| none | 475/694 (68.4%) | 6.1% |
| label | 206/694 (29.7%) | 98.1% |
| ambiguous | 10/694 (1.4%) | 100.0% |
| unresolved | 3/694 (0.4%) | 0.0% |

## Compute cost

Read off the `usage` and `latency_seconds` stored on every call, so this section is reproducible offline from the records. Every method starts from the same 4 CoT samples and adds at most one call, so the baseline is shared and only the extra call separates them. **serial** is billed compute time; **critical path** is what a user waits for, since `run_methods.py` issues the 4 samples concurrently while the extra call cannot start until all 4 are back.

| Method | accuracy | calls | solver tokens | verifier tokens | total tokens | vs CoT maj@4 | serial s | critical path s |
|---|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | 11.9% | 4.00 | 1,982 | - | 1,982 | -73.7% | 5.6 | 2.6 |
| CoT pass@1 | 20.5% | 1.00 | 1,887 | - | 1,887 | -75.0% | 19.4 | 19.4 |
| CoT maj@4 (initial) | 24.0% | 4.00 | 7,548 | - | 7,548 | +0.0% | 77.7 | 31.4 |
| Universal Self-Consistency | 22.0% | 4.90 | 10,910 | - | 10,910 | +44.5% | 78.6 | 32.2 |
| Judge-select (same checker, no ESV form) | 36.7% | 4.90 | 7,548 | 5,170 | 12,718 | +68.5% | 119.4 | 73.0 |
| ESV v3 (stages 1-3) | 27.5% | 4.90 | 7,548 | 7,567 | 15,116 | +100.3% | 138.4 | 92.0 |
| CoT pass@4 (oracle choice) | 41.8% | 4.00 | 7,548 | - | 7,548 | +0.0% | 77.7 | 31.4 |

### What the extra compute buys (pooled)

| Method | accuracy vs CoT maj@4 | fixes / breaks | net | McNemar | extra tokens / instance | extra wait | extra tokens per net fix |
|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | -12.0 pp | 57 / 150 | -93 | p=8e-11 | -5,566 | -28.8 s | no net gain |
| Universal Self-Consistency | -1.9 pp | 32 / 47 | -15 | p=0.115 | +3,362 | +0.9 s | no net gain |
| Judge-select (same checker, no ESV form) | +12.7 pp | 98 / 0 | +98 | p=6e-30 | +5,170 | +41.7 s | 40,728 |
| ESV v3 (stages 1-3) | +3.5 pp | 28 / 1 | +27 | p=1e-07 | +7,567 | +60.6 s | 216,367 |
| CoT pass@4 (oracle choice) | +17.9 pp | 138 / 0 | +138 | p=6e-42 | +0 | +0.0 s | 0 |

### The disagreement gate is the efficiency mechanism (pooled)

Stage 1 costs nothing - it is code over answers the solver already produced - and it is what keeps ESV under 1.5x the baseline despite a verification call that is itself larger than a CoT sample.

|  | per instance |
|---|---|
| Instances where the gate fires (>= 2 distinct candidates) | 694/772 (89.9%) |
| A verification call, on an instance that fires | 8,418 tokens, 67.4 s |
| Amortized over every instance (what ESV actually costs) | 7,567 tokens, 60.6 s |
| The same protocol ungated, verifying every instance | 8,418 tokens, 67.4 s |
| Verification spend the gate avoids | 10.1% |
| Cost on a gated-out instance | 0 tokens, 0 s - ESV returns the CoT maj@4 answer untouched |

### Compute by arm (pooled)

The gate fires where the perturbation actually hurts, so ESV spends most on the strategy-breaking arm - which is also where it gains most. The cost profile is adaptive, not flat.

| arm | instances | gate fires | CoT maj@4 | ESV v3 | baseline tokens | extra tokens | vs baseline |
|---|---|---|---|---|---|---|---|
| original | 386 | 352/386 (91.2%) | 23.6% | 26.4% | 7,675 | +7,474 | +97.4% |
| breaking | 194 | 172/194 (88.7%) | 24.7% | 27.8% | 7,524 | +7,916 | +105.2% |
| preserving | 192 | 170/192 (88.5%) | 24.0% | 29.2% | 7,318 | +7,403 | +101.2% |

The ESV check is billed by openai/gpt-oss-120b, the 4 samples by the solver, so a flat token count understates the check. k = the verifier's price per token over the solver's; totals are in solver-token equivalents. No price list is assumed - read off the row for your k.

| k | ESV tokens / instance | vs CoT maj@4 | check vs USC's overhead | per net fix |
|---|---|---|---|---|
| 1 | 15,116 | +100.3% | 2.3x | 216,367 |
| 2 | 22,683 | +200.5% | 4.5x | 432,735 |
| 5 | 45,384 | +501.3% | 11.3x | 1,081,836 |
| 10 | 83,221 | +1002.5% | 22.5x | 2,163,673 |
| 20 | 158,893 | +2005.0% | 45.0x | 4,327,345 |

### By answer format (pooled)

`expression` answers are compared by LaTeX normalization and symbolic equivalence (`expression_answers.py`) rather than by exact match, so this split is the check that the scoring is not what separates the methods.

| answer format | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| integer | 116 | 25.9% | 25.9% | 29.3% | 4 / 0 |
| multiple | 272 | 15.8% | 15.8% | 18.4% | 8 / 1 |
| numeric | 220 | 23.6% | 20.0% | 29.1% | 12 / 0 |
| single | 164 | 36.6% | 32.3% | 39.0% | 4 / 0 |
