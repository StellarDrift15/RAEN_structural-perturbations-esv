# ESV v3, stage by stage (JEEBench (math), small models)

Version `jee_esv3_v1_2026-11-03`, decision rule `stages1-3_strict_v1_2026-11-03`. Gold is used only to score. Design and motivation: `esv_bottleneck_analysis.md`.

## The pipeline, as run

Every method starts from the same 4 CoT samples and none of them re-solves the problem; they differ only in how they pick among answers the solver already produced. ESV fires only on disagreement, so it can damage a correct majority as easily as repair a wrong one, and is scored that way.

| # | Stage | What it does | Calls | Model called | Outcome |
|---|---|---|---|---|---|
| 0 | CoT sampling (shared input, not part of ESV) | The solver answers the problem 4 times with chain-of-thought. Produced once by `run_methods.py` and reused unchanged by every method, so nothing below ever re-solves the problem. | 4 per instance, T = 0.7, seeds 0-3 | solver | 4 answers. Their majority is the `initial` answer and is scored as the CoT maj@4 baseline. |
| 1 | Disagreement gate | Group the 4 answers into distinct candidates by normalized value, in order of first appearance, keeping each one's vote count. Unreadable answers are dropped. | 0 (free) | none | Fewer than 2 distinct candidates -> decision `unanimous`: keep the initial answer and STOP. 2 or more -> go to stage 2. |
| 2 | Contrastive verification | All distinct candidates in ONE call, one representative solution each, in order of first appearance, with vote counts hidden. Per candidate: the strategy, its bindings to quoted givens (MATCHES / MISMATCH), the conditions the method needs (HOLDS / VIOLATED), then the first divergence point, then SOUND or FLAWED with a failure type (BINDING / METHOD / COMPUTATION), the flawed step and the corrected fact. | 1 per disagreeing instance, T = 0 | the solver itself (self-verification) | A filled-in check per candidate (built, not run yet). |
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
| qwen/qwen-2.5-7b-instruct | 386 | 65/386 (16.8%) | 18/65 (27.7%) | 321/386 (83.2%) | 87 | 76 | 158 | 34.7% | 54.4% |
| mistralai/mistral-nemo | 386 | 16/386 (4.1%) | 10/16 (62.5%) | 370/386 (95.9%) | 57 | 65 | 248 | 16.3% | 33.2% |
| google/gemma-3-4b-it | 386 | 62/386 (16.1%) | 26/62 (41.9%) | 324/386 (83.9%) | 86 | 73 | 165 | 31.6% | 50.5% |
| **pooled** | 1158 | 143/1158 (12.3%) | 54/143 (37.8%) | 1015/1158 (87.7%) | 230 | 214 | 571 | 27.5% | 46.0% |

### What the wrong answers are

ESV targets the *trap*: the answer the original strategy S yields when applied blindly to a strategy-breaking P'. The premise holds only where that column is a large share of the failures.

| model | wrong CoT maj@4 answers | on originals | on preserving P' | on breaking P': the TRAP answer | on breaking P': other wrong | CoT maj@4 on originals | CoT maj@4 on breaking P' | CoT samples giving the trap answer |
|---|---|---|---|---|---|---|---|---|
| qwen/qwen-2.5-7b-instruct | 252 | 125/252 (49.6%) | 55/252 (21.8%) | 14/252 (5.6%) | 58/252 (23.0%) | 35.2% | 25.8% | 51/388 (13.1%) |
| mistralai/mistral-nemo | 323 | 161/323 (49.8%) | 84/323 (26.0%) | 11/323 (3.4%) | 67/323 (20.7%) | 16.6% | 19.6% | 42/388 (10.8%) |
| google/gemma-3-4b-it | 264 | 134/264 (50.8%) | 62/264 (23.5%) | 16/264 (6.1%) | 52/264 (19.7%) | 30.6% | 29.9% | 53/388 (13.7%) |
| **pooled** | 839 | 420/839 (50.1%) | 201/839 (24.0%) | 41/839 (4.9%) | 177/839 (21.1%) | 27.5% | 25.1% | 146/1164 (12.5%) |
