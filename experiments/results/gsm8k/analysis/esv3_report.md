# ESV v3, stage by stage (GSM8K, large models)

Version `jee_esv3_v1_2026-11-03`, decision rule `stages1-3_strict_v1_2026-11-03`. Gold is used only to score. Design and motivation: `esv_bottleneck_analysis.md`.

| Stage | Status |
|---|---|
| 1. Disagreement gate | built, measured below |
| 2-3. Candidate verification + choice | built, not run yet |
| 4. Diagnosis-driven repair | not built (decision `none_sound` keeps the initial answer) |

Excluded from every score: 6 instances whose published gold is wrong or ambiguous (`common.DISPUTED_INSTANCES`): 105_original, 111_original, 189_original, 40_original, 82_original, 95_original.

## Stage 1: disagreement gate (no calls)

Unanimous = fewer than 2 distinct readable answers among the 4 CoT samples: ESV keeps the answer. Only the disagreeing instances can change, and only the fixable ones can gain.

| model | instances | unanimous (no call) | unanimous but wrong (lost) | disagreeing (ESV runs) | disagreeing, majority right (can break) | disagreeing, fixable (majority wrong, a candidate right) | disagreeing, no candidate right | CoT maj@4 | CoT pass@4 (ceiling) |
|---|---|---|---|---|---|---|---|---|---|
| qwen/qwen-2.5-72b-instruct | 390 | 364/390 (93.3%) | 1/364 (0.3%) | 26/390 (6.7%) | 15 | 8 | 3 | 96.9% | 99.0% |
| mistralai/mistral-small-3.2-24b-instruct | 390 | 350/390 (89.7%) | 0/350 (0.0%) | 40/390 (10.3%) | 27 | 8 | 5 | 96.7% | 98.7% |
| meta-llama/llama-3.3-70b-instruct | 390 | 368/390 (94.4%) | 0/368 (0.0%) | 22/390 (5.6%) | 11 | 8 | 3 | 97.2% | 99.2% |
| **pooled** | 1170 | 1082/1170 (92.5%) | 1/1082 (0.1%) | 88/1170 (7.5%) | 53 | 24 | 11 | 96.9% | 99.0% |

### What the wrong answers are

ESV targets the *trap*: the answer the original strategy S yields when applied blindly to a strategy-breaking P'. The premise holds only where that column is a large share of the failures.

| model | wrong CoT maj@4 answers | on originals | on preserving P' | on breaking P': the TRAP answer | on breaking P': other wrong | CoT maj@4 on originals | CoT maj@4 on breaking P' | CoT samples giving the trap answer |
|---|---|---|---|---|---|---|---|---|
| qwen/qwen-2.5-72b-instruct | 12 | 2/12 (16.7%) | 1/12 (8.3%) | 4/12 (33.3%) | 5/12 (41.7%) | 99.0% | 90.9% | 14/396 (3.5%) |
| mistralai/mistral-small-3.2-24b-instruct | 13 | 3/13 (23.1%) | 1/13 (7.7%) | 4/13 (30.8%) | 5/13 (38.5%) | 98.4% | 90.9% | 13/396 (3.3%) |
| meta-llama/llama-3.3-70b-instruct | 11 | 5/11 (45.5%) | 0/11 (0.0%) | 1/11 (9.1%) | 5/11 (45.5%) | 97.4% | 93.9% | 4/396 (1.0%) |
| **pooled** | 36 | 10/36 (27.8%) | 2/36 (5.6%) | 9/36 (25.0%) | 15/36 (41.7%) | 98.3% | 91.9% | 31/1188 (2.6%) |
