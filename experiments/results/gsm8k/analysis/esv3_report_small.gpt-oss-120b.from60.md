# ESV v3, stage by stage (GSM8K, small models, verifier openai/gpt-oss-120b, HELD-OUT pairs >= 60)

Version `jee_esv3_v1_2026-11-03`, decision rule `stages1-3_strict_v1_2026-11-03`. Gold is used only to score. Design and motivation: `esv_bottleneck_analysis.md`.

| Stage | Status |
|---|---|
| 1. Disagreement gate | built, measured below |
| 2-3. Candidate verification + choice | run, measured below |
| 4. Diagnosis-driven repair | not built (decision `none_sound` keeps the initial answer) |

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

**Go / no-go (fixed before the run): GO.** ESV v3 - CoT maj@4 = +5.3 pp (needs >= +3); ESV v3 - USC = +5.3 pp (needs > 0).

### All instances

**qwen/qwen-2.5-7b-instruct** (271 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 18.1% | 3 | 199 | -196 (p=0.000) |  |
| CoT pass@1 | 87.6% |  |  |  |  |
| CoT maj@4 (initial) | 90.4% |  |  |  |  |
| CoT pass@4 (oracle choice) | 94.5% |  |  |  |  |
| Universal Self-Consistency | 90.4% | 0 | 0 | +0 (p=-) | 0 / 0.00 |
| ESV v3 (stages 1-3) | 94.1% | 10 | 0 | +10 (p=0.002) | 413 / 0.20 |

**mistralai/mistral-nemo** (271 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 9.6% | 2 | 207 | -205 (p=0.000) |  |
| CoT pass@1 | 78.9% |  |  |  |  |
| CoT maj@4 (initial) | 85.2% |  |  |  |  |
| CoT pass@4 (oracle choice) | 95.2% |  |  |  |  |
| Universal Self-Consistency | 85.2% | 0 | 0 | +0 (p=-) | 0 / 0.00 |
| ESV v3 (stages 1-3) | 93.7% | 25 | 2 | +23 (p=0.000) | 840 / 0.44 |

**google/gemma-3-4b-it** (271 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 6.6% | 1 | 229 | -228 (p=0.000) |  |
| CoT pass@1 | 89.1% |  |  |  |  |
| CoT maj@4 (initial) | 90.8% |  |  |  |  |
| CoT pass@4 (oracle choice) | 95.6% |  |  |  |  |
| Universal Self-Consistency | 90.8% | 0 | 0 | +0 (p=-) | 0 / 0.00 |
| ESV v3 (stages 1-3) | 94.5% | 10 | 0 | +10 (p=0.002) | 309 / 0.16 |

**Pooled** (813 instances)

| Method | accuracy | fixes | breaks | net vs CoT maj@4 | extra tokens / calls per instance |
|---|---|---|---|---|---|
| Direct QA maj@4 | 11.4% | 6 | 635 | -629 (p=0.000) |  |
| CoT pass@1 | 85.2% |  |  |  |  |
| CoT maj@4 (initial) | 88.8% |  |  |  |  |
| CoT pass@4 (oracle choice) | 95.1% |  |  |  |  |
| Universal Self-Consistency | 88.8% | 0 | 0 | +0 (p=-) | 0 / 0.00 |
| ESV v3 (stages 1-3) | 94.1% | 45 | 2 | +43 (p=0.000) | 521 / 0.27 |

ESV v3 vs USC (ESV-only / USC-only right, McNemar p): 45 / 2 (p=0.000)

### The choice, on disagreeing instances only (pooled)

| Choice among the candidates (217 disagreeing instances) | right |
|---|---|
| majority vote (CoT maj@4) | 139/217 (64.1%) |
| uniform random candidate (chance) | 84/217 (38.7%) |
| Universal Self-Consistency | 139/217 (64.1%) |
| ESV v3, strict SOUND (used) | 182/217 (83.9%) |
| ESV v3, verdict-only SOUND (secondary) | 182/217 (83.9%) |
| oracle choice (CoT pass@4) | 190/217 (87.6%) |

### Stages 2-3 as a detector, per candidate (pooled)

| SOUND rule | right candidates called SOUND | wrong candidates called SOUND | precision of SOUND |
|---|---|---|---|
| strict (used) | 168/190 (88.4%) | 4/331 (1.2%) | 97.7% |
| verdict only | 168/190 (88.4%) | 4/331 (1.2%) | 97.7% |

|  | n | some binding MISMATCH | some condition VIOLATED | verdict unreadable | failure types when FLAWED |
|---|---|---|---|---|---|
| right candidates | 190 | 14/190 (7.4%) | 9/190 (4.7%) | 0/190 (0.0%) | BINDING: 16, COMPUTATION: 3, METHOD: 3 |
| wrong candidates | 331 | 230/331 (69.5%) | 151/331 (45.6%) | 0/331 (0.0%) | BINDING: 203, COMPUTATION: 87, METHOD: 37 |

### Decisions (pooled)

| decision | n | CoT maj@4 | ESV v3 | fixes | breaks | a candidate is right |
|---|---|---|---|---|---|---|
| unanimous | 596 | 97.8% | 97.8% | 0 | 0 | 583 |
| one_sound | 172 | 72.7% | 97.7% | 45 | 2 | 171 |
| none_sound | 45 | 31.1% | 31.1% | 0 | 0 | 19 |

`none_sound` is where stage 4 would act; `a candidate is right` there is its ceiling.

### By arm (pooled)

| arm | n | CoT maj@4 (initial) | Universal Self-Consistency | ESV v3 (stages 1-3) | ESV v3 fixes / breaks |
|---|---|---|---|---|---|
| original | 399 | 90.5% | 90.5% | 95.0% | 19 / 1 |
| breaking | 207 | 79.7% | 79.7% | 87.9% | 17 / 0 |
| preserving | 207 | 94.7% | 94.7% | 98.6% | 9 / 1 |
