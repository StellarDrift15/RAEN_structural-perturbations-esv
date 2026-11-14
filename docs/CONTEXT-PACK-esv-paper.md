# ESV context pack: everything needed to write the paper

Written for teammates who will draft the ESV paper in their own Claude sessions. Paste this file in
whole, or point the session at it. It is self-contained: it states the research question, every model,
every benchmark, how the perturbation datasets were built, the full pipeline stage by stage, all
results currently in the repo with exact numbers, the metric definitions, what each table and figure
should contain, the reviewer exposure, and the open runs.

Written 2026-11-14 from the repo at commit `5ad272c` (branch `ESV`). Every number below was read from
`experiments/results/**` or recomputed from the stored records. Where a number is a recomputation
rather than a committed report, it says so.

Immediate deadline: AAAI workshop, 2026-11-20. Then ICML 2027 in January.

---

## 1. What changed since the last project summary, and why it matters

Two results landed after the project notes were last written, and both change what the paper can
claim. Read this section before writing a sentence of the paper.

**The judge-select control has been run, and it beats ESV on all three benchmarks.** Judge-select
gives the *same* checker the *same* candidates in the *same* order with vote counts hidden and the
same right to answer NONE, but asks it to reason freely and name one, with no bindings, conditions or
verdict form. It wins:

| benchmark | CoT maj@4 | ESV v3 (structured) | judge-select (free-form, same checker) | ESV vs judge-select, paired |
|---|---|---|---|---|
| GSM8K (1170 instances) | 88.5% | 93.5% | **94.1%** | 0 / 7 (McNemar p=0.016) |
| MATH-500 (976) | 56.4% | 62.3% | **66.0%** | 4 / 40 (p=2e-08) |
| JEEBench (772) | 24.0% | 27.5% | **36.7%** | pairing not printed; judge-select is +98/-0 vs maj@4 against ESV's +28/-1 |

So the structured ESV form is not the source of the gain. The honest reading: the gain comes from
putting a stronger independent checker in front of the solver's own competing candidates, and ESV's
template is, if anything, a constraint that costs accuracy and tokens. ESV spends 524 extra tokens per
GSM8K instance against judge-select's 182, and loses.

**The same-size cross-verifier control has also been run, and independence alone does not work.**
Recomputed from the stored `esv3_records.<verifier>.json` files, excluding the six disputed GSM8K
golds:

| solver | checker | maj@4 | ESV | fixes / breaks |
|---|---|---|---|---|
| Qwen 2.5 7B | Mistral Nemo 12B | 91.0% | 89.0% | 4 / 12 |
| Qwen 2.5 7B | Gemma 3 4B | 91.0% | 89.2% | 3 / 10 |
| Gemma 3 4B | Mistral Nemo 12B | 90.0% | 87.7% | 3 / 12 |
| Mistral Nemo 12B | Gemma 3 4B (383 of 390 done) | 84.9% | 82.0% | 12 / 23 |
| MATH-500: Mistral Nemo | Gemma 3 4B | 45.3% | 39.8% | 17 / 44 |
| MATH-500: Gemma 3 4B | Mistral Nemo | 67.4% | 65.8% | 12 / 20 |
| Qwen 2.5 7B as checker | partial: 550 of 1170 instances, only unanimous ones processed | 97.6% | 97.6% | 0 / 0 |

Every completed same-size cross-check loses to the majority vote. A different model is not enough: the
checker has to be better at the task, not merely other. The Qwen-as-checker runs are incomplete and
have touched only unanimous instances, so they carry no information yet; finish or discard them.

**What this means for the claim.** The sentence in the current abstract, "gold-free verification
requires a verifier other than the model under test", is not supported as written. The supported claim
is narrower in one direction and stronger in another: gold-free selection over a solver's own
candidates works only when the checker is substantially more capable than the solver, and when it does
work, the structured protocol is not what makes it work. Section 11 gives three framings that survive
review.

---

## 2. The research question and the setup

When a problem is altered so the familiar strategy no longer applies, models apply it anyway. The
project asks whether that can be caught from the problem text alone, with no answer key, and whether
catching it beats trusting the majority vote.

Four research questions, as stated in the repo README:

- RQ1: can a verifier detect an invalid strategy from the problem's own givens?
- RQ2: does that signal beat the free majority vote it has to overturn?
- RQ3: does it matter who verifies, the solver itself or another model?
- RQ4: does ESV beat Direct QA, CoT maj@4 and Universal Self-Consistency?

The design constraint that makes everything comparable: every method starts from the same four CoT
samples and none of them re-solves the problem. They differ only in which evidence they use to pick
among answers the solver already produced. Gold answers enter only at scoring, so a method can break a
correct majority as easily as repair a wrong one, and is scored that way with paired fixes and breaks
and an exact McNemar test.

| method | extra calls | evidence used to choose |
|---|---|---|
| Direct QA maj@4 | 0 | none, no reasoning produced |
| CoT pass@1 | 0 | first sample only |
| CoT maj@4 (the baseline) | 0 | the vote |
| Universal Self-Consistency (Chen et al., 2023) | 1 | the samples with duplicates shown, so it can see multiplicity |
| Judge-select (control) | 1 | same candidates and checker as ESV, free-form reasoning, no ESV form |
| ESV v3 (ours) | 1 | the problem's quoted givens; vote counts hidden from the checker |
| CoT pass@4 | 0 | oracle ceiling, needs gold |

---

## 3. Models

Solvers, small, the primary arm: `qwen/qwen-2.5-7b-instruct`, `mistralai/mistral-nemo` (12B),
`google/gemma-3-4b-it`. In code: `common.SMALL_SOLVER_MODELS`.

Solvers, large, a no-headroom control: `qwen/qwen-2.5-72b-instruct`,
`mistralai/mistral-small-3.2-24b-instruct`, `meta-llama/llama-3.3-70b-instruct`. In code:
`common.SOLVER_MODELS`, which is the default, so the primary arm needs `--models small`.

Checker for the independent condition: `openai/gpt-oss-120b`, via `run_esv3.py --verifier-model`.
Without that flag the solver checks itself, which is the condition that fails. Same-size cross-checks
pass another small solver to the same flag.

Dataset construction: generator `deepseek/deepseek-v4-flash-0731`; two independent verifiers,
`qwen/qwen3.5-35b-a3b` (model B) and `openai/gpt-oss-120b` (model C). `common.JUDGE_MODEL` is
`openai/gpt-oss-120b` and graded ESV diagnoses in the retired pilot only.

Sampling config, all in `common.py`: `NUM_SAMPLES = 4`, `SAMPLE_TEMPERATURE = 0.7`, seeds 0 to 3,
`SAMPLE_SEED = 2026` for problem sampling and the mode balancer, verification calls at T = 0,
`SOLVER_MAX_TOKENS = 4000` raised to 6000 for MATH-500, `NUMERIC_TOLERANCE = 0.01`. Everything runs
through OpenRouter; the key is read from `experiments/.env` or `RAEN_ENV_FILE`.

Run version `jee_esv3_v1_2026-11-03`, decision rule `stages1-3_strict_v1_2026-11-03`. Both are stamped
into every record.

---

## 4. Benchmarks and how the perturbation datasets were built

### 4.1 The pair structure

Every item is a frozen pair (P, P'): an original problem and a perturbed variant that changes one
structure, relationship or constraint while staying well defined, solvable and superficially similar.
P' keeps the original's answer format. Two modes, balanced near 50/50 by an adaptive balancer that
assigns each problem one mode at the start and keeps it across attempts:

- Strategy-breaking: applying the original strategy S unchanged to P' gives a wrong answer or cannot
  be completed. That wrong answer is the *trap*.
- Strategy-preserving: P' is a genuinely new instance, the answer changes and must be recomputed, but
  S still works. This is the control for "any change hurts".

Worked example, GSM8K pair 131. Original: Sofie walks twice as far as Brian over 7 days, answer 35.
Perturbed: Sofie walks twice as far as Brian walks in one day, answer 245. The comparison changes, so
the original strategy's answer is the trap. This pair is already the worked example in the deck.

### 4.2 The construction pipeline (`build_dataset.py`, prompts in `perturb_prompts.py`)

1. Sample the benchmark's n problems with seed 2026, written to `sample_<n>.json`.
2. Generator call (DeepSeek V4 Flash): name the original strategy S, change one condition, and state
   P', the new answer, the changed condition, the failure reason, an alternative strategy, a naive
   answer and a confidence level. Fields are tagged, not JSON.
3. Verifier call 1, blind solve: each verifier sees only P', not P and not S, and solves it. This is
   what prevents anchoring. The pair is rejected if the verifier calls the problem invalid, if its
   answer disagrees with the generator's, or if its confidence is below 0.6 (`confidence_threshold:
   0.6` in the frozen metadata).
4. Verifier call 2, mechanical strategy application: the verifier sees P, P', S and the verified
   answer, and applies S to P' mechanically. In breaking mode the vote is yes when S's naive answer
   differs from the verified answer; in preserving mode, yes when it matches and the status is not
   INVALID. A soft vote means S could not be carried out at all
   (`none_invalid_counts_as_break: true`).
5. Acceptance: no "no" vote from either verifier, and at least one concrete yes. A soft vote alone
   never accepts. Verifier C runs first because it is faster; B is skipped when C already rejects.
6. Rejected attempts get feedback and are retried up to a cap. Every attempt, accepted or rejected, is
   appended to `generation_log.jsonl`, and `--freeze-only` rebuilds the frozen dataset from that log.

Each frozen pair stores `id`, `source`, `source_index`, `sample_position`, `type`, `answer_format`,
`perturbation_type`, `original_problem`, `original_gold`, `perturbed_problem`, `answer_prime`,
`attempts_used`, `original_strategy`, `changed_condition`, `failure_reason`, `alternative_strategy` and
`validation` (per-model records for A, B and C). MATH-500 pairs also store `subject` and `level`.

### 4.3 The three benchmarks as frozen

| | GSM8K | MATH-500 | JEEBench |
|---|---|---|---|
| source | `openai/gsm8k`, main, test | `HuggingFaceH4/MATH-500`, test | JEE Advanced 2016-2023 math |
| usable source problems | 1317 (2 negative answers dropped) | 500 | 236 |
| sampled | 200 | 250 | 200 |
| frozen pairs | **198** | **244** | **193** |
| breaking / preserving | 99 / 99 | 123 / 121 | 97 / 96 |
| acceptance rate per mode | 1.000 / 0.980 | 0.984 / 0.968 | 0.951 / 0.980 |
| attempts used | 182 first-attempt, 16 second | 216 / 23 / 5 | 155 / 25 / 13 |
| answer formats | integer 198 | integer 145, expression 88, numeric 11 | multiple 68, numeric 55, single 41, integer 29 |
| frozen at | 2026-11-04 | 2026-11-09 | 2026-11-01 |
| status | primary, complete, 3 solvers | second benchmark, run on 2 of 3 solvers | retired dev benchmark, run on 2 of 3 solvers |

MATH-500 composition in the frozen 244: levels 1 to 5 at 20 / 43 / 58 / 61 / 62; subjects Algebra 59,
Intermediate Algebra 47, Prealgebra 44, Precalculus 30, Number Theory 27, Geometry 21, Counting and
Probability 16. All 500 source problems were kept rather than filtering to integer answers, which is
what makes the sample a sample of MATH-500 and preserves the subject mix.

The `expression` answer format is scored by `expression_answers.py`, applied identically to the gold
and to every method's answers. It escalates from canonical string equality, through case-insensitive
equality, through element-wise comparison of tuples, intervals, sets and column vectors with
delimiters required to agree, through exact numeric equality, to a guarded sympy equivalence that is
abandoned unless the LaTeX conversion consumed every backslash and left no `_`, `,`, `|`, `=` or
bracket. A decimal approximation of an exact value is deliberately never accepted. The analyzer prints
a by-format table so the expression slice can be checked against the exactly scored ones.

### 4.4 Splits, exclusions and pre-registration

- Dev: pairs 0 to 59, 357 GSM8K instances. The decision rule and the choice of independent checker
  were fixed here.
- Held-out: pairs 60 to 197, 813 GSM8K instances, never used for tuning. Reports carry `.upto60` and
  `.from60` in the file name.
- Pre-registered go/no-go, fixed before the held-out run: ESV must beat CoT maj@4 by at least 3 pp and
  beat USC, otherwise the project becomes an analysis paper.
- Disputed golds: six GSM8K originals whose published answer is wrong or does not follow from the
  text, excluded from every score by both analyzers: `40_original`, `82_original`, `95_original`,
  `105_original`, `111_original`, `189_original` (`common.DISPUTED_INSTANCES`). Perturbed siblings keep
  their verified gold. This is why GSM8K is 390 instances per solver, not 396.
- Chronology, which the paper should state plainly because it is the honest order: the dev self-check
  run was a NO-GO, the bottleneck analysis diagnosed the checker as the weak link, the independent
  checker was then confirmed on untouched held-out pairs, and the judge-select control came last and
  undercut the structural claim.

---

## 5. The pipeline, end to end

Stage 0 is shared input, produced once by `run_methods.py` and reused unchanged by every method.

| stage | what it does | calls | model | outcome |
|---|---|---|---|---|
| 0. CoT sampling | solve the problem 4 times with chain of thought | 4, T = 0.7, seeds 0-3 | solver | 4 answers; their majority is the `initial` answer, scored as CoT maj@4 |
| 1. Disagreement gate | group the 4 answers into distinct candidates by normalized value, in order of first appearance, keeping vote counts; drop unreadable answers | 0, free | none | fewer than 2 distinct candidates means `unanimous`: keep the initial answer and stop |
| 2. Contrastive verification | all distinct candidates in one call, one representative solution each, vote counts hidden; per candidate the strategy, bindings to quoted givens marked MATCHES or MISMATCH, conditions marked HOLDS or VIOLATED, then the first divergence point, then SOUND or FLAWED with failure type BINDING, METHOD or COMPUTATION, the first flawed step quoted, and the corrected fact | 1 per disagreeing instance, T = 0 | checker | a filled-in check per candidate |
| 3. Decision rule | code only: a candidate is strictly SOUND only if its verdict is SOUND and it lists zero MISMATCH and zero VIOLATED. Exactly one such candidate wins; several, the most voted among them, ties to the initial answer then the earliest; none, keep the initial answer | 0, free | none | `one_sound`, `several_sound`, `none_sound`, `unreadable` |
| 4. Diagnosis-driven repair | would re-solve where no candidate survives | n/a | n/a | not built. `none_sound` keeps the initial answer, so ESV never invents an answer the solver did not produce. Its ceiling is the "a candidate is right" column of the decisions table |
| 5. Scoring | gold enters here and nowhere else; paired against CoT maj@4 on the same instances with exact McNemar | 0 | none | the accuracy tables |

Prompt mechanics worth stating in the paper, all in `method_prompts.py`:

- Everything is tag-structured, never JSON, with strict format rules and a concision instruction of
  about 150 words per solution. Truncated or tag-missing replies get one nudge and a doubled cap.
- The verification prompt tells the checker explicitly that the solutions reach different answers, so
  at most one can be correct and all may be wrong, that they are in arbitrary order, and that how many
  attempts reached each answer is not shown and is not evidence.
- Bindings are instructed against the problem statement itself, "not the solution's restatement of it",
  and quotes must be word for word.
- Reasoning passed into later stages is clipped to 6000 characters (`CONTEXT_CHAR_LIMIT`).
- Judge-select is matched on everything except the form: same model, same candidates, same order, same
  hidden vote counts, the same licence to answer NONE, and a free-form `reasoning` tag first, so the
  comparison is structured versus unstructured reasoning rather than versus no thinking at all. Both
  choices deliberately make the control harder for ESV to beat, which is why its win is damaging.
- USC gets all four samples with duplicates shown, so it can see multiplicity, and names its pick in a
  `<selected_response>` tag resolved first as a label 1-4 and then as the answer value. An unresolved
  tag falls back to the initial answer, so the "did USC actually select" rate must be reported before
  any USC comparison. GSM8K: 9 of 303 unresolved (3.0%). MATH-500: 0 of 601.
- A prompt bug found on 2026-11-04 and fixed: the CoT and Direct instructions repeated "for MCQs give
  the option letter" and Gemma 3 4B returned a bare "A" on 138 of 168 GSM8K samples. The instruction
  now has integer, numeric and expression variants chosen by `common.is_choice_format`; MCQ wording is
  byte-identical, so JEEBench MCQ prompts are unchanged. If the paper discusses prompts, this belongs
  in the appendix rather than being left out.

---

## 6. Results

### 6.1 GSM8K, three small solvers, 198 pairs, 1170 instances

Sources: `results/gsm8k/analysis/esv3_report_qwen-2.5-7b-instruct+mistral-nemo+gemma-3-4b-it.gpt-oss-120b.md`,
and `esv3_report_small.md` for the self-check arm.

| method | accuracy | fixes / breaks vs maj@4 | McNemar p |
|---|---|---|---|
| Direct QA maj@4 | 12.1% | 7 / 900 | 2e-256 |
| CoT pass@1 | 85.1% | | |
| CoT maj@4 (baseline) | 88.5% | | |
| Universal Self-Consistency | 89.3% | 19 / 9 | 0.087 |
| ESV v3, self-check | 85.1% | 18 / 57 | 7e-06 |
| ESV v3, independent check (gpt-oss-120b) | 93.5% | 61 / 2 | 4e-16 |
| Judge-select, same checker | 94.1% | 67 / 1 | 5e-19 |
| CoT pass@4 (oracle) | 94.5% | 71 / 0 | 8e-22 |

Paired head to head: ESV vs USC 53 / 4 (p=6e-12); ESV vs judge-select 0 / 7 (p=0.016); self-check ESV
vs USC 17 / 66 (p=6e-08).

Splits, dev (357) / held-out (813): CoT maj@4 87.7 / 88.8, USC 88.5 / 89.7, ESV self 85.2 / 85.1, ESV
independent 92.2 / 94.1, pass@4 93.3 / 95.1.

Per solver, maj@4 to ESV independent: Qwen 91.0 to 94.6 (14 / 0), Nemo 84.4 to 92.8 (35 / 2), Gemma
90.0 to 93.1 (12 / 0).

By arm, maj@4 to ESV independent: originals (576) 90.6 to 94.4 (23 / 1), breaking P' (297) 79.1 to 87.5
(25 / 0), preserving P' (297) 93.6 to 97.6 (13 / 1). USC on the same arms: 91.5, 79.8, 94.6.

Stage 1 gate: 867 of 1170 unanimous (74.1%), 17 of those wrong; 303 disagreeing (25.9%), of which 185
have the majority already right (breakable), 71 are fixable (majority wrong, some candidate right) and
47 have no right candidate. Per-solver disagreement rates differ sharply: Qwen 18.7%, Nemo 41.8%,
Gemma 17.2%.

What the wrong answers are, over 135 wrong maj@4 answers: 54 on originals (40.0%), 19 on preserving P'
(14.1%), 25 the trap on breaking P' (18.5%), 37 other wrong on breaking P' (27.4%). The trap appears in
90 of 1188 individual samples (7.6%).

Decisions, independent checker: `unanimous` 867 at 98.0% unchanged; `one_sound` 231, 70.6% to 96.1%,
61 fixes and 2 breaks; `none_sound` 72 at 30.6% unchanged, of which 30 do have a right candidate
available, which is stage 4's ceiling.

Selection quality on the 303 disagreeing instances: majority vote 61.1%, uniform random candidate
37.0%, USC 64.4%, ESV 80.5%, judge-select 82.8%, oracle 84.5%.

Judge-select stratified by what ESV decided: on `one_sound` (231) judge 97.0 vs ESV 96.1, paired 0 / 2
(p=0.500); on `none_sound` (72) judge 37.5 vs ESV 30.6, paired 0 / 5 (p=0.062). Judge-select returned a
label on 255 of 303 (84.2%, correct 96.5%), NONE on 46 (15.2%, correct 6.5%), ambiguous on 2.

### 6.2 MATH-500, two small solvers (Nemo, Gemma), 244 pairs, 976 instances

Qwen 2.5 7B has not been run on MATH-500. Sources:
`results/math500/analysis/esv3_report_small.gpt-oss-120b.md` and `esv3_report_small.md`.

| method | accuracy | fixes / breaks | p |
|---|---|---|---|
| Direct QA maj@4 | 10.7% | 22 / 468 | 6e-110 |
| CoT pass@1 | 51.6% | | |
| CoT maj@4 | 56.4% | | |
| Universal Self-Consistency | 56.8% | 34 / 30 | 0.708 |
| ESV v3, self-check | 54.1% | 28 / 50 | 0.017 |
| ESV v3, independent check | 62.3% | 59 / 1 | 1e-16 |
| Judge-select | 66.0% | 99 / 5 | 1e-23 |
| CoT pass@4 (oracle) | 68.8% | 121 / 0 | 8e-37 |

ESV vs USC 71 / 17 (p=5e-09). ESV vs judge-select 4 / 40 (p=2e-08).

Gate: 375 of 976 unanimous (38.4%), 15 of them wrong; 601 disagreeing (61.6%), of which 190 breakable,
121 fixable, 290 with no right candidate. Per-solver disagreement: Nemo 80.1%, Gemma 43.0%.

By arm: originals (488) 57.6 to 64.3 (33 / 0), breaking (246) 43.9 to 50.0 (16 / 1), preserving (242)
66.5 to 70.7 (10 / 0).

By answer format: expression (352) 40.6 to 46.0 (20 / 1), integer (580) 65.5 to 71.9 (37 / 0), numeric
(44) 61.4 to 65.9 (2 / 0). The expression slice moves with the others, which is the evidence that the
LaTeX scorer is not what separates the methods.

Decisions: `unanimous` 375 at 96.0%; `one_sound` 187, 59.4 to 89.3 (57 / 1); `several_sound` 34, 38.2 to
44.1 (2 / 0); `none_sound` 379 at 17.4% unchanged with 116 right candidates available; `unreadable` 1.
Note how much larger `none_sound` is here than on GSM8K, 63% of disagreeing instances, and that this is
exactly where judge-select wins: 26.4% vs 17.4%, paired 2 / 36 (p=5e-09).

Trap share is small: 24 of 426 wrong answers (5.6%). On a competition benchmark the failures are mostly
capability rather than misapplied strategy, which is the same diagnosis that retired JEEBench.

Selection quality on 601 disagreeing: majority 31.6%, random 20.0%, USC 32.3%, ESV 41.3%, judge-select
47.3%, oracle 51.7%.

Judge-select returned NONE on 299 of 601 (49.8%, correct 4.0%), a label on 278 (46.3%, correct 89.6%),
ambiguous on 24 (95.8%).

### 6.3 JEEBench, two small solvers, 193 pairs, 772 instances

Retired as a dev benchmark, but the runs exist and are usable as a third point. Source:
`results/jeebench/analysis/esv3_report_mistral-nemo+gemma-3-4b-it.gpt-oss-120b.md`.

| method | accuracy | fixes / breaks | p |
|---|---|---|---|
| Direct QA maj@4 | 11.9% | 57 / 150 | 8e-11 |
| CoT pass@1 | 20.5% | | |
| CoT maj@4 | 24.0% | | |
| Universal Self-Consistency | 22.0% | 32 / 47 | 0.115 |
| ESV v3, self-check | 21.8% (recomputed from records) | 33 / 50 | |
| ESV v3, independent check | 27.5% | 28 / 1 | 1e-07 |
| Judge-select | 36.7% | 98 / 0 | 6e-30 |
| CoT pass@4 (oracle) | 41.8% | | |

ESV vs USC 69 / 27 (p=2e-05). Detector with the independent checker: right candidates SOUND 70 of 280
(25.0%), wrong 9 of 1760 (0.5%), precision 88.6%. The checker is extremely conservative here, which is
why ESV recovers only 3.5 of the 17.8 points of headroom while judge-select takes 12.7.

Note for the abstract: the current JEEBench headroom is 17.8 pp (41.8 minus 24.0), not 18.5. Fix that
number or trace it to whichever run produced it.

### 6.4 Large solvers, GSM8K, stage 1 only

Source: `results/gsm8k/analysis/esv3_report.md`. CoT maj@4 96.9%, pass@4 99.0%, so 2.1 pp of headroom.
Only 88 of 1170 instances disagree (7.5%), 24 of them fixable; 1 unanimous-but-wrong in 1082. Per
solver maj@4: Qwen 72B 96.9%, Mistral Small 24B 96.7%, Llama 3.3 70B 97.2%. Stage 2 was never run on
this arm. Report it as a no-headroom control and as the reason the small-solver arm is the test, not as
a failed experiment.

### 6.5 The mechanism: why self-verification fails

This is the most publishable analysis in the repo, from
`results/gsm8k/analysis/esv3_bottleneck_gsm8k.md`.

The vote is already a strong prior. On dev, a candidate with 1 / 2 / 3 votes is correct 9.7% (15/154),
51.1% (24/47) and 77.8% (28/36) of the time. So a verdict must carry a likelihood ratio above:

| to lift | needs LR |
|---|---|
| a 2-vote candidate over a 3-vote majority | > 3.4 |
| a 1-vote candidate over a 2-vote majority | > 9.7 |
| a 1-vote candidate over a 3-vote majority | > 32.4 |

Measured detector strength, per candidate:

| condition | right called SOUND | wrong called SOUND | precision | LR |
|---|---|---|---|---|
| GSM8K, self-check | 57.0% (146/256) | 36.5% (180/493) | 44.8% | 1.6 |
| GSM8K, gpt-oss-120b | 86.7% (222/256) | 1.8% (9/493) | 96.1% | 47.5 |
| MATH-500, self-check | 56.3% (175/311) | 43.3% (672/1552) | 20.7% | 1.3 |
| MATH-500, gpt-oss-120b | 60.8% (189/311) | 5.2% (80/1552) | 70.3% | 11.7 |
| JEEBench, gpt-oss-120b | 25.0% (70/280) | 0.5% (9/1760) | 88.6% | 49 |

At LR 1.64 the self-check is 6 to 20 times too weak everywhere except a tied vote, and at three votes
SOUND is actually negatively informative (80.0% correct when FLAWED against 76.2% when SOUND). Nine
alternative stage 3 rules were evaluated offline on the stored checks, plus a Bayes-optimal selector
fitted on the same data as an optimistic upper bound. Both cap at +1.1 pp, below the pre-registered
bar; the best rule is "SOUND breaks vote ties only", 4 fixes and 0 breaks. The conclusion is
quantitative rather than rhetorical: the missing accuracy is information the checker does not produce,
not a rule-design problem.

Why the checker fails while looking straight at the evidence: on the 41 disagreeing breaking-P' verify
calls, the checker's text engages the changed condition in 83% of calls, measured as at least 35%
content-word overlap with the pair's `changed_condition`. It quotes the perturbed given and then
ratifies the solution that misused it. Two transcripts to quote in the paper:

- GSM8K pair 4, Sara mowing, where the perturbed clause pays nothing for the first four mowings. The
  checker quotes "$4 for each time she mows the lawn after the first 4 times" verbatim, then marks the
  trap solution's `$4x`, which pays for every mowing, as MATCHES, verdict SOUND.
- GSM8K pair 36, a relay where the first leg is shortened to 380 m with nominal times unchanged. The
  checker writes "the first runner runs 380 m in 60 s, HOLDS", calls the trap's per-metre rescaling
  "correctly calculates", and its divergence section argues for the wrong side.

The claim this supports: judging whether a solution uses a quoted given correctly is a reasoning task
of the same difficulty as solving the problem, and the same model fails it in the same direction,
because it holds the same prior that made the trap tempting. There is no generator-verifier gap to
exploit inside one model. The same-size cross-verifier results in section 1 extend this: there is no
usable gap between two models of similar capability either.

### 6.6 Compute

GSM8K, pooled, per instance, read from the stored `usage` and `latency_seconds`:

| method | calls | solver tokens | checker tokens | total | vs maj@4 | critical path |
|---|---|---|---|---|---|---|
| Direct QA maj@4 | 4.00 | 1,095 | - | 1,095 | -54.0% | 1.0 s |
| CoT pass@1 | 1.00 | 595 | - | 595 | -75.0% | 4.8 s |
| CoT maj@4 | 4.00 | 2,379 | - | 2,379 | 0.0% | 6.5 s |
| USC | 4.26 | 2,799 | - | 2,799 | +17.6% | 6.7 s |
| Judge-select | 4.26 | 2,379 | 492 | 2,871 | +20.7% | 10.0 s |
| ESV v3 | 4.26 | 2,379 | 1,038 | 3,417 | +43.6% | 13.7 s |

Extra tokens per net fix: USC 49,112; judge-select 8,727; ESV 20,585. Judge-select is both more
accurate and 2.4 times cheaper per fix than ESV.

The gate is the efficiency mechanism and deserves its own paragraph. It is free, being code over
answers already paid for, and it fires on 25.9% of GSM8K instances, so it avoids 74.1% of verification
spend; ungated, the same protocol would cost +168% instead of +43.6% and gain nothing, because a
unanimous instance has no second candidate to choose between. A verification call on an instance that
fires costs 4,008 tokens and 27.8 s; amortized it is 1,038 tokens and 7.2 s. The spend is adaptive: the
gate fires on 45.1% of breaking P' against 17.5% of preserving P', so ESV spends most where the
accuracy gap is largest, +72.7% tokens on the breaking arm against +28.0% on preserving.

Price sensitivity, with k the checker's price per token over the solver's: ESV costs less per net fix
than USC for any k below 2.4 on GSM8K and 6.0 on MATH-500. Above that USC buys a fix more cheaply, but
USC's total gain is capped at +10 net fixes against ESV's +59, so more USC cannot reach ESV's accuracy
at any price. State this caveat rather than letting a reviewer find it.

MATH-500 is much more expensive because the gate fires on 61.6% of instances: maj@4 4,592 tokens, USC
6,159 (+34.1%), judge-select 6,826 (+48.7%), ESV 8,399 (+82.9%); per net fix USC 382,293, judge-select
23,197, ESV 64,060.

---

## 7. Metric definitions, so every number in the paper means one thing

- Instance: one problem, either an original P or a perturbed P'. A pair contributes 2 instances per
  solver; pooled counts multiply by the number of solvers.
- CoT maj@4: the majority answer over the 4 CoT samples, grouped by normalized value, ties to the
  earliest. This is the `initial` field on every record and the baseline for every comparison.
- CoT pass@1: the first sample's answer. CoT pass@4: correct if any of the 4 samples is correct, an
  oracle ceiling that needs gold.
- Fixes and breaks: paired against CoT maj@4 on the same instances. A fix is maj@4 wrong and the method
  right; a break is the reverse. Reported as fixes / breaks with an exact McNemar p over the discordant
  pairs. Accuracy deltas should never be reported without them.
- Headroom: pass@4 minus maj@4. GSM8K 6.1 pp, MATH-500 12.4 pp, JEEBench 17.8 pp with small solvers;
  2.1 pp for large solvers on GSM8K.
- Fixable instance: disagreeing, majority wrong, at least one candidate right. Breakable: disagreeing,
  majority already right.
- Detector rates: per candidate, not per instance. Right candidates called SOUND, wrong candidates
  called SOUND, precision of a SOUND verdict, and the likelihood ratio between the first two.
- Strictly SOUND: verdict SOUND and zero MISMATCH and zero VIOLATED. The reports also print a
  verdict-only variant; on GSM8K the two coincide.
- Trap: the answer the original strategy yields when applied unchanged to a breaking P'. Stored per
  pair as the generator's `naive_answer` and confirmed by both verifiers.
- Selection quality: accuracy restricted to disagreeing instances, the only population where a selector
  can act, with a uniform-random-candidate row as the chance floor.
- Tokens: prompt plus completion from `usage`, split into solver and checker. Serial seconds are billed
  compute; critical path is user-visible latency, since the 4 samples are issued concurrently and the
  extra call cannot start until all 4 return.
- Disputed instances are dropped from every score, so n = 390 per solver on GSM8K.

---

## 8. Tables and figures the paper needs

Nothing here requires new runs. Each item names its source.

1. Main results table: methods by benchmark, accuracy with fixes / breaks and McNemar p, one block per
   benchmark, pass@4 as the ceiling row, judge-select included. Sources: sections 6.1 to 6.3.
2. Who-verifies table, the centrepiece: four checker conditions (self, same-size cross, strong
   independent, and judge-select as the form control) by benchmark. Sources: sections 1 and 6.5.
3. Detector table: right-SOUND, wrong-SOUND, precision and LR per condition per benchmark, placed next
   to the required-LR thresholds from the vote prior. Source: section 6.5.
4. Vote-prior figure: P(correct) against vote count, with SOUND and FLAWED curves overlaid for
   self-check and for the independent checker. The self-check curves cross at 3 votes, which is the
   visual form of the whole argument. Data in `esv3_bottleneck_gsm8k.md`.
5. Decision-mix table: `unanimous`, `one_sound`, `several_sound`, `none_sound`, with n, maj@4, ESV,
   fixes, breaks and "a candidate is right", per benchmark, plus the stage 4 ceiling.
6. By-arm table: originals, breaking P', preserving P'. The preserving arm answers "does any change
   hurt", so it belongs in the main paper rather than the appendix.
7. Compute table plus price-sensitivity rows, and the gate paragraph with the 74.1% avoided spend.
8. MATH-500 by-answer-format table, as the scoring-validity check.
9. Rule-sweep table: the nine alternative stage 3 rules and the overfit Bayes-optimal bound, capping at
   +1.1 pp. This forecloses "you just need a better decision rule".
10. Two transcript boxes: GSM8K pairs 4 and 36, the quoted-then-ratified failures.
11. Dataset table: the three benchmarks as in section 4.3, with acceptance rates, plus the pair 131
    worked example.
12. Existing JEEBench figures are in `results/jeebench/analysis/`: `fig_accuracy_overall.png`,
    `fig_accuracy_perturbed.png`, `fig_esv_funnel.png`, `fig_recovery_by_perturbation.png`. They belong
    to the retired design; regenerate equivalents for GSM8K rather than reusing them.

---

## 9. Reviewer exposure, with current status

Ordered by how much damage each does.

1. The structure does not earn its keep. Judge-select beats ESV on all three benchmarks with fewer
   tokens. Status: measured, unavoidable. Do not frame ESV's form as the contribution; report it as a
   negative result about structured verification templates, which is publishable and honest.
2. Checker capability was confounded with independence. Status: now partly resolved, and against the
   original framing. Same-size cross-checks all lose; only the much larger checker wins. State it as
   the finding instead of as a limitation.
3. Compute is not matched. A reviewer will ask whether sampling the solver more (maj@8, maj@16) at
   equal spend beats one checker call. Status: not run, and the stored `records.json` cannot answer it
   because it holds only 4 samples. This is the most valuable remaining experiment.
4. gpt-oss-120b answering alone is not reported. If the 120B checker simply solves GSM8K at 95% or
   better, the practical framing becomes "route to a bigger model" and the paper must argue cost rather
   than accuracy. Status: not run, cheap, must be in the paper.
5. Trap share is small outside GSM8K: 18.5% of wrong answers on GSM8K, 5.6% on MATH-500. The premise
   that failures are misapplied strategy holds weakly. Status: measured; state it and scope the
   strategy-perturbation story to GSM8K.
6. Single-seed sampling: one set of 4 samples at T = 0.7, seeds 0 to 3, no repeats. Status: not run.
   Either report as a limitation or re-run stage 0 with seeds 4 to 7 on one solver as a stability check.
7. MATH-500 is missing Qwen 2.5 7B, so its pooled numbers are over two solvers while GSM8K's are over
   three. Status: one run away. Fix it or say so in the table caption.
8. LLM-generated benchmark: perturbations are produced and verified by models, and acceptance rates are
   95 to 100%, so filtering is light. Status: 19 GSM8K pairs were hand-solved and correct, and 6
   disputed originals were excluded by hand. Report the hand-audit numbers, and consider auditing 20
   random MATH-500 pairs before submission.
9. Contamination of GSM8K and MATH-500 in the solvers' training data is not discussed. The perturbed
   arm is the partial answer, so make that argument explicitly.
10. Stage 4 is not built, so ESV cannot create an answer. Status: by design, with the ceiling reported
    per decision bucket. Keep it as a stated scope limit.
11. The abstract's JEEBench headroom (18.5) does not match the current report (17.8). Fix before
    submission, and re-derive every number in the abstract from the committed reports.

---

## 10. Open runs, with the exact commands

Offline, no API cost, do these first:

```bash
python experiments/tests/smoke_test_offline.py
python experiments/tests/test_expression_answers.py
python experiments/analyze_esv3.py --benchmark gsm8k   --models small --verifier-model openai/gpt-oss-120b
python experiments/analyze_esv3.py --benchmark gsm8k   --models small --verifier-model openai/gpt-oss-120b --min-pair 60
python experiments/analyze_esv3.py --benchmark math500 --models small --verifier-model openai/gpt-oss-120b
# the cross-verifier arms have records but no committed report:
python experiments/analyze_esv3.py --benchmark gsm8k --models small --verifier-model mistralai/mistral-nemo
python experiments/analyze_esv3.py --benchmark gsm8k --models small --verifier-model google/gemma-3-4b-it
```

Live runs still needed, in priority order:

1. Compute-matched baseline: 4 more CoT samples per GSM8K instance for the three small solvers, then
   maj@8 and pass@8, compared against one checker call at equal spend.
2. `gpt-oss-120b` solving GSM8K and MATH-500 directly, 4 samples, for the "just use the bigger model"
   row.
3. Qwen 2.5 7B on MATH-500: `run_methods.py --benchmark math500 --models qwen/qwen-2.5-7b-instruct`,
   then `run_esv3.py --benchmark math500 --verifier-model openai/gpt-oss-120b --judge-select`.
4. Finish or discard the Qwen-as-checker cross-verifier runs (currently 550 of 1170 instances, all
   unanimous, hence 0 fixes and 0 breaks and no information).
5. Optional: seeds 4 to 7 on one solver as a sampling-stability check.

Budget: read the spendable amount from `GET https://openrouter.ai/api/v1/key`, field
`data.limit_remaining`, not `/api/v1/credits`, which reports the whole organization pool. The key in
`experiments/.env` is capped at $250. Exhaustion arrives as a per-call HTTP 402, which the runners treat
as retryable and defer to the next run.

---

## 11. Three framings that survive review

Pick one before drafting. All three are supported by what is already measured.

A. The verifier-capability paper, recommended for the workshop. Direction: who verifies decides
whether gold-free selection beats majority voting. The contribution is the four-condition sweep (self,
same-size cross, strong independent, plus the free-form form control) across three benchmarks, the
likelihood-ratio criterion that says in advance how strong a verdict must be to overturn a k-vote
majority, and the transcript-level mechanism for why self-verification fails while quoting the right
evidence. ESV becomes the probe rather than the product, and the judge-select result is a finding
inside this paper rather than a wound.

B. The negative-result paper on structured verification. Structured, form-filled verification does not
beat free-form judgement by the same model, and costs 2 to 3 times the tokens. Supported on three
benchmarks with matched candidates, order, hidden votes and the same licence to decline. Pair it with
the rule sweep showing no decision rule recovers the gap, and the LR arithmetic explaining why.

C. The practical selection paper. A free disagreement gate plus one call to a stronger checker buys
+5.0 pp on GSM8K, +5.9 on MATH-500 and +3.5 on JEEBench over maj@4 at +43.6% tokens, while USC buys
nothing significant anywhere. Here judge-select is the method and ESV is the ablation, and the honest
headline is the gate plus a strong checker rather than the protocol. This framing needs the
compute-matched baseline from section 10 before it is reviewable.

What cannot be claimed: that ESV's structured form helps; that independence alone suffices; that the
result holds for strong solvers, which are at ceiling on GSM8K; or that this is specifically about
strategy perturbation, given that 40% of GSM8K failures and 48.6% of MATH-500 failures are on untouched
originals.

---

## 12. Repo map, gotchas and reproduction

Current code lives in `experiments/`. The pilot in `pilot_experiment/src/` is archived, predates
`common.py`, and should be treated as history.

| file | role |
|---|---|
| `common.py` | single source of truth: models, `NUM_SAMPLES`, temperature, seed, benchmark table, paths, OpenRouter client, tagged-output calls, type-aware scoring, exact McNemar, disputed instances, resumable I/O |
| `benchmarks.py` | normalizes each benchmark to one shape; pulls and caches GSM8K and MATH-500 |
| `expression_answers.py` | LaTeX normalization and equivalence for MATH-500's expression answers |
| `perturb_prompts.py` | generator and verifier prompts for both perturbation modes, format preserving |
| `build_dataset.py` | sample, perturb, verify, balance modes, freeze; `--freeze-only` rebuilds from the log |
| `method_prompts.py` | Direct QA, CoT, ESV v3 verification, USC and judge-select prompts |
| `run_methods.py` | the 4 Direct QA and 4 CoT samples every method starts from |
| `run_esv3.py` | ESV v3 and its controls; `--verifier-model` for an independent checker, `--judge-select` for the structure control, added to existing records without repeating the ESV call |
| `analyze_esv3.py` | every report, offline from stored records; `--min-pair` and `--max-pair` select the split |
| `tests/smoke_test_offline.py` | every step on all three benchmarks against a fake client, no API calls |

Data at `experiments/data/<benchmark>/`: `source.json`, `sample_<n>.json`, `generation_log.jsonl`,
`frozen_dataset.json`. Results at `experiments/results/<benchmark>/<solver>/`: `records.json` (4 DQA and
4 CoT samples), `esv3_records.json` (self-check plus the USC arm), `esv3_records.<verifier>.json` (one
per checker). Reports in `experiments/results/<benchmark>/analysis/`. About 140 MB of raw records are
committed deliberately, so every table regenerates offline.

Gotchas that will cost a day if missed:

- Paths must be read through `common.DATA_DIR`, `common.RESULTS_DIR` and `common.FROZEN_DATASET`, never
  imported as module constants, because `--benchmark` rebinds them.
- `reusable` in `run_methods.py` keys on run version, model and gold, not the prompt. A prompt change
  needs `--no-resume`, which backs up the old file, or a `RUN_VERSION` bump; otherwise prompt variants
  mix silently.
- USC is measured once, in the self-verify run, because it asks the solver and has no checker. In a
  `--verifier-model` report the USC row is read from `esv3_records.json`. Never compare against a USC
  row taken from a verifier-mode file.
- Model outputs come from hosted OpenRouter providers and may drift. The frozen datasets and every raw
  response are committed, so tables regenerate exactly offline; live re-runs approximate rather than
  bit-reproduce them.
- There is no pinned `requirements.txt` yet. Python 3.12 with `openai`, `python-dotenv`, `tqdm`,
  `sympy`, and `datasets` only for pulling sources.
- Never commit API keys. `.env` and `env` are git-ignored.

Key references already collected: GSM8K (Cobbe et al., 2021); MATH (Hendrycks et al., 2021) and the
MATH-500 subset from Let's Verify Step by Step (Lightman et al., 2023, arXiv:2305.20050); JEEBench
(Arora et al., 2023, arXiv:2305.15074); Universal Self-Consistency (Chen et al., 2023); Large Language
Models Cannot Self-Correct Reasoning Yet (Huang et al., 2024, arXiv:2310.01798). The last is the closest
prior work to the self-check result and must be cited where that result is stated.

---

## 13. Writing notes

The current abstract draft is now out of date in two ways: it claims verifier independence rather than
verifier capability, and its JEEBench headroom figure is 18.5 where the report says 17.8. Rewrite it
against section 11 and re-derive every number from the committed reports.

On style, since the drafts so far have read as machine-written: detectors score polished technical
prose above 80% "AI" regardless of authorship, so the score is not worth chasing, but uniformity is a
real tell. Avoid an epigram opening, three aphoristic sentence-endings in a row, the same triadic
benchmark rhythm repeated, em dashes as the universal connector, and bold labels on list items. Name
the models, give exact instance counts, state k and the temperature. Specificity reads human; polish
does not.
