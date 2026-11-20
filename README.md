<div align="center">

# Explicit Strategy Verification (ESV)

**Gold-free answer selection without re-solving.**

![python](https://img.shields.io/badge/python-3.12-3776AB)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

</div>

Self-consistency picks the majority answer among sampled chains of thought, so it selects by answer
frequency alone and is blind to whether any candidate's reasoning is valid. For small open-weight
solvers, the best answer *already present* in four CoT samples beats their majority vote by 6 points
on GSM8K, 12 on MATH-500 and 18 on JEEBench.

**ESV** closes part of that gap with one extra call and no re-solving: a free disagreement gate, one
contrastive verification call that makes each candidate bind its strategy to quoted givens and check
its preconditions, and a deterministic rule that accepts only candidates with no mismatched binding
and no violated condition. Gold answers enter only at scoring, so a method can damage the baseline as
easily as improve it and is scored that way (paired fixes / breaks, exact McNemar).

Every problem is paired with a **structural perturbation** that either breaks the strategy the
original invites or preserves it — a design that manufactures selection errors of known cause.

Paper source: [`docs/paper/`](docs/paper/) (AAAI-27 workshop, in preparation).

---

## Results

2,918 instances; small solvers (Qwen2.5-7B, Mistral-NeMo-12B, Gemma-3-4B); 4 CoT samples at T = 0.7
shared by every method; checker `gpt-oss-120b`. *F/B* = fixes / breaks vs CoT maj@4 on the same instances.

| Method | Calls | GSM8K (1,170) | F/B | MATH-500 (976) | F/B | JEEBench (772) | F/B |
|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | 4 | 12.1% | 7/900 | 10.7% | 22/468 | 11.9% | 57/150 |
| CoT pass@1 | 1 | 85.1% | – | 51.6% | – | 20.5% | – |
| CoT maj@4 (baseline) | 4 | 88.5% | – | 56.4% | – | 24.0% | – |
| Universal Self-Consistency | 4+1 | 89.3% | 19/9 | 56.8% | 34/30 | 22.0% | 32/47 |
| ESV, self-check | 4+1 | 85.1% | 18/57 | 54.1% | 28/50 | 21.8% | 33/50 |
| **ESV, `gpt-oss-120b`** | 4+1 | **93.5%** | 61/2 | **62.3%** | 59/1 | **27.5%** | 28/1 |
| Judge-select (form control) | 4+1 | 94.1% | 67/1 | 66.0% | 99/5 | 36.7% | 98/0 |
| CoT pass@4 (oracle) | 4 | 94.5% | 71/0 | 68.8% | 121/0 | 41.8% | 138/0 |

ESV gains **+5.0 / +5.9 / +3.5 points** over CoT maj@4 (p ≤ 1e-7 on all three); USC reaches
significance on none. It passed the pre-registered criterion (≥ +3 pp over maj@4 *and* beat USC),
scoring 94.1% on the 813 held-out GSM8K instances against 88.8% for maj@4.

**Who checks decides the sign.** Swapping the checker for the solver itself reverses the gain on every
benchmark (−3.3 / −2.3 / −2.2 pp). A SOUND verdict carries a likelihood ratio of 1.6 (self) against
47.5 (`gpt-oss-120b`) on GSM8K, while overturning a 3–1 vote needs Λ > 32. The shortfall is in the
evidence, not the decision rule.

**The structured form is not what pays.** Dropping it for free-form judge-select *raises* the gain
everywhere (+5.6 / +9.6 / +12.7). Where ESV's rule commits to a candidate the two agree (2 of 231
GSM8K instances differ); the form decides only where ESV abstains, and what it buys is
interpretability: on GSM8K, 75.6% of trap candidates are labelled BINDING errors against 55.7% of
other wrong candidates (OR 2.46, p = 0.014).

**Cost.** The gate is free and fires on 25.9% / 61.6% / 89.9% of instances, so ESV adds +43.6% /
+82.9% / +100.3% tokens rather than a multiple of the baseline. Ungated, GSM8K overhead would be +168%
for no gain.

Full tables, p-values and per-arm breakdowns: [`experiments/results/<benchmark>/analysis/`](experiments/results/).

---

## Method

Starts from the 4 CoT samples already in `records.json`; never re-solves.

1. **Disagreement gate** (0 calls). Group the 4 answers into distinct candidates. All agree → keep the
   majority answer and stop.
2. **Contrastive verification** (1 call, T = 0, vote counts hidden). Per candidate: the strategy, its
   bindings to quoted givens (MATCHES / MISMATCH), the method's conditions (HOLDS / VIOLATED), the
   divergence point, then SOUND / FLAWED with a failure type (BINDING / METHOD / COMPUTATION).
3. **Decision rule** (code only). *Strictly SOUND* = verdict SOUND **and** zero MISMATCH **and** zero
   VIOLATED. One → take it; several → most votes; none → keep the initial answer.
4. **Diagnosis-driven repair** is **not built**, so ESV never invents an answer the solver did not
   produce. Its ceiling is CoT pass@4.

### The perturbation benchmark

Each item is a frozen pair **(P, P′)**: an original and a variant that changes one structure,
relationship or constraint while staying well-defined, solvable and superficially similar.
*Strategy-breaking* P′ makes the original strategy give a wrong answer (the **trap**);
*strategy-preserving* P′ is a new instance the same strategy still solves, controlling for "any change
hurts". A generator model names the strategy and changes one condition; two independent verifiers
solve P′ blind and mechanically apply the old strategy, and a pair is accepted only if the intended
mode is confirmed.

| | GSM8K | MATH-500 | JEEBench |
|---|---|---|---|
| Sampled (seed 2026) → frozen pairs | 200 → 198 | 250 → 244 | 200 → 193 |
| Breaking / preserving | 99 / 99 | 123 / 121 | 97 / 96 |
| Solvers → scored instances | 3 → 1,170 | 2 → 976 | 2 → 772 |
| Disagreement rate | 25.9% | 61.6% | 89.9% |
| Headroom (pass@4 − maj@4) | 6.1 pp | 12.4 pp | 17.9 pp |

GSM8K is integer-only; MATH-500 keeps all formats (145 integer, 88 LaTeX expression, 11 numeric),
scored by [`expression_answers.py`](experiments/expression_answers.py). GSM8K was the pre-registered
benchmark — decision rule and checker fixed on pairs 0–59, evaluated once on pairs 60–197; six
originals with disputed gold are excluded everywhere. JEEBench was the development benchmark for an
earlier pipeline, so it is a third data point rather than held-out evidence.

---

## Quick start

```bash
pip install openai python-dotenv tqdm sympy datasets   # Python 3.12
```

**Offline — no API key, no cost.** Every table in this repo regenerates from the stored raw responses.

```bash
python experiments/tests/smoke_test_offline.py          # every pipeline step, fake client, temp dirs only
python experiments/tests/test_expression_answers.py     # expression scorer, incl. all 500 MATH-500 golds

python experiments/analyze_esv3.py --benchmark gsm8k --models small --verifier-model openai/gpt-oss-120b
python experiments/analyze_esv3.py --benchmark gsm8k --models small --verifier-model openai/gpt-oss-120b --min-pair 60  # held-out
```

**Live run.** Put `OPENROUTER_API_KEY=...` in `experiments/.env` or point `RAEN_ENV_FILE` at one.
Every step resumes where it stopped.

```bash
B=gsm8k   # or math500, jeebench
S="qwen/qwen-2.5-7b-instruct mistralai/mistral-nemo google/gemma-3-4b-it"

python experiments/benchmarks.py $B                                   # 1. pull / summarize source
python experiments/build_dataset.py --benchmark $B --limit 10         # 2. small live check, then:
python experiments/build_dataset.py --benchmark $B                    #    freeze all pairs
python experiments/run_methods.py --benchmark $B --models $S           # 3. the 4 DQA + 4 CoT samples
python experiments/analyze_esv3.py --benchmark $B --models small       #    stage 1: check headroom first
python experiments/run_esv3.py --benchmark $B --models $S \
    --verifier-model openai/gpt-oss-120b                              # 4. ESV + USC control
python experiments/run_esv3.py --benchmark $B --models $S \
    --verifier-model openai/gpt-oss-120b --judge-select               #    form control
python experiments/analyze_esv3.py --benchmark $B --models small \
    --verifier-model openai/gpt-oss-120b                              # 5. reports
```

Check stage 1 before paying for the checker: no disagreement, or no fixable disagreement, means no
headroom. Omit `--verifier-model` for the self-check condition.

Hosted providers drift and sampling is at T = 0.7, so a live re-run approximates rather than
bit-reproduces the committed records.

---

## Repository layout

```text
experiments/            current code, data and results  -- see experiments/README.md for file-by-file detail
  common.py             config (models, n=4, T, seeds), OpenRouter client, scoring, exact McNemar
  benchmarks.py         loads / caches source problems per benchmark
  build_dataset.py      sample -> two-stage perturbation -> data/<benchmark>/frozen_dataset.json
  run_methods.py        the 4 Direct QA + 4 CoT samples every method starts from
  run_esv3.py           ESV + USC and judge-select controls (--verifier-model, --judge-select)
  analyze_esv3.py       offline reports (--min-pair / --max-pair select the split)
  expression_answers.py LaTeX normalization + equivalence for MATH-500
  method_prompts.py     Direct QA, CoT, ESV, USC prompts;  perturb_prompts.py  dataset prompts
  data/ results/        frozen datasets, generation logs, per-solver records, analysis reports
docs/paper/             AAAI-27 workshop paper source and figures
pilot_experiment/       archived Oct 2026 pilot: 5-stage ESV, JEEBench, one solver
```

Where to read next: [`experiments/README.md`](experiments/README.md) for the run order and file roles;
[`esv3_bottleneck_gsm8k.md`](experiments/results/gsm8k/analysis/esv3_bottleneck_gsm8k.md) for the
dev NO-GO → diagnosis → held-out GO narrative;
[`esv3_report_small.gpt-oss-120b.md`](experiments/results/gsm8k/analysis/esv3_report_small.gpt-oss-120b.md)
for stage-by-stage tables; [`method_prompts.py`](experiments/method_prompts.py) and
[`run_esv3.py`](experiments/run_esv3.py) for the exact prompt and decision rule.

## Limitations

- **Selection only.** ESV chooses among existing candidates, so CoT pass@4 is a hard ceiling and
  stage-4 repair is not built. Abstentions grow with difficulty (72 → 379 → 612 instances).
- **Checker strength is confounded with independence.** `gpt-oss-120b` is larger than the solvers, so
  the experiments do not separate "a stronger checker" from "a different checker".
- **The form earns interpretability, not accuracy.** Free-form judge-select is stronger on all three
  benchmarks.
- **Benchmark is model-generated.** Perturbations are produced and verified by models (acceptance
  0.95–1.00 per mode, so rejection filtering is light). The checker also verified every frozen P′
  during construction; gains on untouched originals (+3.8 GSM8K, +6.7 MATH-500) argue this does not
  drive the result.
- **Strategy-misfit reading is GSM8K-only.** The trap explains 18.5% of wrong majority answers there
  but 5.6% on MATH-500.
- One sample set per instance, so seed variance is unmeasured. No pinned `requirements.txt` yet.
- All models are served through [OpenRouter](https://openrouter.ai), so provider-side changes can move
  the live numbers.

## Citation and license

A paper is in preparation; until a preprint exists, please cite this repository.
Licensed under the [Apache License 2.0](LICENSE).

**Key references.** GSM8K (Cobbe et al., 2021); MATH (Hendrycks et al., 2021) and the MATH-500 subset
of *Let's Verify Step by Step* (Lightman et al., 2023); JEEBench ([Arora et al.,
2023](https://arxiv.org/abs/2305.15074)); Universal Self-Consistency (Chen et al., 2023); *LLMs Cannot
Self-Correct Reasoning Yet* ([Huang et al., 2024](https://arxiv.org/abs/2310.01798)).
