# ESV experiments: can a checker fix reasoning failures under structural perturbation?

One code base, several benchmarks. Every script takes `--benchmark` (default `jeebench`) and reads / writes
`data/<benchmark>/` and `results/<benchmark>/`.

## Where this stands

**GSM8K is a GO.** ESV v3 with an *independent* checker (`openai/gpt-oss-120b`) verifying the solver's own
competing candidate solutions beats CoT maj@4 by a wide margin on the held-out split, and beats the
Universal Self-Consistency control:

| method (3 small solvers pooled, held-out pairs 60-197, 813 instances) | accuracy | fixes / breaks vs CoT maj@4 | p |
|---|---|---|---|
| CoT maj@4 (baseline) | 88.8% | - | - |
| Universal Self-Consistency | 89.7% | - | - |
| ESV v3, **self**-check | 85.1% | 18 / 57 | worse |
| **ESV v3, independent check (gpt-oss-120b)** | **94.1%** | **45 / 2** | **p < 1e-10** |
| CoT pass@4 (oracle ceiling) | 95.1% | - | - |

The whole result rests on *who* does the checking, not on the ESV stages: self-verification scores near
chance (right candidates called SOUND 57% vs wrong 37%, LR 1.6), while gpt-oss-120b separates them cleanly
(87% vs 2%, LR 47). The full diagnosis, every split and the per-arm breakdown are in
**`results/gsm8k/analysis/esv3_bottleneck_gsm8k.md`** — read that first; it is the authoritative write-up.

**MATH-500 is the second confirmation benchmark**, wired up and sampled but not yet run. It tests the GSM8K finding
where it should be hardest: competition-level problems, three answer formats in one benchmark, and 182 of its 500
problems answering with a LaTeX expression rather than a number. Everything downstream is identical, so the question
it answers is whether the independent-checker result survives a harder benchmark and a less forgiving answer space.
Decide on its stage-1 table (headroom and trap rate) before paying for the checker.

JEEBench was the dev benchmark and is **retired**: only 5-10% of the models' wrong answers were the "trap"
answer a misapplied strategy produces, and half the failures were on untouched problems (base accuracy
26-40%), which no checker can fix (`results/jeebench/analysis/esv_bottleneck_analysis.md`). Earlier
JEEBench-only experiments (ESV v4 with an oracle trigger, ESV v2 + controls, gold-free ESV v2) all failed
and their runners were removed; their reports are kept under `results/jeebench/analysis/` and their raw
records are still in `results/jeebench/<model>/records.json` (the `esv`, `retry` and `extra` blocks). Recover the
code from git history if it is ever needed.

## Benchmarks (`benchmarks.py`)

| name | source | problems | answer formats | status |
|---|---|---|---|---|
| `gsm8k` | `openai/gsm8k`, config main, test split (cached in `data/gsm8k/source.json`) | 1317 (2 negative answers dropped) -> 200 sampled -> 198 pairs | integer | **GO**, complete |
| `math500` | `HuggingFaceH4/MATH-500`, test split (cached in `data/math500/source.json`) | 500 -> 250 sampled -> pairs TBD | integer, numeric, **expression** | sampled, not yet run |
| `jeebench` | JEE Advanced 2016-2023 math (pilot's `dataset.json`) | 236 -> 200 sampled -> 193 pairs | single / multiple MCQ, integer, numeric | retired dev benchmark |

Per-benchmark settings live in `common.BENCHMARKS`: `n` (how many problems to sample, hence the `sample_<n>.json`
file name) and `solver_max_tokens` (the solver / checker reply cap, raised to 6000 for MATH-500 because its problems
are competition-level and a reply cut off by the cap costs a second, doubled-cap call on top of the wasted one).

### MATH-500 and the `expression` answer format

MATH-500 is the 500 curated MATH problems of Lightman et al. (2023), *Let's Verify Step by Step*. Unlike GSM8K its
answers are not all plain numbers, so **all 500 are kept** and the question type is read off the shape of the
published answer (`benchmarks.math500_type`):

| type | n | examples | scored by |
|---|---|---|---|
| `Integer` | 293 | `6`, `120` | exact match |
| `Numeric` | 25 | `-50`, `1.25`, `.35625` | within ±0.01 |
| `Expression` | 182 | `\frac{14}{3}`, `3\sqrt{13}`, `90^\circ`, `(6,31,-1)`, `(-\infty,2)\cup(3,\infty)`, `52_8`, `\begin{pmatrix}-1/3 \\ 2/3 \\ 5/3\end{pmatrix}`, `\text{Evelyn}` | `expression_answers.py` |

Keeping all three is what makes the 250 a sample *of MATH-500* rather than of its integer subset, and it preserves
the subject mix — Precalculus and Intermediate Algebra are mostly `Expression`-valued. The drawn sample mirrors the
population on all three axes: 146 integer / 11 numeric / 93 expression, levels 1-5 at 21/44/59/62/64, and the same
subject proportions.

`expression_answers.py` is the only place that knows how to compare LaTeX answers. `match(a, b)` escalates from
cheapest and safest to most permissive, and bails out rather than guess:

1. canonical strings equal, then case-insensitively (`\text{East}` vs `east`);
2. element by element for lists, tuples, intervals, sets and column vectors — **and the delimiters must agree**, so
   `(3,4]` never equals `[3,4]` and `(6,31,-1)` never equals `(6,-1,31)`;
3. exact numeric equality when both sides are plain numbers;
4. symbolic equality via sympy, on a conservative subset only: the LaTeX is converted to sympy syntax and the attempt
   is **abandoned** unless the conversion consumed every backslash and left no `_`, `,`, `|`, `=` or bracket. So
   `\frac{1}{2}` matches `0.5` and `(a+5)(b+2)` matches `ab+2a+5b+10`, while `52_8` (base 8) is never compared
   numerically with `52` and `1 \pm \sqrt{19}` is only ever compared as a string.

Normalization strips `\boxed{}`, `$`, `\left\right`, spacing commands, `^\circ` and thousands separators, rewrites
`\dfrac`/`\tfrac`/`\frac12` to one spelling, flattens matrices to tuples, and drops a trailing `=` only when the left
side is a bare variable (`x=5` -> `5`, but `5x-7y+11z+4=0` is kept whole). A `\text{...}` that **is** the whole
answer keeps its word (`\text{ellipse}`); one attached to a value is a unit and is dropped with its exponent
(`864 \mbox{ inches}^2` -> `864`), since the solver is told to answer without units.

What is deliberately **not** accepted: a decimal approximation of an exact value (`4.67` is not `\frac{14}{3}`),
because "give the exact value" is part of what these problems ask. Equivalence checking on free text is never
perfect; it is applied identically to the gold and to every method's answers, so no arm is favoured, and
`analyze_esv3.py` prints a **By answer format** table so the `expression` slice can be compared against the exactly
scored ones. `tests/test_expression_answers.py` asserts that all 500 golds normalize, normalize idempotently, match
themselves and pass `build_dataset.valid_answer`.

## Models

- **Solvers, small** (`common.SMALL_SOLVER_MODELS`): Qwen 2.5 7B, Mistral Nemo 12B, Gemma 3 4B — the GSM8K
  result above.
- **Solvers, large** (`common.SOLVER_MODELS`, the default): Qwen 2.5 72B, Mistral Small 3.2 24B, Llama 3.3 70B.
  On GSM8K these are at 97% CoT maj@4 with only 7.5% of instances disagreeing, so there is little headroom.
- **Checker:** `openai/gpt-oss-120b`, passed to `run_esv3.py --verifier-model`. Without that flag the solver
  checks itself, which is the control that fails.

## The method: ESV v3

Starts from the 4 CoT samples already in `records.json`; it never re-solves.

1. **Disagreement gate** (no call): group the 4 CoT answers into distinct candidates. All agree -> keep the
   CoT majority answer. Only disagreeing instances can change, so the checker runs on ~25% of instances.
2. **Candidate verification** (one call, T = 0): for every distinct candidate solution, in order of first
   appearance and *without* vote counts — the strategy, its bindings to the problem text (MATCHES /
   MISMATCH), the method's conditions (HOLDS / VIOLATED), the divergence point, then SOUND or FLAWED with a
   failure type (BINDING / METHOD / COMPUTATION), the first flawed step and the corrected fact.
3. **Choice** (code, no call): a candidate counts as SOUND only if its verdict is SOUND *and* it lists no
   MISMATCH and no VIOLATED. Exactly one SOUND -> choose it; several -> most votes; none -> keep the initial
   answer (stage 4, diagnosis-driven repair, is not built).

The gold answer is used only to score, never in a decision — so the comparison against CoT maj@4 is fair:
every method starts from the same answer and can fix a wrong one or break a right one. **Control:**
Universal Self-Consistency (Chen et al., 2023) on the same samples and the same gate.

## Files

| File | Role |
|---|---|
| `common.py` | benchmarks and paths, config (models, seeds, n = 4, temperature), OpenRouter client, tagged-output calls, type-aware scoring, McNemar |
| `benchmarks.py` | source problems per benchmark in one shape; pulls and caches GSM8K and MATH-500 |
| `expression_answers.py` | normalization and equivalence for the `expression` answer format (MATH-500's LaTeX answers) |
| `perturb_prompts.py` | generator / verifier prompts for both perturbation modes, format-preserving |
| `build_dataset.py` | sample the benchmark's `n` problems, two-stage perturbation with the mode balancer -> `data/<benchmark>/frozen_dataset.json` |
| `method_prompts.py` | the Direct QA, CoT, ESV v3 and USC prompts |
| `run_methods.py` | the 4 Direct QA + 4 CoT samples every method starts from |
| `run_esv3.py` | ESV v3 and the USC control; `--verifier-model` for the independent checker |
| `analyze_esv3.py` | the stage-by-stage report, the go / no-go line, every comparison table and the **Compute cost** section (calls, solver vs verifier tokens, serial and critical-path latency, tokens per net fix) |
| `tests/smoke_test_offline.py` | every step on all three benchmarks against a fake model client, no API calls |
| `tests/test_expression_answers.py` | the `expression` normalizer and matcher, incl. all 500 MATH-500 golds |

## Run order (from the repo root)

The OpenRouter key is read from `experiments/.env` (or `RAEN_ENV_FILE`). Every step resumes where it stopped.

```bash
python experiments/tests/smoke_test_offline.py              # 0. offline check of every step, no API calls
python experiments/tests/test_expression_answers.py         #    and the expression scorer (MATH-500)

B=math500                                                   # or gsm8k
python experiments/benchmarks.py $B                          # 1. pull / summarize the source
python experiments/build_dataset.py --benchmark $B --limit 10 # 2. small live check of the perturbation pipeline
python experiments/build_dataset.py --benchmark $B           #    then every sampled problem (200, or 250 for math500)
python experiments/run_methods.py --benchmark $B --models <small solvers>   # 3. the 4 DQA + 4 CoT samples
python experiments/analyze_esv3.py --benchmark $B --models small            #    stage 1 only: headroom + trap rate
python experiments/run_esv3.py --benchmark $B --verifier-model openai/gpt-oss-120b --models <small solvers>
python experiments/analyze_esv3.py --benchmark $B --models small --verifier-model openai/gpt-oss-120b
```

Decide on stage 1 before spending on the checker: if the solvers' 4 CoT samples almost never disagree, or
almost no disagreement is fixable (`disagreeing, fixable` in the stage-1 table), there is no headroom.

**Split discipline.** Pairs 0-59 are the dev split and pairs >= 60 are held out. `analyze_esv3.py` takes
`--max-pair 60` for the dev split and `--min-pair 60` for the held-out one; a report's file name records
which (`.upto60`, `.from60`). Tune on dev only.

Every report rebuilds offline from the stored records, with no API calls.

## Outputs

- `data/<benchmark>/`: `sample_<n>.json` (`sample_200.json`, or `sample_250.json` for MATH-500),
  `generation_log.jsonl` (every attempt), `frozen_dataset.json`; GSM8K and MATH-500 also `source.json`
- `results/<benchmark>/<model>/`: `records.json` (the 4 + 4 samples), `esv3_records.json` (self-check) and
  `esv3_records.<verifier>.json` (independent check)
- `results/<benchmark>/analysis/`: the reports

The retired JEEBench experiments left extra keys on their records: `esv` / `retry` (Experiment 1, ESV v4) and
`extra` (Experiment 2). No current script reads or writes them, and `run_methods.py --no-resume` would discard
them, so treat git history as their archive.
