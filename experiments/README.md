# ESV experiments: Direct QA vs CoT vs ESV on structurally perturbed problems

One code base, several benchmarks. Every script takes `--benchmark` (default `jeebench`) and reads / writes
`data/<benchmark>/` and `results/<benchmark>/`.

> **Folder name.** This folder was built as `jeebench_experiment/` and now holds every benchmark. Rename it to
> `experiments/` (close VS Code first; the folder is locked while it is open) so the commands below work as written.

## Benchmarks (`benchmarks.py`)

| name | source | problems | answer formats | status |
|---|---|---|---|---|
| `jeebench` | JEE Advanced 2016-2023 math (pilot's `dataset.json`) | 236 -> 200 sampled -> 193 pairs | single / multiple MCQ, integer, numeric | dev benchmark, Experiments 1-3 and ESV v3 done |
| `gsm8k` | `openai/gsm8k`, config main, test split (cached in `data/gsm8k/source.json`) | 1317 (2 negative answers dropped) -> 200 sampled | integer | source pulled; pairs not built yet |

Why GSM8K next: on JEEBench only 5-10% of the models' wrong answers are the "trap" answer that a misapplied strategy
produces; half the failures are on untouched problems (base accuracy 26-40%), which no strategy check can fix. On
GSM8K these models solve ~90% of originals, so failures on perturbed problems should mostly be strategy failures,
which is the case ESV is designed for (`results/jeebench/analysis/esv_bottleneck_analysis.md`).

## Solver models

- **Current (default):** Qwen 2.5 72B, Mistral Small 3.2 24B and Llama 3.3 70B (`common.SOLVER_MODELS`).
- **Small:** Qwen 2.5 7B, Mistral Nemo 12B and Gemma 3 4B (`common.SMALL_SOLVER_MODELS`), which ran Experiments 1-3
  on JEEBench.

## The experiments

All use the same frozen pairs and the same 4 Direct QA + 4 CoT samples per instance (`run_methods.py`).

| | Where ESV runs | Gold answer used for | Code | Report (in `results/<benchmark>/analysis/`) |
|---|---|---|---|---|
| **1. ESV v4** (frozen) | only where all 4 CoT samples are wrong | the trigger and scoring | `run_methods.py` phases 1-2 | `analyze.py` -> `report.md` |
| **2. ESV v2 + controls** | the same consistent failures | the trigger and scoring | `run_methods.py --extra-methods` | `compare_methods.py` -> `method_comparison.md` |
| **3. Gold-free ESV v2** | every instance; blind recognition | scoring only | `run_open.py` | `analyze_open.py` -> `open_report_{large,small}.md` |
| **ESV v3** (stages 1-3) | every instance where the 4 CoT samples disagree | scoring only | `run_esv3.py` | `analyze_esv3.py` -> `esv3_report.md` |

Experiments 1-2 use an oracle trigger, so ESV can only gain there. Experiment 3 and ESV v3 are the like-for-like
comparisons: every method starts from the CoT majority answer (CoT maj@4), decides on its own whether to change it,
and can fix a wrong answer or break a right one.

- **Experiment 3 methods:** ESV v2 (blind recognition -> diagnosis -> 3 strategies -> one solve each -> vote with the
  initial answer); Recheck x3 (same recognition gate, 3 plain re-solves, same vote); Self-Refine (one review-and-correct
  call).
- **ESV v3:** no re-solving. Stage 1 keeps the answer when the 4 samples agree; one call verifies every distinct
  candidate solution against the problem text (strategy, bindings MATCHES / MISMATCH, conditions, SOUND / FLAWED,
  failure type, corrected fact); a candidate that is the only SOUND one is chosen. Control: Universal
  Self-Consistency on the same samples. Stage 4 (diagnosis-driven repair) is not built.

### JEEBench results so far (large models, 30 dev pairs = 180 instances)

| Method | accuracy | fixes / breaks vs CoT maj@4 |
|---|---|---|
| CoT maj@4 | 40.0% | - |
| Recheck x3 | 40.0% | 3 / 3 |
| Self-Refine | 38.9% | 7 / 9 |
| Universal Self-Consistency | 38.3% | 7 / 10 |
| ESV v2 (gold-free) | 38.3% | 1 / 4 |
| ESV v3 (stages 1-3) | 32.8% | 9 / 22 (p = 0.029) |

Every self-check the same model performs (recognition, Self-Refine's verdict, the answer checks, the v3 candidate
verdicts) scores near chance: the checker re-encodes the solver's confidence instead of adding information. See
`esv_bottleneck_analysis.md` for the full diagnosis.

## Files

| File | Role |
|---|---|
| `common.py` | benchmarks and paths, config (models, seeds, n = 4, temperature), OpenRouter client, tagged-output calls, type-aware scoring |
| `benchmarks.py` | source problems per benchmark in one shape; pulls and caches GSM8K |
| `perturb_prompts.py` | generator / verifier prompts for both perturbation modes, format-preserving |
| `build_dataset.py` | sample 200, two-stage perturbation with the mode balancer -> `data/<benchmark>/frozen_dataset.json` |
| `method_prompts.py` | every solver and judge prompt: DQA, CoT, ESV v4, ESV v2, controls, ESV v3, USC |
| `run_methods.py` | 4 DQA + 4 CoT samples (`--samples-only`), then Experiments 1-2 |
| `run_open.py` | Experiment 3 |
| `run_esv3.py` | ESV v3 and Universal Self-Consistency |
| `analyze.py`, `compare_methods.py`, `analyze_open.py`, `analyze_esv3.py` | the reports above |
| `failure_analysis.py` | where ESV v4 loses (JEEBench, frozen Experiment 1 records) |
| `tests/smoke_test_offline.py` | every step on both benchmarks against a fake model client, no API calls |

## Run order (from the repo root)

The OpenRouter key is read from `experiments/.env` (or `RAEN_ENV_FILE`). Every step resumes where it stopped.

```bash
python experiments/tests/smoke_test_offline.py                      # 0. offline check, no API calls

# GSM8K (next)
python experiments/benchmarks.py gsm8k                              # pull / summarize the source (done, cached)
python experiments/build_dataset.py --benchmark gsm8k --limit 10    # 1. small live check of the perturbation pipeline
python experiments/build_dataset.py --benchmark gsm8k               #    all 200 problems
python experiments/run_methods.py --benchmark gsm8k --samples-only --limit-pairs 30   # 2. 4 DQA + 4 CoT samples
python experiments/analyze_esv3.py --benchmark gsm8k                #    stage 1: CoT maj@4 vs pass@4, trap rate
python experiments/run_esv3.py --benchmark gsm8k --limit-pairs 30   # 3. ESV v3 + USC
python experiments/run_open.py --benchmark gsm8k --limit-pairs 30   #    ESV v2 + controls (same instances)

# JEEBench (all done; reports rebuild offline)
python experiments/analyze.py; python experiments/compare_methods.py
python experiments/analyze_open.py; python experiments/analyze_esv3.py
```

## Outputs

- `data/<benchmark>/`: `sample_200.json`, `generation_log.jsonl` (every attempt), `frozen_dataset.json`; GSM8K also `source.json`
- `results/<benchmark>/<model>/`: `records.json` (samples + Experiments 1-2), `open_records.json` (Experiment 3), `esv3_records.json`
- `results/<benchmark>/analysis/`: the reports and figures
- `results_frozen/jeebench/experiment1_esv_v4/`: the final Experiment 1 records (checksums in `SHA256SUMS`); never modified
