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
| `jeebench` | JEE Advanced 2016-2023 math (pilot's `dataset.json`) | 236 -> 200 sampled -> 193 pairs | single / multiple MCQ, integer, numeric | retired dev benchmark |

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
| `benchmarks.py` | source problems per benchmark in one shape; pulls and caches GSM8K |
| `perturb_prompts.py` | generator / verifier prompts for both perturbation modes, format-preserving |
| `build_dataset.py` | sample 200, two-stage perturbation with the mode balancer -> `data/<benchmark>/frozen_dataset.json` |
| `method_prompts.py` | the Direct QA, CoT, ESV v3 and USC prompts |
| `run_methods.py` | the 4 Direct QA + 4 CoT samples every method starts from |
| `run_esv3.py` | ESV v3 and the USC control; `--verifier-model` for the independent checker |
| `analyze_esv3.py` | the stage-by-stage report, the go / no-go line and every comparison table |
| `tests/smoke_test_offline.py` | every step on both benchmarks against a fake model client, no API calls |

## Run order (from the repo root)

The OpenRouter key is read from `experiments/.env` (or `RAEN_ENV_FILE`). Every step resumes where it stopped.

```bash
python experiments/tests/smoke_test_offline.py              # 0. offline check of every step, no API calls

B=gsm8k                                                     # or the next benchmark
python experiments/benchmarks.py $B                          # 1. pull / summarize the source
python experiments/build_dataset.py --benchmark $B --limit 10 # 2. small live check of the perturbation pipeline
python experiments/build_dataset.py --benchmark $B           #    then all 200 problems
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

- `data/<benchmark>/`: `sample_200.json`, `generation_log.jsonl` (every attempt), `frozen_dataset.json`;
  GSM8K also `source.json`
- `results/<benchmark>/<model>/`: `records.json` (the 4 + 4 samples), `esv3_records.json` (self-check) and
  `esv3_records.<verifier>.json` (independent check)
- `results/<benchmark>/analysis/`: the reports

The retired JEEBench experiments left extra keys on their records: `esv` / `retry` (Experiment 1, ESV v4) and
`extra` (Experiment 2). No current script reads or writes them, and `run_methods.py --no-resume` would discard
them, so treat git history as their archive.
