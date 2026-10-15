# Baseline report: `qwen/qwen-2.5-7b-instruct`

- Baseline version: `baselines_v1|esv_v4_2026-10-14` (prompts, call settings and scorer imported from methods/esv.py)
- ESV results: `C:\Ekansh_Folders\Ekansh_Folders\2026\RAEN_structural-perturbations-esv\pilot_experiment\src\results\esv_results_qwen-2.5-7b-instruct.json`
- Generated 2026-10-15T14:33:02.319184+00:00

## Accuracy

| Method | Overall | Original | Perturbed | Drop (orig - pert) | Pairs both / orig only / pert only / neither |
|---|---|---|---|---|---|
| Direct QA | 28.1% (9/32) | 31.2% (5/16) | 25.0% (4/16) | +6.2 pp | 1 / 4 / 3 / 8 |
| CoT | 37.5% (12/32) | 43.8% (7/16) | 31.2% (5/16) | +12.5 pp | 3 / 4 / 2 / 7 |
| ESV Stage 0 (its initial CoT solve) | 40.6% (13/32) | 37.5% (6/16) | 43.8% (7/16) | -6.2 pp | 3 / 3 / 4 / 6 |
| ESV (final, after recovery) | 56.2% (18/32) | 56.2% (9/16) | 56.2% (9/16) | +0.0 pp | 5 / 4 / 4 / 3 |

## Answer extraction and truncation

| Method | No readable answer | Call status counts | Hit token cap on final attempt |
|---|---|---|---|
| Direct QA | 0/32 | {'completed': 32} | 0 |
| CoT | 0/32 | {'completed': 32} | 0 |

## Paired comparison: ESV vs CoT (same instances)

- Instances in both: 32
- Both correct: 9 | ESV only: 9 | CoT only: 3 | neither: 11
- Exact McNemar test on the discordant instances: p = 0.146
- Sanity check: the CoT baseline and ESV Stage 0 use the identical prompt at temperature 0, and agree on correctness for 21/32 instances. Some disagreement is normal (providers are not fully deterministic at temperature 0); a large share would mean the runs were not set up the same way.

## Notes on validity

- All methods share the solver model, temperature, token cap and retries, question text, answer-format note, answer parsing and scorer (imported from methods/esv.py). Only the instruction differs.
- ESV runs its stages only when the initial answer is wrong according to the gold answer, so its final accuracy can never fall below its Stage 0 accuracy. Report this alongside any ESV vs CoT comparison.
- Pairs [10, 11, 12, 13, 14] have integer-answer originals but 4-option multiple-choice perturbed versions, so original vs perturbed accuracy is not like-for-like on those pairs.
