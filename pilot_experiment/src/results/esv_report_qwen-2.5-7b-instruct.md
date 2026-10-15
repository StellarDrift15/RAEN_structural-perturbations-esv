# ESV evaluation report: `qwen/qwen-2.5-7b-instruct`

- Dataset: `C:\Ekansh_Folders\Ekansh_Folders\2026\RAEN_structural-perturbations-esv\pilot_experiment\src\frozen_dataset.json` (32/32 instances scored)
- Judge model (grades diagnosis and alternative strategy only): `openai/gpt-oss-120b`
- Prompt version: `esv_v4_2026-10-14`, last updated 2026-10-15T04:33:04.263981+00:00
- Pipeline: the model solves each problem; every genuine failure then goes through all five ESV stages: recognition (strategy check + independent re-solve), diagnosis, alternative strategy (3 verified candidates), recovery, and final answer (agreement of two independent derivations, adjudicated if they disagree).

## Headline

- Initial accuracy: **40.6% (13/32)**, so **19** failures went through ESV.
- Recognition **79.0% (15/19)**, diagnosis **15.8% (3/19)**, valid alternative strategy **36.8% (7/19)**, recovery **26.3% (5/19)**.
- Final accuracy: **40.6% (13/32) -> 56.2% (18/32)** after ESV.

## 1. Initial solving performance

Answer Accuracy = #Correct / #Total. Failure Rate = #Incorrect / #Total.

| Metric | Overall | Original | Perturbed |
|---|---|---|---|
| Answer accuracy | 40.6% (13/32) | 37.5% (6/16) | 43.8% (7/16) |
| Failure rate | 59.4% (19/32) | 62.5% (10/16) | 56.2% (9/16) |
| No readable answer (counted as failure) | 1 | 0 | 1 |

## 2. Failure recognition

Recognition Accuracy = #Correctly recognized failures / #Actual failures. The model is not told its answer is wrong. A failure is recognized if the strategy check flags it or an independent re-solve contradicts the initial answer.

| Metric | Overall | Original | Perturbed |
|---|---|---|---|
| Recognition accuracy | 79.0% (15/19) | 80.0% (8/10) | 77.8% (7/9) |
| ... flagged by the strategy check | 68.4% (13/19) | 70.0% (7/10) | 66.7% (6/9) |
| ... flagged by re-solve disagreement | 73.7% (14/19) | 80.0% (8/10) | 66.7% (6/9) |
| Independent re-solve already correct | 15.8% (3/19) | 20.0% (2/10) | 11.1% (1/9) |

## 3. Failure diagnosis

Diagnosis Accuracy = #Correctly diagnosed failures / #Actual failures (judge-graded).

| Metric | Overall | Original | Perturbed |
|---|---|---|---|
| Diagnosis accuracy | 15.8% (3/19) | 20.0% (2/10) | 11.1% (1/9) |
| Partial diagnoses | 4 | 2 | 2 |
| Diagnosis accuracy, given recognized | 20.0% (3/15) | 25.0% (2/8) | 14.3% (1/7) |

## 4. Alternative strategy

Strategy Accuracy = #Failures with a valid alternative strategy / #Actual failures (judge-graded). Self-verified = the model's own verification passed.

| Metric | Overall | Original | Perturbed |
|---|---|---|---|
| Strategy accuracy | 36.8% (7/19) | 50.0% (5/10) | 22.2% (2/9) |
| Strategy accuracy, given correct diagnosis | 66.7% (2/3) | 100.0% (2/2) | 0.0% (0/1) |
| Self-verified by the model | 100.0% (19/19) | 100.0% (10/10) | 100.0% (9/9) |
| Needed a second strategy | 2 | 1 | 1 |

## 5. Failure recovery

Recovery Rate = #Recovered failures / #Actual failures. Conditional Recovery Rate = #Recovered / #Correctly recognized failures.

| Metric | Overall | Original | Perturbed |
|---|---|---|---|
| Recovery rate | 26.3% (5/19) | 30.0% (3/10) | 22.2% (2/9) |
| Conditional recovery rate | 26.7% (4/15) | 37.5% (3/8) | 14.3% (1/7) |
| Recovery rate, given valid strategy | 14.3% (1/7) | 20.0% (1/5) | 0.0% (0/2) |
| Correct after recovery stage | 26.3% (5/19) | 20.0% (2/10) | 33.3% (3/9) |
| Cross-check (backup strategy) correct | 31.6% (6/19) | 30.0% (3/10) | 33.3% (3/9) |
| Recovery and cross-check agree | 36.8% (7/19) | 40.0% (4/10) | 33.3% (3/9) |
| ... accuracy when they agree | 42.9% (3/7) | 25.0% (1/4) | 66.7% (2/3) |
| ... accuracy when adjudicated | 16.7% (2/12) | 33.3% (2/6) | 0.0% (0/6) |
| Correct after final answer stage | 26.3% (5/19) | 30.0% (3/10) | 22.2% (2/9) |
| Final stage fixed / broke | 1 / 1 | 1 / 0 | 0 / 1 |
| Repeated the known-wrong answer | 21.1% (4/19) | 20.0% (2/10) | 22.2% (2/9) |

## 6. Stage funnel

How many failures make it through each stage in sequence.

| Metric | Overall | Original | Perturbed |
|---|---|---|---|
| Failures | 19 | 10 | 9 |
| Recognized | 15 | 8 | 7 |
| ... and correctly diagnosed | 3 | 2 | 1 |
| ... and valid strategy | 2 | 2 | 0 |
| ... and recovered | 0 | 0 | 0 |

## 7. Perturbation robustness

Perturbation Accuracy Drop = Accuracy(original) - Accuracy(perturbed), over 16 complete pairs.

| Stage | Acc. original | Acc. perturbed | Drop | Pairs: both / orig only / pert only / neither |
|---|---|---|---|---|
| Initial | 37.5% (6/16) | 43.8% (7/16) | -6.2 pp | 3 / 3 / 4 / 6 |
| After ESV | 56.2% (9/16) | 56.2% (9/16) | +0.0 pp | 5 / 4 / 4 / 3 |

## Failures through ESV

`recog`: recognized in stage 1. `diag`/`strat`: judge verdicts. Y = correct.

| Instance | Gold | Initial | recog | diag | strat | Recovery answer | Final answer | Recovered |
|---|---|---|---|---|---|---|---|---|
| 0_original | C | D | Y | PARTIAL | INVALID | A | A | N |
| 2_original | C | B | N | INCORRECT | VALID | B | B | N |
| 2_perturbed | A | C | Y | INCORRECT | INVALID | C | C | N |
| 3_perturbed | C | A | Y | INCORRECT | VALID | C | B | N |
| 5_perturbed | D | ABCD | Y | PARTIAL | INVALID | C | C | N |
| 6_original | ACD | AD | Y | INCORRECT | VALID | ACD | ACD | Y |
| 6_perturbed | CD | AB | Y | INCORRECT | INVALID | CD | CD | Y |
| 7_original | BC | AD | Y | INCORRECT | INVALID | BD | B | N |
| 7_perturbed | BC | E | Y | CORRECT | INVALID | BD | BD | N |
| 8_perturbed | AC | C | Y | PARTIAL | INVALID | AD | D | N |
| 10_original | 2 | 4 | Y | CORRECT | VALID | 1 | 1 | N |
| 10_perturbed | B | A | N | INCORRECT | INVALID | B | B | Y |
| 11_original | 5 | 16 | N | PARTIAL | INVALID | \boxed{1} | 1 | N |
| 11_perturbed | A | B | N | INCORRECT | INVALID | B | B | N |
| 12_original | 1 | 0 | Y | INCORRECT | INVALID | 1 | 1 | Y |
| 13_original | 7 | 18 | Y | CORRECT | VALID | 8 | 8 | N |
| 14_original | 1 | 2 | Y | INCORRECT | VALID | 2 | 2 | N |
| 15_original | B | D | Y | INCORRECT | INVALID | A | B | Y |
| 15_perturbed | B | A | Y | INCORRECT | VALID | D | C | N |

Solved correctly at the start (no ESV needed): 0_perturbed, 1_original, 1_perturbed, 3_original, 4_original, 4_perturbed, 5_original, 8_original, 9_original, 9_perturbed, 12_perturbed, 13_perturbed, 14_perturbed

## Execution

- Stage calls: 177 (180 API requests including retries)
- Stage status counts: {'completed': 177}
- Tokens: {'prompt_tokens': 233770, 'completion_tokens': 128529, 'total_tokens': 362299}
- Total model latency: 1339.2 s

## Notes on validity

- The solver never sees the gold answer or the frozen annotations; they are used only for local scoring and by the judge.
- Recognition is blind. From diagnosis onward the model is told that its initial answer is incorrect (but never what the correct answer is), because ESV is applied only to failures. State this when reporting recovery rates.
- Diagnosis and strategy accuracy rely on an LLM judge; spot-check `judge_diagnosis` entries in the results JSON.
