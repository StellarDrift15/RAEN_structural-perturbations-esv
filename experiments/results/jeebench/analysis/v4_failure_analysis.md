# ESV v4 failure analysis (JEEBench dev set, Experiment 1)

625 ESV runs on consistent failures (all 4 CoT samples wrong). Source: the `esv` blocks of `results/jeebench/<model>/records.json` (was `results_frozen/jeebench/experiment1_esv_v4`, since removed as a duplicate).

## 1. What happened on each failure

Mutually exclusive, assigned in this order. *Lost* = a correct answer was produced at some stage but not kept.

| Outcome | Qwen 2.5 7B | Mistral Nemo 12B | Gemma 3 4B | Pooled |
|---|---|---|---|---|
| recovered | 25/176 (14.2%) | 23/258 (8.9%) | 15/191 (7.9%) | 63/625 (10.1%) |
| lost: independent re-solve only | 10/176 (5.7%) | 8/258 (3.1%) | 8/191 (4.2%) | 26/625 (4.2%) |
| lost: adjudication chose wrong | 7/176 (4.0%) | 10/258 (3.9%) | 8/191 (4.2%) | 25/625 (4.0%) |
| stubborn: repeated known-wrong | 25/176 (14.2%) | 37/258 (14.3%) | 68/191 (35.6%) | 130/625 (20.8%) |
| recycled: another CoT wrong answer | 38/176 (21.6%) | 39/258 (15.1%) | 20/191 (10.5%) | 97/625 (15.5%) |
| near miss | 14/176 (8.0%) | 18/258 (7.0%) | 7/191 (3.7%) | 39/625 (6.2%) |
| other wrong | 57/176 (32.4%) | 123/258 (47.7%) | 65/191 (34.0%) | 245/625 (39.2%) |

By answer format (pooled counts):

| Format | n | recovered | lost: independent re-solve only | lost: adjudication chose wrong | stubborn: repeated known-wrong | recycled: another CoT wrong answer | near miss | other wrong |
|---|---|---|---|---|---|---|---|---|
| single | 80 | 13 | 10 | 8 | 23 | 21 | 0 | 5 |
| multiple | 266 | 29 | 10 | 10 | 59 | 42 | 34 | 82 |
| integer | 88 | 6 | 2 | 3 | 19 | 17 | 2 | 39 |
| numeric | 191 | 15 | 4 | 4 | 29 | 17 | 3 | 119 |

## 2. Free test: would a different final rule have done better?

Each rule picks a final answer from outputs ESV v4 already produced, without the gold answer. *Excl. known-wrong* never returns the answer the model was told is wrong.

| Final-answer rule | Qwen 2.5 7B | Mistral Nemo 12B | Gemma 3 4B | Pooled |
|---|---|---|---|---|
| v4 as run (agree, else adjudicate) | 25/176 (14.2%) | 23/258 (8.9%) | 15/191 (7.9%) | 63/625 (10.1%) |
| recovery only (original lifecycle) | 17/176 (9.7%) | 13/258 (5.0%) | 12/191 (6.3%) | 42/625 (6.7%) |
| vote of 3 derivations | 22/176 (12.5%) | 16/258 (6.2%) | 13/191 (6.8%) | 51/625 (8.2%) |
| vote of 3 derivations, excl. known-wrong | 24/176 (13.6%) | 19/258 (7.4%) | 17/191 (8.9%) | 60/625 (9.6%) |
| vote of 3 derivations + adjudicator, excl. known-wrong | 22/176 (12.5%) | 24/258 (9.3%) | 18/191 (9.4%) | 64/625 (10.2%) |
| v4, but option-vote on multiple-MCQ | 22/176 (12.5%) | 22/258 (8.5%) | 12/191 (6.3%) | 56/625 (9.0%) |
| retry control | 25/176 (14.2%) | 20/258 (7.8%) | 10/191 (5.2%) | 55/625 (8.8%) |
| ceiling: any derivation correct | 42/176 (23.9%) | 41/258 (15.9%) | 31/191 (16.2%) | 114/625 (18.2%) |

## 3. Multiple-correct MCQs (the largest group)

266 runs. Options wrong = size of the symmetric difference with the gold set.

| Answer | per-option accuracy | 0 wrong | 1 wrong | 2 wrong | 3 wrong | 4 wrong |
|---|---|---|---|---|---|---|
| ESV final | 54.0% | 29 | 75 | 87 | 60 | 15 |
| retry | 54.7% | 25 | 83 | 88 | 57 | 13 |
| initial (known-wrong) | 49.2% | 0 | 89 | 97 | 62 | 18 |

Gold set size: {1: 27, 2: 142, 3: 88, 4: 9}; ESV final set size: {0: 1, 1: 132, 2: 89, 3: 36, 4: 8}.

## 4. Stage behaviour

| Model | indep. = known-wrong | recovery = known-wrong | cross-check = known-wrong | recognition said 'no failure' | strategy failed own check | all 3 derivations agree |
|---|---|---|---|---|---|---|
| Qwen 2.5 7B | 52/176 (29.5%) | 22/176 (12.5%) | 32/176 (18.2%) | 45/176 (25.6%) | 2/176 (1.1%) | 32/176 (18.2%) |
| Mistral Nemo 12B | 41/258 (15.9%) | 47/258 (18.2%) | 45/258 (17.4%) | 22/258 (8.5%) | 2/258 (0.8%) | 9/258 (3.5%) |
| Gemma 3 4B | 62/191 (32.5%) | 81/191 (42.4%) | 77/191 (40.3%) | 24/191 (12.6%) | 0/191 (0.0%) | 38/191 (19.9%) |
| Pooled | 155/625 (24.8%) | 150/625 (24.0%) | 154/625 (24.6%) | 91/625 (14.6%) | 4/625 (0.6%) | 79/625 (12.6%) |

## 5. Does the quality of diagnosis / strategy matter? (judge-graded, pooled)

| Diagnosis | Strategy | n | ESV final | recovery stage | retry |
|---|---|---|---|---|---|
| CORRECT | VALID | 88 | 8/88 (9.1%) | 5/88 (5.7%) | 9/88 (10.2%) |
| CORRECT | INVALID | 175 | 19/175 (10.9%) | 12/175 (6.9%) | 15/175 (8.6%) |
| PARTIAL | VALID | 36 | 6/36 (16.7%) | 4/36 (11.1%) | 3/36 (8.3%) |
| PARTIAL | INVALID | 111 | 9/111 (8.1%) | 8/111 (7.2%) | 6/111 (5.4%) |
| INCORRECT | VALID | 50 | 12/50 (24.0%) | 6/50 (12.0%) | 3/50 (6.0%) |
| INCORRECT | INVALID | 165 | 9/165 (5.5%) | 7/165 (4.2%) | 19/165 (11.5%) |

Keywords in the judge's explanation for the 451 INVALID strategies (not exclusive): mathematically wrong / leads to wrong answer 95, misses the changed/critical condition 61, same as failed method 34.

## 6. Output tokens per ESV run, by stage (pooled mean)

| Stage | mean output tokens |
|---|---|
| esv_independent_solve | 1079 |
| esv_recognition | 294 |
| esv_diagnosis | 197 |
| esv_alternative_strategy | 532 |
| esv_recovery | 1301 |
| esv_cross_check | 1340 |
| esv_final_answer | 121 |
| judge_diagnosis | 988 |
| retry (for comparison) | 1349 |
