# Why ESV's stages do not turn into recoveries (Experiment 3, offline analysis)

Data: every gold-free record (`open_records.json`). Small models are Qwen 2.5 7B, Mistral Nemo and Gemma 3 4B (965
instances). Large models are Qwen 2.5 72B, Mistral Small 3.2 24B and Llama 3.3 70B (180 instances, 30 pilot pairs). No
API calls were made. "Flagged" means ESV recognition said the initial answer fails. "k" is how many of the 4 CoT
samples were correct.

## Summary

The pipeline generates new candidate answers but has no reliable way to choose among them. Every stage runs on the same
model with the same blind spots, so errors are correlated rather than corrected. Scoring is not the cause.

| Stage | What it should add | What it actually adds | Evidence |
|---|---|---|---|
| Recognition | find wrong answers | close to chance; worse than the free CoT-agreement signal | precision 77% vs 73.5% base rate (small), 67% vs 60% (large) |
| Diagnosis | locate the error | text that names *some* real issue but does not change what happens next | judge CORRECT: per-solve 9.5%; INCORRECT: 12.0% |
| Strategies | a better method | no effect on solve accuracy | solve correct after a VALID strategy 10.6%, after an INVALID one 10.3% |
| Solves | correct derivations | the same accuracy as plain re-solves; worse on correct answers | flagged-wrong: ESV 10.4% vs Recheck 11.1%; flagged-right: 44.6% vs 52.1% vs CoT 67% |
| Vote | pick the right candidate | discards most correct candidates | right answer among ESV solves: +91 possible fixes; actually fixed: 20 (and 23 breaks) |

## 1. Selection is the biggest gap

| | Small | Large |
|---|---|---|
| Flagged-wrong instances where some ESV solve is correct | 91 | 10 |
| Fixed by the current vote | 20 | 1 |
| CoT maj@4 vs CoT pass@4 (answer already among the 4 samples) | 26.5% vs 44.6% (+18.0 pp) | 40.0% vs 60.0% (+20.0 pp) |

- A correct candidate is usually a minority: the 3 solves give 2–3 distinct answers in about 80% of runs.
- No vote rule fixes this. Net change for each rule (ESV, then Recheck ×3; small models):

  | Rule | ESV | Recheck ×3 |
  |---|---|---|
  | initial answer votes (current) | −3 | +1 |
  | the 3 derivations only | −19 | −11 |
  | 4 CoT samples + 3 derivations | +1 | +8 |

  The best rule is just more self-consistency, and it helps Recheck ×3 more than ESV.
- The largest headroom needs no new solves: the correct answer is already among the 4 CoT samples in 18–20 pp of
  instances. ESV never looks at those alternatives. It only examines the majority answer.

## 2. Solves are limited by capability, and ESV's framing hurts correct answers

- **Many failures are out of reach.** For 75.5% (small) and 66.7% (large) of wrong initial answers, all 4 CoT samples
  were wrong. On those, every method's per-solve accuracy is about 8–15%.
  - When some CoT sample was right (k ≥ 1), solves reach 17–25%.
- **Strategies don't change solve accuracy.** VALID 10.6% vs INVALID 10.3% (small). On the strategy-breaking arm, ESV
  solves are *worse* than plain re-solves (9.4% vs 13.7%).
- **A solve restarts from scratch.** The solve stage gets the diagnosis and strategy text but not the initial
  reasoning. A localized fix becomes a full re-derivation, which reintroduces errors.
  - 25.6% (small) and 46.1% (large) of solves return the same wrong initial answer.
  - Example, Mistral Small `7_original`: the diagnosis correctly says the minimum in option D is wrong; solve 1
    recomputes the same −1/512.
- **Strategies inherit the flawed setup.** The strategy prompt includes the initial solution, so all 3 strategies reuse
  its model of the problem.
  - Example, Mistral Small `12_original`: all 3 strategies fix the rectangle at the origin, as the failed solution did,
    so the judge rated all 3 INVALID.
- **"Must not be the failed method" breaks correct answers.**
  - On false alarms (the initial answer was right), ESV solves are correct 44.6% / 54.9% of the time. CoT samples on
    the same instances: 67% / 65%. Self-Refine, which may keep the method, keeps 75.6% / 82.4%.
  - Forcing a detour away from a working method is what produces ESV's breaks.
- **The answer checks don't verify anything.** Solves mark options TRUE while contradicting their own working.
  - Example, Qwen 72B `24_original`: "(C) α−γ=1 → TRUE (since α−γ = 1−1 = 0)".
  - When no option fits, the solve outputs no final answer: 7% of large-model solves were wasted votes.

## 3. Recognition checks textbook hypotheses, not the problem's own definitions

- The prompt asks for "signs, domains, intervals, existence, hypotheses of each formula". So models check whether the
  theorem applies, not whether the problem's givens were mapped into it correctly.
- Structural perturbations break exactly that mapping.
  - Example, `27_perturbed` (gold A): q = PQ and r = RP are permuted, a genuine strategy-breaking trap. **All 6 models
    answered ABC and none flagged it.** Their checks read "Law of Cosines applies to any triangle — HOLDS".
- Recognition reads the initial solution's restatement of the problem, which already contains the misreading, so it
  confirms that restatement.
- Large models are confidently unflagging: catch rate 31.5% (Qwen 72B 14.7%).
- A free detector, "CoT samples agree ≤ 2/4", beats it on every metric:

  | Detector | Small: catch / precision | Large: catch / precision |
  |---|---|---|
  | ESV recognition | 57.0% / 77.2% | 31.5% / 66.7% |
  | CoT agreement ≤ 2/4 (free) | 77.3% / 81.8% | 61.1% / 71.7% |

- The diagnosis claims a broken problem condition in 95% of cases, including 116 of 119 false alarms. The prompt asks
  it to name one, so it invents one.

## 4. The judge's measurements can't support the stages

- The judge rates a diagnosis CORRECT if it names *any* genuine error, even a non-decisive one. Diagnoses also describe
  the error without stating the correction.
- So the diagnosis and strategy verdicts don't predict recovery (sections 1–2), and the lifecycle tables can't support
  claims about the stages.

## 5. What is not the problem

- **Scoring:**
  - exact match is the JEEBench standard;
  - the numeric tolerance (0.01) is lenient;
  - 0% of ESV calls were truncated;
  - 3–7% of answers are unreadable, mostly "none of the options" replies.
  - Per-option voting on multiple-answer MCQs would *lower* CoT accuracy (60 → 49 small, 30 → 24 large), so the strict
    MCQ scoring is not hiding progress.
- **Gold labels:** in 9 of the 60 shared instances, ≥ 4 of 6 models agree on the same non-gold answer. The ones checked
  (`27_perturbed`, `20_perturbed`) have correct gold; the originals carry official JEE keys.
- **The vote rule:** no rule closes the gap (section 1).

## 6. Efficiency today (mean completion tokens per instance, large models)

| | CoT ×4 | + method | total |
|---|---|---|---|
| Self-Refine | 4818 | 777 | 5595 |
| Recheck ×3 | 4818 | 1561 | 6379 |
| ESV | 4818 | 1941 | 6759 (+6% vs Recheck ×3) |

- On average, ESV's compute is already close to Recheck ×3.
- A flagged instance costs 6.2k completion and 8.7k prompt tokens, over 6–7 sequential calls.

## 7. Implication for the design

Adding stages helps only if each stage brings in information independent of the original error: the problem text read
afresh, a concrete check, or a competing solution. The current stages re-read the model's own flawed solution.
Evidence-backed changes:

1. **Gate on CoT disagreement instead of a recognition call.** It is free and more accurate. Unanimous 4/4 means keep
   the answer.
2. **Verify the competing candidates the model already has.** On disagreeing instances, extract each distinct CoT
   answer's strategy and its problem-specific *bindings*: each given quoted from the problem and the role it plays in
   the method. Check them against the problem text, and pick the candidate whose strategy survives. This targets the
   +18–20 pp headroom with no new solves.
3. **Repair locally, not from scratch, and don't forbid the original method.** If every candidate fails, do one repair
   that keeps the reasoning up to the failed step and must state the corrected fact.
4. **Use controls that fit the new design:**
   - Universal Self-Consistency (the model picks among its own samples) is the direct competitor of step 2;
   - keep Recheck ×3 and Self-Refine.
   - Metric: accuracy of the choice on disagreeing instances vs majority vote. The chance level there is *below*
     majority (small 137.8 vs 187, large 34.5 vs 51), so the selector must beat majority, not chance.
