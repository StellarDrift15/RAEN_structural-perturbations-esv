# Why ESV v3 fails on GSM8K: the bottleneck, layer by layer

Dev set: pairs 0–59, small models (Qwen 2.5 7B, Mistral Nemo, Gemma 3 4B), 360 instances, 89 disagreeing.
Records: `results/gsm8k/<model>/esv3_records.json`. Report: `esv3_report_small.md`. All numbers below are
reproducible offline from the stored records (no API calls). Written 2026-11-04.

## Headline

| Method | accuracy | fixes | breaks | extra tokens/instance |
|---|---|---|---|---|
| CoT maj@4 (baseline) | 87.7% | — | — | — |
| ESV v3 (stages 1–3) | 85.2% | 6 | 15 | 234 |
| Universal Self-Consistency | 88.5% | 4 | 1 | 3 |
| CoT pass@4 (oracle ceiling) | 93.3% | — | — | — |

NO-GO under the pre-registered rule (≥ +3 pp over CoT maj@4 AND > USC). ESV v3 is significantly worse
than USC (McNemar 6/18, p = 0.023) at ~78× the token cost.

## What the benchmark switch DID fix

On JEE the SOUND verdict carried zero information: right and wrong candidates were both called SOUND 36%
of the time. On GSM8K it discriminates:

| | called SOUND |
|---|---|
| right candidates (n=66) | 57.6% |
| wrong candidates (n=162) | 35.2% |

So the dataset premise now holds (failures concentrate on strategy-breaking P′: 79.1% vs 90.6% on
originals, and two-thirds of real failures are on breaking P′ after excluding GSM8K label errors). The
signal exists. It is just far too weak, for the reasons below.

## Layer 1: the arithmetic of selection

The pass@4 − maj@4 gap on dev is 21 fixable instances (+5.8 pp). To realise any of it, a selector must
overturn the vote — and the vote is itself a strong signal:

| a candidate with… | P(correct) |
|---|---|
| 1 vote | 9.7% (15/154) |
| 2 votes | 51.1% (24/47) |
| 3 votes | 77.8% (28/36) |

The SOUND verdict adds almost nothing on top, and at 3 votes it is *negatively* informative:

| | FLAWED | SOUND |
|---|---|---|
| 1 vote | 6.2% | 15.5% |
| 2 votes | 45.8% | 56.5% |
| 3 votes | 80.0% | **76.2%** |

Likelihood ratio of SOUND ≈ **1.64**. Required to overturn the vote:

| to lift… | needs LR |
|---|---|
| 1-vote candidate over a 2-vote majority | > 9.7 |
| 1-vote candidate over a 3-vote majority | > 32.4 |
| 2-vote candidate over a 3-vote majority | > 3.4 |

The verdict is 6–20× too weak everywhere except a tied vote (prior odds 1:1). The 21 fixable instances
split accordingly: **9 vote ties** (winnable with a weak signal) and **12 minority-right** (unreachable
at LR 1.64).

## Layer 2: no decision rule can rescue it

Nine alternative stage-3 rules evaluated offline on the stored checks, plus the Bayes-optimal selector
with P(correct | votes, SOUND, mismatch, violation) fitted on this very data (an optimistic upper bound):

| rule | net vs CoT maj@4 | fixes | breaks |
|---|---|---|---|
| current (one SOUND wins, ignores votes) | −2.5 pp | 6 | 15 |
| require a quoted MISMATCH against the majority | +0.0 pp | 1 | 1 |
| SOUND breaks vote ties only | **+1.1 pp** | 4 | 0 |
| Bayes-optimal on every stored signal (overfit) | **+1.1 pp** | — | — |
| oracle (sees gold) | +5.8 pp | 21 | 0 |

The current rule's loss comes entirely from the `one_sound` branch (n=61: 62.3% → 45.9%, 5 fixes vs 15
breaks): it discards a 62%-right majority on LR-1.64 evidence. But even the best rule caps at +1.1 pp —
below the bar, indistinguishable from USC's +0.8 pp, and dev-tuned. **The missing 4.7 pp is information
the verifier does not produce, not a rule-design or implementation problem.**

## Layer 3: WHY the verdict is weak — the checker misjudges what it quotes

On the 41 disagreeing breaking-P′ verify calls (the exact case ESV was designed for):

- The checker's bindings/conditions/divergence text **engages the changed condition in 83%** of calls
  (≥35% content-word overlap with the pair's `changed_condition`). It is not failing to look.
- Engaged → ESV still ends wrong in 21/34. Trap candidates are called SOUND 36% of the time; right
  candidates only 52%.

Two transcripts show the mechanism:

**4_perturbed (Sara, mowing pays $0 for the first 4 times).** The checker quotes the perturbed clause
verbatim — *"$4 for each time she mows the lawn after the first 4 times"* — then marks the trap solution's
`$4x` (which pays every mowing) as **MATCHES**, verdict SOUND. The evidence is on the page; the judgment
ratifies the misuse anyway.

**36_perturbed (relay, first leg shortened to 380 m, nominal times unchanged).** The checker writes
*"the first runner runs 380 m in 60 s — HOLDS"*, endorses the trap solution's per-meter rescaling as
"correctly calculates", verdict SOUND, and its divergence section actively argues for the wrong side.

Conclusion: **verifying a binding (does this solution use this quoted given correctly?) is a reasoning
task of the same difficulty as solving the problem, and the same model fails it at the same rate, in the
same direction** — it holds the same prior that made the trap tempting. Self-verification has no
generator–verifier gap to exploit here; the verdict is close to a noisy re-read of the vote (which is why
P(correct) at 3 votes is unchanged by SOUND). This is the same conclusion the JEE analysis reached
("the checker re-encodes the solver's confidence"), now with the mechanism visible at transcript level.

## The one untested lever: an independent verifier

Everything above is *self*-verification. An independent checker's errors need not be correlated with the
solver's prior, and the thresholds it must clear are now quantified:

- LR ≈ 3.4 → 2-vote-majority overturns come into range (~+2 pp)
- LR ≈ 10 → the 12 minority-right instances come into range (clears the +3 pp bar)

Test (89 verify calls on the stored candidates, ~$0.30; `--verifier-model` added to run_esv3.py
2026-11-04, records land in `esv3_records.<verifier>.json`, nothing overwritten):

```
python run_esv3.py --benchmark gsm8k --workers 12 --limit-pairs 60 --verifier-model openai/gpt-oss-120b --models qwen/qwen-2.5-7b-instruct mistralai/mistral-nemo google/gemma-3-4b-it
python analyze_esv3.py --benchmark gsm8k --models small --verifier-model openai/gpt-oss-120b
```

Read the "per candidate" table of the new report: if right-SOUND vs wrong-SOUND stays near 58/35, the
negative result is complete across two benchmarks, six solvers, and both self- and cross-model
verification — a strong analysis-paper claim. If it jumps to ~75/25 or better, ESV is viable with the
tie-aware decision rule and the held-out pairs (60–197) are still clean for confirmation.

## Outcome of the independent-verifier test (2026-11-04, dev pairs 0-59, verifier openai/gpt-oss-120b)

**GO.** The detector jumped from LR 1.64 to LR ~26 (right candidates SOUND 81.8%, wrong 3.1%, precision
91.5%). Selection on disagreeing instances: 72.1% vs the 76.7% oracle. Pooled: ESV v3 92.2% vs CoT maj@4
87.7% (+4.5 pp, 16 fixes, 0 breaks, p<0.001); self-model USC was 88.5%. Breaking arm 77.8% -> 86.7%.
The one_sound branch flipped from the liability (5/15 self) to the engine (16/0). Verdict-only and strict
SOUND agree exactly; the rule did not need to change. Report: `esv3_report_small.gpt-oss-120b.md`.

Confirms the layer-3 diagnosis: the bindings/MATCHES judgment was the bottleneck, and it is a capability
+ independence property of the checker, not of the protocol. Next: held-out confirmation (pairs 60-197)
and the structure controls (judge-select without ESV stages; a same-size independent verifier) before
any claim is written.

## Held-out confirmation (2026-11-04, pairs 60-197, 813 instances, never used for tuning)

| method | accuracy | fixes / breaks | vs CoT maj@4 |
|---|---|---|---|
| CoT maj@4 | 88.8% | - | - |
| USC | 89.7% | 11 / 4 | +0.9 pp (p=0.118) |
| ESV, self-check | 85.1% | 12 / 42 | -3.7 pp (p=5e-5) |
| **ESV, gpt-oss-120b check** | **94.1%** | **45 / 2** | **+5.3 pp (p=1.6e-11)** |

ESV-independent vs USC: +39 / -3, +4.4 pp, p=5.6e-9. Detector on held-out: right candidates SOUND
88.4%, wrong 1.2% (LR ~74). Every model shows the same ordering; Nemo gains +8.5 pp. Both pre-registered
conditions pass on data untouched by any tuning, and the self-check failure replicates (42 breaks).
Reports from repo code: `analyze_esv3.py --min-pair 60` -> `esv3_report_small.from60.md` and
`esv3_report_small.gpt-oss-120b.from60.md`.

Remaining before the claim is reviewer-proof: (1) judge-select control - same verifier, same candidates,
no ESV structure; (2) a same-size cross-verifier (e.g. Nemo checks Qwen) to separate independence from
capability; (3) the large-solver arm if time allows.

## Final results table (all methods, all splits, 2026-11-04)

Pooled over the three small solvers. dev = pairs 0-59, held-out = pairs 60-197 (no tuning), all = 198 pairs.
fixes / breaks and p are against CoT maj@4 on the same instances (exact McNemar).

| method | dev (357) | held-out (813) | all (1170) | all: fixes/breaks | all: p |
|---|---|---|---|---|---|
| Direct QA maj@4 | 13.7% | 11.4% | 12.1% | 7 / 900 | 1.8e-256 |
| CoT pass@1 | 85.0% | 85.2% | 85.1% | - | - |
| CoT maj@4 (baseline) | 87.7% | 88.8% | 88.5% | - | - |
| Universal Self-Consistency | 88.5% | 89.7% | 89.3% | 15 / 5 | 0.041 |
| ESV, self-check | 85.2% | 85.1% | 85.1% | 18 / 57 | 7.2e-06 |
| **ESV, independent check** | **92.2%** | **94.1%** | **93.5%** | **61 / 2** | **4.4e-16** |
| CoT pass@4 (oracle ceiling) | 93.3% | 95.1% | 94.5% | - | - |

ESV-independent vs USC, all 198: +52 / -3, +4.2 pp, p=1.5e-12. Per solver (all 198, CoT maj@4 -> ESV-independent):
Qwen 2.5 7B 91.0 -> 94.6 (14/0), Mistral Nemo 84.4 -> 92.8 (35/2), Gemma 3 4B 90.0 -> 93.1 (12/0).
By arm (all 198): originals 90.6 -> 94.4 (23/1), breaking P' 79.1 -> 87.5 (25/0), preserving P' 93.6 -> 97.6 (13/1).
Detector, all 198: self-check right-SOUND 57.0% vs wrong-SOUND 36.5% (LR 1.6, precision 44.8%); gpt-oss-120b
86.7% vs 1.8% (LR 47.5, precision 96.1%).

Deck: `presentation/ESV_Project_Proposal_v3.pptx` (14 slides) carries these numbers plus the pipeline, the
verification-call internals and the pair-131 worked example.
