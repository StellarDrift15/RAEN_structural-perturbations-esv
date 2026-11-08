# Why ESV v3 fails on GSM8K: the bottleneck, layer by layer

Dev set: pairs 0–59, small models (Qwen 2.5 7B, Mistral Nemo, Gemma 3 4B), 360 instances, 89 disagreeing.
Records: `results/gsm8k/<model>/esv3_records.json`. Report: `esv3_report_small.upto60.md`
(`analyze_esv3.py --benchmark gsm8k --models small --max-pair 60`). All numbers below are
reproducible offline from the stored records (no API calls). Written 2026-11-04.

## Headline

Dev-split snapshot of the original NO-GO finding. `extra tokens/instance` here counts **completion tokens
only**, as the early reports did; the full bill, prompt side included and solver separated from verifier
tokens, is in "Compute cost" below.

| Method | accuracy | fixes | breaks | extra completion tokens/instance |
|---|---|---|---|---|
| CoT maj@4 (baseline) | 87.7% | — | — | — |
| ESV v3 (stages 1–3) | 85.2% | 6 | 15 | 234 |
| Universal Self-Consistency | 88.5% | 4 | 1 | 3 |
| CoT pass@4 (oracle ceiling) | 93.3% | — | — | — |

NO-GO under the pre-registered rule (≥ +3 pp over CoT maj@4 AND > USC). ESV v3 is significantly worse
than USC (McNemar 6/18, p = 0.023) at ~78× the extra completion tokens (1.9× counting the whole
bill: +34.6% vs +18.7% over the baseline).

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
SOUND agree exactly; the rule did not need to change. Report: `esv3_report_small.gpt-oss-120b.upto60.md`.

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

## Compute cost (2026-11-08)

Every number here is read off the `usage` and `latency_seconds` that `common.call_stage` stored on each call —
3,168 solver calls per model, plus 303 verification calls and 303 USC calls in all — so the whole section
reproduces offline with no API calls:
`analyze_esv3.py --benchmark gsm8k --models small [--verifier-model openai/gpt-oss-120b]`, section
"Compute cost".

The comparison is unusually clean because **no method re-solves the problem**. All four share the same 4 CoT
samples and add at most one call, so the baseline is a constant and the extra call is the entire difference.

### Per instance, pooled over the three solvers (all 198 pairs, 1,170 instances)

`serial` is billed compute time. `critical path` is what a user waits for: `run_methods.py` issues the 4 samples
concurrently, so only the slowest is on the path, and the extra call cannot start until all 4 are back.

| method | accuracy | calls | solver tok | verifier tok | total tok | vs maj@4 | serial | path |
|---|---|---|---|---|---|---|---|---|
| Direct QA maj@4 | 12.1% | 4.00 | 1,095 | - | 1,095 | -54.0% | 2.4 s | 1.0 s |
| CoT pass@1 | 85.1% | 1.00 | 595 | - | 595 | -75.0% | 4.8 s | 4.8 s |
| CoT maj@4 (baseline) | 88.5% | 4.00 | 2,379 | - | 2,379 | - | 19.2 s | 6.5 s |
| Universal Self-Consistency | 89.3% | 4.26 | 2,799 | - | 2,799 | +17.6% | 19.4 s | 6.7 s |
| ESV, self-check | 85.1% | 4.26 | 3,158 | - | 3,158 | +32.7% | 23.5 s | 10.8 s |
| **ESV, independent check** | **93.5%** | **4.26** | **2,379** | **1,038** | **3,417** | **+43.6%** | **26.4 s** | **13.7 s** |
| CoT pass@4 (oracle ceiling) | 94.5% | 4.00 | 2,379 | - | 2,379 | - | 19.2 s | 6.5 s |

The headline: **ESV buys +5.0 pp for +43.6% tokens and +0.26 calls** — not the 2x or 5x that a
"verify every answer" pipeline would cost. Three facts do that work.

### 1. What the extra compute buys

`net` = fixes - breaks, so a method that trades a correct answer for a wrong one is charged for both.

| method | Δ accuracy | fixes / breaks | net | extra tok | extra wait | tok per net fix |
|---|---|---|---|---|---|---|
| Direct QA maj@4 | -76.3 pp | 7 / 900 | -893 | -1,284 | -5.5 s | no net gain |
| Universal Self-Consistency | +0.9 pp | 19 / 9 | +10 | +420 | +0.2 s | 49,112 |
| ESV, self-check | -3.3 pp | 18 / 57 | -39 | +779 | +4.3 s | no net gain |
| **ESV, independent check** | **+5.0 pp** | **61 / 2** | **+59** | **+1,038** | **+7.2 s** | **20,585** |
| CoT pass@4 (oracle ceiling) | +6.1 pp | 71 / 0 | +71 | 0 | 0 s | 0 |

ESV-independent costs 2.5x USC's token overhead and returns 5.9x the net fixes, so it is **2.4x more
token-efficient per net fix than USC** (20.6k vs 49.1k). The self-check arm is strictly dominated: it pays
+779 tokens and +4.3 s to lose 3.3 pp, which is the cost side of the layer-3 diagnosis above.

The oracle row is the useful framing for the compute angle: the right answer is *already in the 4 samples we
paid for* 94.5% of the time. ESV is not buying reasoning, it is buying selection — which is why its cost is a
fraction of the baseline rather than a multiple of it.

### 2. The disagreement gate is the efficiency mechanism

Stage 1 is code over answers already paid for, so it costs nothing and retires three-quarters of the work.

| | per instance |
|---|---|
| Gate fires (>= 2 distinct candidates) | 303/1170 (25.9%) |
| A verification call, when it runs | 4,008 tokens, 27.8 s |
| Amortized over every instance | 1,038 tokens, 7.2 s |
| The same protocol ungated | 4,008 tokens, 27.8 s |
| **Verification spend the gate avoids** | **74.1%** |
| Cost on a gated-out instance | 0 tokens, 0 s — ESV returns the maj@4 answer untouched |

A fired verification call is large: 4,008 tokens, bigger than all 4 CoT samples put together. Ungated, ESV
would cost +168% instead of +43.6%, and no accuracy would be gained — a unanimous instance has one candidate,
so there is nothing to choose between.

### 3. The spend is adaptive, and tracks where the gain is

| arm | instances | gate fires | maj@4 | ESV | baseline tok | extra tok | vs baseline |
|---|---|---|---|---|---|---|---|
| originals | 576 | 20.3% | 90.6% | 94.4% | 2,279 | +757 | +33.2% |
| **strategy-breaking P'** | 297 | **45.1%** | 79.1% | 87.5% | 2,764 | **+2,009** | **+72.7%** |
| preserving P' | 297 | 17.5% | 93.6% | 97.6% | 2,188 | +612 | +28.0% |

Disagreement is a free proxy for "the strategy may have broken": the gate fires 2.6x more often on breaking P'
than on preserving P', which is exactly where the accuracy gap is. The same pattern holds per solver — and the
cost of a *fix* stays nearly constant while total spend scales to how much repair the solver needs:

| solver | gate fires | baseline tok | ESV tok | vs baseline | net fixes | tok per net fix |
|---|---|---|---|---|---|---|
| Qwen 2.5 7B | 18.7% | 2,025 | 2,761 | +36.3% | +14 | 20,500 |
| Mistral Nemo 12B | 41.8% | 1,958 | 3,512 | +79.4% | +33 | 18,363 |
| Gemma 3 4B | 17.2% | 3,153 | 3,978 | +26.1% | +12 | 26,793 |

Nemo disagrees with itself most, pays the most (+79%) and gains the most (+8.4 pp). Nobody pays for headroom
they do not have.

### Splits

| method | dev (357) | held-out (813) | all (1,170) |
|---|---|---|---|
| CoT maj@4 | 2,446 tok / 6.7 s | 2,349 tok / 6.4 s | 2,379 tok / 6.5 s |
| USC | +18.7% / 6.9 s | +17.1% / 6.6 s | +17.6% / 6.7 s |
| ESV, self-check | +34.6% / 11.0 s | +31.9% / 10.7 s | +32.7% / 10.8 s |
| ESV, independent check | +44.8% / 12.7 s | +43.1% / 14.2 s | +43.6% / 13.7 s |

The cost profile is stable across splits, so the held-out accuracy gain did not come with a cost surprise.

### The honest caveat: the verifier is a bigger model

The 1,038 verifier tokens are billed by gpt-oss-120b while the 2,379 solver tokens are billed by a 4-12B
model, so a flat token count understates the check. With k = the verifier's price per token over the solver's,
in solver-token equivalents:

| k | ESV tok / instance | vs maj@4 | check vs USC's overhead | per net fix |
|---|---|---|---|---|
| 1 | 3,417 | +43.6% | 2.5x | 20,585 |
| 2 | 4,455 | +87.3% | 4.9x | 41,170 |
| 5 | 7,569 | +218.2% | 12.4x | 102,925 |
| 10 | 12,759 | +436.4% | 24.7x | 205,849 |

**Break-even against USC is k = 2.4.** Below it ESV is the cheaper way to buy a fix; above it USC is. But the
two are not interchangeable at any price: USC's total gain is capped at +10 net fixes against ESV's +59, so
more USC cannot reach 93.5%. The right way to read the table is that ESV buys accuracy USC cannot, and the
price of that accuracy is set by k — plug in the providers' list prices; this report assumes none.

Two latency facts belong in the paper next to this. The extra call is **strictly sequential** (it needs all 4
answers), so it lands on the critical path in full: 6.5 s -> 13.7 s, a 2.1x wait, and 34.3 s on the 25.9% of
instances where it actually fires. And `cot_pass1` is the real efficiency frontier for anyone who cares about
latency over accuracy: 595 tokens and 4.8 s for 85.1%, i.e. 90% of ESV's accuracy for 17% of its tokens.

### Open compute questions

1. **The size control is also a cost control.** The pending judge-select and same-size cross-verifier controls
   decide whether the +43.6% has to be spent on a 120B model. A same-size independent verifier would put k near
   1 and make ESV unambiguously cheaper per fix than USC.
2. **Not yet measured: spending the same tokens on more samples.** CoT maj@8 costs about +100% tokens; ESV
   costs +43.6%. The oracle ceiling rises slowly with n, so the comparison probably favours ESV, but it is a
   reviewer's first question and it is not in the records.
3. **The verification call is not optimized.** 4,008 tokens per fire, of which ~2,000 are completion, for a
   form of 4n+1 tags. Nothing has been tried to shorten it, so the +43.6% is an upper bound on this design.

Deck: `presentation/ESV_Project_Proposal_v3.pptx` (16 slides) carries these numbers plus the pipeline, the
verification-call internals, the pair-131 worked example and the two compute slides built from this section.
