#!/usr/bin/env python3
"""Offline end-to-end smoke test: build_dataset -> run_methods -> run_esv3 -> analyze_esv3, with a fake model
client. No API calls.

The fake client recognizes each prompt by its wording and answers in the requested tags with random but well-formed
content, so every code path (rejections, both perturbation modes, the 4+4 samples, ESV v3's candidate verification,
the independent-verifier mode, USC, the report, resume and the expired-key stop) runs on both benchmarks. Outputs go
to a temporary folder, never to data/ or results/.

Run: python experiments/tests/smoke_test_offline.py
"""

import json
import os
import random
import re
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

WORK = Path(tempfile.mkdtemp(prefix="jee_smoke_"))
os.environ["ESV_DATA_DIR"] = str(WORK / "data")
os.environ["ESV_RESULTS_DIR"] = str(WORK / "results")
os.environ["OPENROUTER_API_KEY"] = "offline-test"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_dataset  # noqa: E402
import common  # noqa: E402
import run_methods  # noqa: E402


def detect_format(prompt: str) -> str:
    if "EXACTLY ONE correct option" in prompt or "exactly one option is correct" in prompt or "single capital letter" in prompt:
        return "single"
    if "one or more options" in prompt or "ALL correct options" in prompt:
        return "multiple"
    if "non-negative integer" in prompt or "just the integer" in prompt:
        return "integer"
    if "in LaTeX" in prompt or "exact value" in prompt:  # MATH-500's expression format
        return "expression"
    return "numeric"


# One answer per format, and a different one, so a fake reply can be made to agree or disagree on purpose.
FORMAT_ANSWERS = {"single": ("B", "C"), "multiple": ("AC", "A"), "integer": ("7", "5"), "numeric": ("2.25", "3.5"),
                  "expression": (r"\frac{3}{2}", r"3\sqrt{13}")}


def random_answer(rng: random.Random, fmt: str) -> str:
    return {"single": lambda: rng.choice("ABCD"), "multiple": lambda: rng.choice(["A", "AB", "AC", "BCD", "ABCD"]),
            "integer": lambda: str(rng.randint(0, 9)), "numeric": lambda: rng.choice(["0.5", "2.25", "3", "12.75"]),
            "expression": lambda: rng.choice([r"\frac{3}{2}", r"3\sqrt{13}", "(6,31,-1)", "x^2-4", r"\frac{\pi}{2}"])}[fmt]()


def tags(**fields) -> str:
    return "\n".join(f"<{k}>{v}</{k}>" for k, v in fields.items())


class FakeCompletions:
    def __init__(self, seed: int, fail_with: str = None):
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.fail_with = fail_with
        self.calls = 0

    def create(self, model, messages, max_tokens, temperature=None, seed=None, extra_body=None):
        if self.fail_with:
            raise RuntimeError(self.fail_with)
        prompt = messages[0]["content"]
        with self.lock:
            self.calls += 1
            text = self.respond(prompt)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=len(prompt) // 4, completion_tokens=len(text) // 4,
                                                     total_tokens=(len(prompt) + len(text)) // 4))

    def respond(self, prompt: str) -> str:
        rng = self.rng
        if "Create a perturbation P'" in prompt:
            fmt = detect_format(prompt.split("Answer format of P'")[1])
            if rng.random() < 0.15:
                return tags(status="REJECT")
            mode = "preserving" if "STILL WORKS" in prompt else "breaking"
            answer, other = FORMAT_ANSWERS[fmt]
            naive = answer if mode == "preserving" else other
            opts = {f"option_{l}": f"${i + 1}\\sqrt{{{rng.randint(2, 99)}}}$" for i, l in enumerate("ABCD")} if fmt in ("single", "multiple") \
                else {f"option_{l}": "(empty)" for l in "ABCD"}
            return tags(status="CANDIDATE", P_prime=f"[MODE={mode}] Perturbed problem {rng.randint(0, 10**6)}: compute \\frac{{1}}{{2}}.",
                        **opts, original_strategy="direct formula", changed_condition="changed the domain",
                        failure_reason="sign flips", alternative_strategy="case split", derivation="...",
                        naive_answer=naive, answer_prime=answer, confidence_level="0.9")
        if "Solve this math problem independently" in prompt:
            fmt = detect_format(prompt.split("Additional rules:")[1])
            m = re.search(r"\[MODE=(\w+)\]", prompt)
            answer = FORMAT_ANSWERS[fmt][0]
            if rng.random() < 0.1:
                answer = random_answer(rng, fmt)  # occasional verifier disagreement
            return tags(valid_problem="true", reason="well-posed", derivation="...", answer_prime=answer,
                        confidence_level=str(rng.choice([0.5, 0.9, 0.95])) if m else "0.9")
        if "Check whether an original solution strategy" in prompt:
            mode = re.search(r"\[MODE=(\w+)\]", prompt).group(1)
            reference = re.search(r"established by an independent solver\): (\S+)", prompt).group(1)
            ok = rng.random() < 0.85
            if mode == "preserving":
                return tags(strategy_status="STILL_VALID" if ok else "INVALID", strategy_reason="...",
                            naive_answer=reference if ok else "NONE")
            return tags(strategy_status="INVALID" if ok else "STILL_VALID", strategy_reason="...",
                        naive_answer=("C" if reference.isalpha() else "99") if ok else reference)
        fmt = detect_format(prompt)
        if "Explicit Strategy Verification (ESV) on several solutions" in prompt:  # ESV v3 stages 2-4
            n = len(re.findall(r"^Solution (\d+):", prompt, flags=re.MULTILINE))
            out = {}
            for i in range(1, n + 1):
                out.update({f"s{i}_strategy": "...", f"s{i}_bindings": '"x > 0" -> used x > 0 -> ' + rng.choice(["MATCHES", "MATCHES", "MISMATCH"]),
                            f"s{i}_conditions": "x real - HOLDS"})
            out["divergence"] = "..."
            for i in range(1, n + 1):
                v = rng.choice(["SOUND", "FLAWED", "FLAWED"])
                out.update({f"s{i}_verdict": v, f"s{i}_failure_type": "NONE" if v == "SOUND" else rng.choice(["BINDING", "METHOD", "COMPUTATION"]),
                            f"s{i}_flawed_step": "...", f"s{i}_corrected_fact": "..."})
            return tags(**out)
        if "Select the most consistent response" in prompt:  # Universal Self-Consistency control
            return tags(selected_response=f"Response {rng.randint(1, 4)}")
        if "Give ONLY the final answer" in prompt:
            return tags(final_answer=random_answer(rng, fmt))
        if "Solve the following problem carefully" in prompt:  # CoT samples
            if rng.random() < 0.05:
                return "I think the answer is unclear."  # unreadable reply -> nudged retry
            return tags(reasoning="step by step ...", final_answer=random_answer(rng, fmt))
        raise AssertionError("unrecognized prompt: " + prompt[:200])


class FakeClient:
    def __init__(self, seed: int = 0, fail_with: str = None):
        self.completions = FakeCompletions(seed, fail_with)
        self.chat = SimpleNamespace(completions=self.completions)


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        raise SystemExit(1)


def main() -> int:
    print(f"Working folder: {WORK}")
    # Scoring unit checks
    check(common.score_answer("(A), (C)", "AC", "multiple")["correct"], "MCQ(multiple) parsing")
    check(not common.score_answer("AB", "A", "single")["correct"], "multi-letter reply to single MCQ is wrong")
    check(common.score_answer("2.296", "2.30", "numeric")["correct"], "numeric tolerance 0.01")
    check(not common.score_answer("2.32", "2.30", "numeric")["correct"], "numeric outside tolerance")
    check(not common.score_answer("7.4", "7", "integer")["correct"], "integer exact")
    check(common.score_answer("\\frac{1}{2}", "0.5", "numeric")["correct"], "fraction parsing")
    check(common.normalize_answer("NONE", "single") is None and common.normalize_answer("NONE", "numeric") is None, "NONE is unreadable")
    check(common.majority_answer(["B", "C", "C", None], "single") == "C", "majority vote")
    check(common.majority_answer(["2.25", "2.26", "3", None], "numeric") == "2.25", "numeric majority groups within tolerance")
    # The expression format (MATH-500). Exhaustive checks live in tests/test_expression_answers.py; these confirm
    # that common.py routes the format to it and that majority voting groups equivalent LaTeX.
    check(common.answer_format_of("Expression") == "expression", "question type Expression -> expression format")
    check(common.score_answer(r"\dfrac{14}{3}", r"\frac{14}{3}", "expression")["correct"], r"expression: \dfrac == \frac")
    check(common.score_answer(r"$3\sqrt{13}$", r"\sqrt{117}", "expression")["correct"], "expression: symbolic equivalence")
    check(not common.score_answer("4.67", r"\frac{14}{3}", "expression")["correct"], "expression: a decimal is not the exact value")
    check(not common.score_answer("52", "52_8", "expression")["correct"], "expression: base 8 is not a plain number")
    check(common.normalize_answer("NONE", "expression") is None, "expression: NONE is unreadable")
    check(common.majority_answer([common.normalize_answer(a, "expression") for a in (r"\frac12", r"\frac{1}{2}", "2", None)],
                                 "expression") == r"\frac{1}{2}", "expression majority groups equivalent forms")
    # Stage 1+2: perturbation pipeline on 24 sampled problems
    sys.argv = ["build_dataset.py", "--limit", "24", "--workers", "4"]
    build_dataset.make_client = lambda: FakeClient(seed=1)
    check(build_dataset.main() == 0, "build_dataset ran")
    ds = json.loads((common.FROZEN_DATASET).read_text(encoding="utf-8"))
    check(common.sample_size() == 200, "jeebench samples 200 problems")
    sample = json.loads((common.DATA_DIR / "sample_200.json").read_text(encoding="utf-8"))
    # JEEBench's "index" restarts in every paper, so a problem is identified by (paper, index).
    check(sample["n"] == 200 and len({(p["source"], p["source_index"]) for p in sample["problems"]}) == 200, "200 distinct problems sampled")
    check(all(p["type"] in common.ANSWER_FORMATS for p in sample["problems"]), "sample is math-only with known types")
    split = ds["perturbation_split"]
    print(f"     split {split}, attempted {ds['problems_attempted_per_mode']}, pairs {ds['num_pairs']}")
    check(ds["num_pairs"] > 0 and abs(split["breaking"] - split["preserving"]) <= max(3, ds["num_pairs"] // 4), "balanced split")
    check(all(p["answer_format"] == common.ANSWER_FORMATS[p["type"]] for p in ds["pairs"]), "P' keeps the original answer format")
    check(all(("(A)" in p["perturbed_problem"]) == common.is_choice_format(p["answer_format"]) for p in ds["pairs"]), "options only on MCQs")
    # Resume: rerunning processes nothing new
    calls_before = len(build_dataset.read_log())
    check(build_dataset.main() == 0 and len(build_dataset.read_log()) == calls_before, "build_dataset resumes without redoing work")

    # Methods on all pairs, 3 solvers
    check(run_methods.main(["--workers", "6"], client=FakeClient(seed=2)) == 0, "run_methods ran")
    for model in common.SOLVER_MODELS:
        recs = json.loads((common.RESULTS_DIR / common.model_slug(model) / "records.json").read_text(encoding="utf-8"))["instances"]
        check(len(recs) == 2 * ds["num_pairs"], f"{model}: one record per instance")
        check(all(all(s is not None for s in r["samples"][m]) for r in recs.values() for m in ("dqa", "cot")), f"{model}: 4+4 samples each")
        check(all(s["temperature"] == common.SAMPLE_TEMPERATURE for r in recs.values() for s in r["samples"]["cot"]), f"{model}: sampled at T=0.7")
    fake = FakeClient(seed=3)
    run_methods.main(["--workers", "6"], client=fake)
    check(fake.completions.calls == 0, "run_methods resumes without new calls")

    # A small solver too, so the report's --models small path has records to read
    small = common.SMALL_SOLVER_MODELS[0]
    check(run_methods.main(["--workers", "6", "--models", small], client=FakeClient(seed=12)) == 0, "sampling a small solver ran")
    recs = json.loads((common.RESULTS_DIR / common.model_slug(small) / "records.json").read_text(encoding="utf-8"))["instances"]
    check(all(all(s is not None for s in r["samples"][m]) for r in recs.values() for m in ("dqa", "cot")),
          "small solver: 4+4 samples each")

    # Expired key: stops cleanly instead of burning through every instance
    run_methods.STOP.clear()
    check(run_methods.main(["--workers", "2", "--no-resume", "--models", common.SOLVER_MODELS[0]],
                           client=FakeClient(fail_with="Error code: 401 - API key expired")) == 1, "expired key stops the run")
    run_methods.STOP.clear()
    run_methods.main(["--workers", "6", "--models", common.SOLVER_MODELS[0]], client=FakeClient(seed=4))  # restore

    # ESV v3 (stages 1-3) and Universal Self-Consistency: gold-free, starting from the same CoT samples
    import analyze_esv3  # noqa: E402
    import run_esv3  # noqa: E402
    mk = lambda v, mm=0, vi=0: {"verdict": v, "mismatches": mm, "violations": vi}
    cands = [{"votes": 2, "is_initial": True}, {"votes": 1, "is_initial": False}, {"votes": 1, "is_initial": False}]
    dv = run_esv3.decide
    check(dv(cands[:1], []) == ("unanimous", None) and dv(cands, [mk("FLAWED"), mk("SOUND"), mk("FLAWED")]) == ("one_sound", 1)
          and dv(cands, [mk("SOUND"), mk("SOUND"), mk("FLAWED")]) == ("several_sound", 0)
          and dv(cands, [mk("FLAWED"), mk("SOUND", mm=1), mk("FLAWED")]) == ("none_sound", None)
          and dv(cands, [mk("FLAWED"), mk("SOUND", mm=1), mk("FLAWED")], strict=False) == ("one_sound", 1)
          and dv(cands, [mk(None), mk(None), mk(None)]) == ("unreadable", None), "ESV v3 decision rule (strict SOUND, votes break ties)")
    check(run_esv3.main(["--workers", "6"], client=FakeClient(seed=13)) == 0, "run_esv3 ran")
    for model in common.SOLVER_MODELS:
        folder = common.RESULTS_DIR / common.model_slug(model)
        base = json.loads((folder / "records.json").read_text(encoding="utf-8"))["instances"]
        v3 = json.loads((folder / "esv3_records.json").read_text(encoding="utf-8"))["instances"]
        check(set(v3) == set(base), f"{model}: ESV v3 on every instance ({len(v3)})")
        una = [r for r in v3.values() if len(r["candidates"]) < 2]
        dis = [r for r in v3.values() if len(r["candidates"]) >= 2]
        check(all(r["methods"]["esv3"]["call"] is None and r["methods"]["usc"]["call"] is None and not r["methods"]["esv3"]["changed"]
                  and not r["methods"]["usc"]["changed"] for r in una), f"{model}: unanimous -> no calls, initial kept ({len(una)})")
        check(all(r["methods"]["esv3"]["call"] is not None and r["methods"]["usc"]["call"] is not None
                  and len(r["methods"]["esv3"]["checks"]) == len(r["candidates"]) for r in dis),
              f"{model}: disagreeing -> one verify + one USC call ({len(dis)})")
        check(all(common.answers_match(r["initial"]["normalized"], common.majority_answer(
            [s["normalized_answer"] for s in base[i]["samples"]["cot"]], r["answer_format"]), r["answer_format"])
            or r["initial"]["normalized"] is None for i, r in v3.items()), f"{model}: ESV v3 starts from CoT maj@4")
        ok = True
        for r in dis:
            dec, idx = run_esv3.decide(r["candidates"], r["methods"]["esv3"]["checks"])
            want = r["candidates"][idx]["normalized"] if idx is not None else r["initial"]["normalized"]
            ok &= r["methods"]["esv3"]["decision"] == dec and (want is None or common.answers_match(
                r["methods"]["esv3"]["final_normalized"], want, r["answer_format"]))
        check(ok, f"{model}: ESV v3 finals follow the decision rule")
    # The verify prompt shows each distinct answer once and never the vote counts.
    model = common.SOLVER_MODELS[0]
    folder = common.RESULTS_DIR / common.model_slug(model)
    base = json.loads((folder / "records.json").read_text(encoding="utf-8"))["instances"]
    by_id = {i["instance_id"]: i for p in ds["pairs"] for i in (run_methods.make_instance(p, k) for k in ("original", "perturbed"))}
    iid = next(i for i, r in json.loads((folder / "esv3_records.json").read_text(encoding="utf-8"))["instances"].items() if len(r["candidates"]) >= 2)
    import method_prompts as mp  # noqa: E402
    cands = run_esv3.candidates_of(base[iid]["samples"]["cot"], by_id[iid]["answer_format"])
    prompt = mp.build_v3_verify_prompt(by_id[iid], [{"reasoning": "r", "answer": c["answer"]} for c in cands])
    check(prompt.count("<final_answer>") == len(cands) and "votes" not in prompt,
          "ESV v3 prompt: one solution per distinct answer, no vote counts")
    same = True
    for iid in list(by_id)[:12]:
        outs = []
        for gold in (by_id[iid]["gold"], "0"):
            o = run_esv3.V3Runner(FakeClient(seed=14), model, None).run({**by_id[iid], "gold": gold}, base[iid]["samples"]["cot"])
            outs.append([o["methods"]["esv3"]["decision"], o["methods"]["esv3"]["final_answer"], o["methods"]["usc"]["final_answer"]])
        same &= outs[0] == outs[1]
    check(same, "ESV v3 gold-free: a different gold answer gives identical decisions and final answers")
    one = base[iid]["samples"]["cot"][0]
    fake = FakeClient(seed=17)
    o = run_esv3.V3Runner(fake, model, None).run(by_id[iid], [one] * 4)
    check(fake.completions.calls == 0 and o["methods"]["esv3"]["decision"] == "unanimous" and not o["methods"]["esv3"]["changed"]
          and not o["methods"]["usc"]["changed"], "ESV v3: unanimous samples -> no calls, initial answer kept")
    fake = FakeClient(seed=15)
    run_esv3.main(["--workers", "6"], client=fake)
    check(fake.completions.calls == 0, "run_esv3 resumes without new calls")
    count = len(json.loads((folder / "esv3_records.json").read_text(encoding="utf-8"))["instances"])
    run_esv3.main(["--workers", "2", "--models", model, "--limit-pairs", "2"], client=FakeClient(seed=16))
    check(len(json.loads((folder / "esv3_records.json").read_text(encoding="utf-8"))["instances"]) == count, "run_esv3 --limit-pairs keeps other records")
    # Independent verifier: its checks land in a separate file, tagged, with no USC duplicate
    run_esv3.main(["--workers", "2", "--models", model, "--limit-pairs", "2", "--verifier-model", "openai/gpt-oss-120b"],
                  client=FakeClient(seed=17))
    vrecs = json.loads((folder / "esv3_records.gpt-oss-120b.json").read_text(encoding="utf-8"))["instances"]
    check(len(vrecs) > 0 and all(r["verifier_model"] == "openai/gpt-oss-120b" for r in vrecs.values()),
          "--verifier-model writes its own records file, tagged with the verifier")
    check(all(r["methods"]["usc"]["call"] is None for r in vrecs.values()), "verifier mode makes no USC calls")
    check(all(len(r["methods"]["esv3"]["checks"]) == len(r["candidates"]) for r in vrecs.values() if len(r["candidates"]) >= 2),
          "verifier mode checks every candidate")
    check(len(json.loads((folder / "esv3_records.json").read_text(encoding="utf-8"))["instances"]) == count,
          "the self-verify records are untouched by a verifier run")
    check(analyze_esv3.main(["--verifier-model", "openai/gpt-oss-120b"]) == 0,
          "analyze_esv3 reads the verifier records via --verifier-model")
    check(analyze_esv3.main([]) == 0 and "Go / no-go" in (common.RESULTS_DIR / "analysis" / "esv3_report.md").read_text(encoding="utf-8"),
          "analyze_esv3 ran (stages 1-3 and the go / no-go line)")

    # A second benchmark (GSM8K, integer answers) through the same pipeline, into its own data/ and results/ folders
    import benchmarks  # noqa: E402
    jee_results = sorted(p.relative_to(WORK) for p in (WORK / "results" / "jeebench").rglob("*.json"))
    fake_src = [{"source": "GSM8K test", "source_index": i, "question": f"Problem {i}: a shop sells {i + 2} pens at $3 each. Total?", "gold": str(3 * (i + 2))}
                for i in range(30)] + [{"source": "GSM8K test", "source_index": 30, "question": "negative", "gold": "-4"}]
    common.save_json(benchmarks.gsm8k_source_path(), fake_src)
    check(len(benchmarks.load_source("gsm8k")) == 30 and all(r["type"] == "Integer" for r in benchmarks.load_source("gsm8k")),
          "gsm8k source: cached file read, negative answers dropped, type Integer")
    sys.argv = ["build_dataset.py", "--limit", "8", "--workers", "4", "--benchmark", "gsm8k", "--quiet"]
    build_dataset.make_client = lambda: FakeClient(seed=21)
    check(build_dataset.main() == 0 and common.BENCHMARK == "gsm8k" and common.FROZEN_DATASET == WORK / "data" / "gsm8k" / "frozen_dataset.json",
          "build_dataset --benchmark gsm8k writes under data/gsm8k/")
    g = json.loads(common.FROZEN_DATASET.read_text(encoding="utf-8"))
    check(g["benchmark"] == "gsm8k" and g["num_pairs"] > 0 and all(p["answer_format"] == "integer" for p in g["pairs"]), f"gsm8k pairs are integer-format ({g['num_pairs']})")
    check(run_methods.main(["--workers", "4", "--samples-only", "--benchmark", "gsm8k", "--models", common.SOLVER_MODELS[0]], client=FakeClient(seed=22)) == 0
          and (WORK / "results" / "gsm8k" / common.model_slug(common.SOLVER_MODELS[0]) / "records.json").exists(), "run_methods --benchmark gsm8k writes under results/gsm8k/")
    check(analyze_esv3.main(["--benchmark", "gsm8k"]) == 0 and (WORK / "results" / "gsm8k" / "analysis" / "esv3_report.md").exists(), "analyze_esv3 --benchmark gsm8k")
    check(sorted(p.relative_to(WORK) for p in (WORK / "results" / "jeebench").rglob("*.json")) == jee_results, "the gsm8k run left the jeebench results untouched")

    # A third benchmark (MATH-500): 250 sampled instead of 200, and three answer formats in one benchmark, including
    # `expression`. Every stage runs end to end, into data/math500/ and results/math500/.
    gsm_results = sorted(p.relative_to(WORK) for p in (WORK / "results" / "gsm8k").rglob("*.json"))
    fake_math = []
    for i in range(36):
        kind = ("Integer", "Numeric", "Expression")[i % 3]
        gold = {"Integer": str(10 + i), "Numeric": f"-{i}.5", "Expression": r"\frac{%d}{7}" % (i + 1)}[kind]
        fake_math.append({"source": "MATH-500 test", "source_index": i, "type": kind, "gold": gold,
                          "question": f"Problem {i}: find the value of $x_{{{i}}}$ in simplest form.",
                          "subject": ("Algebra", "Precalculus", "Geometry")[i % 3], "level": 1 + i % 5,
                          "unique_id": f"test/fake/{i}.json", "solution": "..."})
    common.save_json(benchmarks.math500_source_path(), fake_math)
    rows = benchmarks.load_source("math500")
    check(len(rows) == 36 and {r["type"] for r in rows} == {"Integer", "Numeric", "Expression"},
          "math500 source: cached file read, all three question types kept")
    check(benchmarks.math500_type("7") == "Integer" and benchmarks.math500_type("-2.5") == "Numeric"
          and benchmarks.math500_type(r"\frac{1}{2}") == "Expression", "math500_type reads the type off the gold")
    sys.argv = ["build_dataset.py", "--limit", "18", "--workers", "4", "--benchmark", "math500", "--quiet"]
    build_dataset.make_client = lambda: FakeClient(seed=31)
    check(build_dataset.main() == 0 and common.BENCHMARK == "math500" and common.sample_size() == 250
          and build_dataset.sample_path() == WORK / "data" / "math500" / "sample_250.json",
          "build_dataset --benchmark math500 samples 250 into data/math500/sample_250.json")
    m = json.loads(common.FROZEN_DATASET.read_text(encoding="utf-8"))
    check(m["benchmark"] == "math500" and m["num_pairs"] > 0, f"math500 pairs built ({m['num_pairs']})")
    check(all(p["answer_format"] in ("integer", "numeric", "expression") for p in m["pairs"])
          and all("(A)" not in p["perturbed_problem"] for p in m["pairs"]), "math500 pairs are option-free")
    check(all(p.get("subject") and p.get("level") for p in m["pairs"]), "subject and level are carried into the pairs")
    check(all(common.normalize_answer(p["answer_prime"], p["answer_format"]) is not None for p in m["pairs"]),
          "every math500 answer_prime is readable in its own format")
    solver = common.SMALL_SOLVER_MODELS[0]
    check(common.solver_max_tokens() == 6000, "math500 raises the solver token cap")
    check(run_methods.main(["--workers", "4", "--benchmark", "math500", "--models", solver], client=FakeClient(seed=32)) == 0,
          "run_methods --benchmark math500 ran")
    check(run_esv3.main(["--workers", "4", "--benchmark", "math500", "--models", solver], client=FakeClient(seed=33)) == 0,
          "run_esv3 --benchmark math500 ran")
    check(run_esv3.main(["--workers", "4", "--benchmark", "math500", "--models", solver,
                         "--verifier-model", "openai/gpt-oss-120b"], client=FakeClient(seed=34)) == 0,
          "run_esv3 --benchmark math500 --verifier-model ran")
    folder = WORK / "results" / "math500" / common.model_slug(solver)
    check((folder / "records.json").exists() and (folder / "esv3_records.json").exists()
          and (folder / "esv3_records.gpt-oss-120b.json").exists(), "math500 results land in results/math500/<model>/")
    check(analyze_esv3.main(["--benchmark", "math500", "--models", "small"]) == 0,
          "analyze_esv3 --benchmark math500 ran")
    report = (WORK / "results" / "math500" / "analysis" / "esv3_report_small.md").read_text(encoding="utf-8")
    formats = {p["answer_format"] for p in m["pairs"]}
    check("MATH-500" in report and (len(formats) < 2 or "By answer format" in report),
          f"the math500 report is labelled MATH-500 and splits by answer format ({sorted(formats)})")
    # The split itself, independently of which formats this small run happened to accept.
    synthetic = [{"format": f, "correct": {"cot_maj": True, "usc": True, "esv3": f == "integer"}}
                 for f in ("integer", "numeric", "expression")]
    split = analyze_esv3.format_breakdown(synthetic, ("cot_maj", "usc", "esv3"))
    check("answer format" in split and "expression" in split and not analyze_esv3.format_breakdown(synthetic[:1], ("cot_maj",)),
          "format_breakdown splits a mixed-format benchmark and stays silent on a single-format one")
    check(analyze_esv3.main(["--benchmark", "math500", "--models", "small", "--verifier-model", "openai/gpt-oss-120b"]) == 0,
          "analyze_esv3 --benchmark math500 --verifier-model ran")
    check(sorted(p.relative_to(WORK) for p in (WORK / "results" / "gsm8k").rglob("*.json")) == gsm_results,
          "the math500 run left the gsm8k results untouched")
    common.select_benchmark("jeebench")
    print(f"\nAll offline checks passed. Outputs: {WORK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
