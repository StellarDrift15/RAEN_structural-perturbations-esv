#!/usr/bin/env python3
r"""Source problems per benchmark, all in one shape: {source, source_index, type, question, gold}.

  jeebench  the pilot's JEEBench file, math subset: MCQ, MCQ(multiple), Integer and Numeric questions.
  gsm8k     openai/gsm8k (config main, test split): 1319 grade-school word problems with integer answers. Pulled once
            from Hugging Face into data/gsm8k/source.json and read from there afterwards. The two problems with a
            negative answer are left out: the integer answer format, and the perturbation rules built on it, are
            non-negative.
  math500   HuggingFaceH4/MATH-500 (test split): the 500 curated MATH problems of Lightman et al. (2023), "Let's
            Verify Step by Step". Pulled once into data/math500/source.json. All 500 are kept; the question type is
            read off the shape of the published answer, because MATH answers are not all plain numbers:
              Integer     a non-negative integer (293 problems)         -> scored exactly
              Numeric     a negative integer or a decimal (23)          -> scored to +-0.01
              Expression  any other LaTeX value (184): \frac{14}{3}, 3\sqrt{13}, 90^\circ, (6,31,-1), 52_8,
                          \begin{pmatrix}...\end{pmatrix}, \text{Evelyn} -> scored by expression_answers.py
            Keeping all three types is what makes the sample a sample of MATH-500 rather than of its integer subset,
            and it preserves the subject mix (Precalculus and Intermediate Algebra are mostly Expression-valued).

build_dataset.py samples each benchmark's "n" problems (common.BENCHMARKS, seed SAMPLE_SEED) and perturbs them;
nothing downstream reads the source again.

  python experiments/benchmarks.py gsm8k      # pull (if needed) and summarize a benchmark's source
  python experiments/benchmarks.py math500    # pull MATH-500 and print its type / subject / level breakdown
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

JEEBENCH_SOURCE = common.REPO_DIR / "pilot_experiment" / "src" / "dataset.json"
GSM8K_HF = ("openai/gsm8k", "main", "test")
MATH500_HF = ("HuggingFaceH4/MATH-500", None, "test")


def load_jeebench() -> List[Dict[str, Any]]:
    rows = json.loads(JEEBENCH_SOURCE.read_text(encoding="utf-8"))
    return [{"source": p["description"], "source_index": int(p["index"]), "type": p["type"], "question": p["question"],
             "gold": str(p["gold"]).strip()} for p in rows if p["subject"] == "math"]


def gsm8k_source_path() -> Path:
    return common.DATA_ROOT / "gsm8k" / "source.json"


def pull_gsm8k() -> List[Dict[str, Any]]:
    """Downloads the GSM8K test split (needs the `datasets` package and network access) and caches it."""
    from datasets import load_dataset  # imported here so nothing else needs the package

    rows = []
    for i, r in enumerate(load_dataset(GSM8K_HF[0], GSM8K_HF[1], split=GSM8K_HF[2])):
        match = re.search(r"####\s*(.+)$", r["answer"].strip())
        rows.append({"source": "GSM8K test", "source_index": i, "question": r["question"],
                     "gold": match.group(1).strip().replace(",", "") if match else None, "solution": r["answer"]})
    common.save_json(gsm8k_source_path(), rows)
    return rows


def load_gsm8k() -> List[Dict[str, Any]]:
    rows = common.load_json(gsm8k_source_path()) or pull_gsm8k()
    return [{"source": r["source"], "source_index": int(r["source_index"]), "type": "Integer", "question": r["question"],
             "gold": str(r["gold"])} for r in rows if r.get("gold") is not None and re.fullmatch(r"\d+", str(r["gold"]))]


def math500_source_path() -> Path:
    return common.DATA_ROOT / "math500" / "source.json"


def math500_type(gold: str) -> str:
    """The question type implied by a published MATH answer; see this module's docstring for the three cases."""
    answer = str(gold).strip()
    if re.fullmatch(r"\d+", answer):
        return "Integer"
    if re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)", answer):
        return "Numeric"
    return "Expression"


def pull_math500() -> List[Dict[str, Any]]:
    """Downloads the MATH-500 test split (needs the `datasets` package and network access) and caches it.

    subject and level are kept so the sample's composition can be reported, and unique_id so a problem can be traced
    back to the full MATH test set. source_index is the row's position in the split, which is what every id downstream
    refers to.
    """
    from datasets import load_dataset  # imported here so nothing else needs the package

    rows = []
    for i, r in enumerate(load_dataset(MATH500_HF[0], split=MATH500_HF[2])):
        gold = str(r["answer"]).strip()
        rows.append({"source": "MATH-500 test", "source_index": i, "type": math500_type(gold),
                     "question": r["problem"], "gold": gold, "subject": r["subject"], "level": int(r["level"]),
                     "unique_id": r["unique_id"], "solution": r["solution"]})
    common.save_json(math500_source_path(), rows)
    return rows


def load_math500() -> List[Dict[str, Any]]:
    """All 500 problems. Nothing is dropped: every published answer is representable in one of the three types, and
    expression_answers.normalize is checked against all 500 golds by tests/test_expression_answers.py."""
    rows = common.load_json(math500_source_path()) or pull_math500()
    return [{"source": r["source"], "source_index": int(r["source_index"]), "type": r.get("type") or math500_type(r["gold"]),
             "question": r["question"], "gold": str(r["gold"]).strip(), "subject": r.get("subject"),
             "level": r.get("level")} for r in rows if str(r.get("gold", "")).strip()]


LOADERS = {"jeebench": load_jeebench, "gsm8k": load_gsm8k, "math500": load_math500}


def load_source(benchmark: str) -> List[Dict[str, Any]]:
    return LOADERS[benchmark]()


def main(argv: List[str] = None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(LOADERS)
    for name in names:
        rows = load_source(name)
        kinds = {}
        for r in rows:
            kinds[r["type"]] = kinds.get(r["type"], 0) + 1
        info = common.BENCHMARKS[name]
        print(f"{name}: {len(rows)} problems | types {kinds} | sampled: {info['n']} | {info['source']}")
        for field in ("subject", "level"):  # only MATH-500 carries these
            values = [r[field] for r in rows if r.get(field) is not None]
            if values:
                counts = dict(sorted(Counter(values).items(), key=lambda kv: (-kv[1], str(kv[0]))))
                print(f"  {field}s: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
