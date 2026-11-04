#!/usr/bin/env python3
"""Source problems per benchmark, all in one shape: {source, source_index, type, question, gold}.

  jeebench  the pilot's JEEBench file, math subset: MCQ, MCQ(multiple), Integer and Numeric questions.
  gsm8k     openai/gsm8k (config main, test split): 1319 grade-school word problems with integer answers. Pulled once
            from Hugging Face into data/gsm8k/source.json and read from there afterwards. The two problems with a
            negative answer are left out: the integer answer format, and the perturbation rules built on it, are
            non-negative.

build_dataset.py samples NUM_ORIGINAL_PROBLEMS of these (seed SAMPLE_SEED) and perturbs them; nothing downstream
reads the source again.

  python experiments/benchmarks.py gsm8k      # pull (if needed) and summarize a benchmark's source
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

JEEBENCH_SOURCE = common.REPO_DIR / "pilot_experiment" / "src" / "dataset.json"
GSM8K_HF = ("openai/gsm8k", "main", "test")


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


LOADERS = {"jeebench": load_jeebench, "gsm8k": load_gsm8k}


def load_source(benchmark: str) -> List[Dict[str, Any]]:
    return LOADERS[benchmark]()


def main(argv: List[str] = None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(LOADERS)
    for name in names:
        rows = load_source(name)
        kinds = {}
        for r in rows:
            kinds[r["type"]] = kinds.get(r["type"], 0) + 1
        print(f"{name}: {len(rows)} problems | types {kinds} | {common.BENCHMARKS[name]['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
