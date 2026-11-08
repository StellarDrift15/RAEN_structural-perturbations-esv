"""Shared plumbing for the experiments: benchmarks and paths, config, OpenRouter client, tagged-output calls, answer scoring.

Adapted from pilot_experiment/src/methods/esv.py so that every stage of this experiment (perturbation generation,
Direct QA, CoT, ESV) parses and scores answers with one implementation.

Differences from the pilot that matter for the results:
  - call_stage takes a temperature and a seed, so the n = 4 initial solves are genuinely different samples.
  - Scoring is type-aware: MCQ / MCQ(multiple) exact option set, Integer exact, Numeric within 0.01 (JEE Advanced
    numeric answers are graded to two decimal places).
"""

import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))  # every entry point does this too; repeated for safety
import expression_answers  # noqa: E402  the `expression` answer format (MATH-500); imports nothing from this module

# Windows consoles default to cp1252; never let a stray symbol in a model answer crash a run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_DIR = EXPERIMENT_DIR.parent
# Overridable so the offline smoke test never touches the real data/ and results/ folders.
DATA_ROOT = Path(os.getenv("ESV_DATA_DIR", EXPERIMENT_DIR / "data"))
RESULTS_ROOT = Path(os.getenv("ESV_RESULTS_DIR", EXPERIMENT_DIR / "results"))

# ---- Benchmarks ------------------------------------------------------------------------------------
# One folder per benchmark under data/ and results/ (sources: benchmarks.py). Every script takes --benchmark
# (default: ESV_BENCHMARK or jeebench) and calls select_benchmark() first, so paths are read through the module
# (common.DATA_DIR, common.RESULTS_DIR, common.FROZEN_DATASET), never imported by name: `from common import
# RESULTS_DIR` would freeze the default benchmark's path at import time.
# Per-benchmark settings:
#   n                  how many original problems build_dataset.py samples, and so which sample_<n>.json it writes.
#                      Per benchmark because MATH-500 is sampled at 250 while the two older benchmarks were frozen at
#                      200; changing a benchmark's n would orphan its sample file and invalidate its frozen pairs.
#   solver_max_tokens  the reply cap for the solvers and the ESV checker (default SOLVER_MAX_TOKENS). MATH-500 gets
#                      more room: its problems are competition-level, and a reply cut off by the cap costs a second,
#                      doubled-cap call (call_stage) that is paid for on top of the wasted one.
BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "jeebench": {"label": "JEEBench (math)", "subject": "math", "n": 200,
                 "source": "pilot_experiment/src/dataset.json (JEE Advanced 2016-2023, math subset)"},
    "gsm8k": {"label": "GSM8K", "subject": "grade-school math", "n": 200,
              "source": "openai/gsm8k on Hugging Face, config main, test split, cached in data/gsm8k/source.json"},
    "math500": {"label": "MATH-500", "subject": "competition math", "n": 250, "solver_max_tokens": 6000,
                "source": "HuggingFaceH4/MATH-500 on Hugging Face, test split, cached in data/math500/source.json"},
}
BENCHMARK = os.getenv("ESV_BENCHMARK", "jeebench")
DATA_DIR = RESULTS_DIR = FROZEN_DATASET = Path()


def select_benchmark(name: str) -> None:
    global BENCHMARK, DATA_DIR, RESULTS_DIR, FROZEN_DATASET
    if name not in BENCHMARKS:
        raise SystemExit(f"Unknown benchmark {name!r}; choose from {', '.join(BENCHMARKS)}")
    BENCHMARK = name
    DATA_DIR = DATA_ROOT / name
    RESULTS_DIR = RESULTS_ROOT / name
    FROZEN_DATASET = DATA_DIR / "frozen_dataset.json"


def benchmark_info() -> Dict[str, Any]:
    return BENCHMARKS[BENCHMARK]


def sample_size() -> int:
    """How many original problems this benchmark samples (build_dataset.py writes data/<benchmark>/sample_<n>.json)."""
    return int(BENCHMARKS[BENCHMARK].get("n", NUM_ORIGINAL_PROBLEMS))


def solver_max_tokens() -> int:
    """The reply cap for the solver samples and the ESV v3 check on this benchmark."""
    return int(BENCHMARKS[BENCHMARK].get("solver_max_tokens", SOLVER_MAX_TOKENS))


def add_benchmark_arg(parser) -> None:
    parser.add_argument("--benchmark", choices=list(BENCHMARKS), default=BENCHMARK,
                        help=f"which benchmark's data/ and results/ folders to use (default: {BENCHMARK})")


select_benchmark(BENCHMARK)

# ---- Experiment configuration ----------------------------------------------------------------------

NUM_ORIGINAL_PROBLEMS = 200  # the default; a benchmark overrides it with "n" in BENCHMARKS (see sample_size())
SAMPLE_SEED = 2026  # fixes the sampled problems and the tie-breaks of the perturbation-type balancer

# Perturbation generator and verifiers: one model per family, as in the pilot.
GENERATOR_MODEL = "deepseek/deepseek-v4-flash-0731"
VERIFIER_MODELS = ("qwen/qwen3.5-35b-a3b", "openai/gpt-oss-120b")

# Solvers of Experiments 1-3 (small models). Gemma 2 9B and Mistral 7B v0.3 are no longer served on OpenRouter
# (0 providers as of 2026-11-01), so Gemma 3 4B and Mistral Nemo 12B replace them.
SMALL_SOLVER_MODELS = (
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/mistral-nemo",
    "google/gemma-3-4b-it",
)
# Current solvers (default for every run): larger instruct models, none with hidden reasoning. Qwen and Mistral are
# the same families scaled up; Llama 3.3 70B replaces Gemma, whose 2 providers made runs slow (checked 2026-11-03).
SOLVER_MODELS = (
    "qwen/qwen-2.5-72b-instruct",
    "mistralai/mistral-small-3.2-24b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
)
ALL_SOLVER_MODELS = SMALL_SOLVER_MODELS + SOLVER_MODELS

# Source problems whose published gold is wrong or does not follow from the text (checked by hand, 2026-11-03). These
# instances are dropped from every method's score; the perturbed siblings keep their generator-verified gold.
#   gsm8k 40: 100 + 2*100 + 10 + 8 = 318 seats, gold 319 counts the host    82: "$4 each" is per pack of 5 -> profit 288, gold 160
#   gsm8k 95: half of the 20 left, split equally -> 5, gold 10               111: 70 * 2000 + 20000 = 160000, gold 162000
#   gsm8k 189: the question is inconsistent with its own givens (gold 8)     105: "21% per year" read as simple interest (gold 7400)
# math500: none found yet. MATH-500 golds are curated, but check this list again after the first full run - the
# place to look is the instances where every solver's 4 CoT samples agree on one answer that is scored wrong.
DISPUTED_INSTANCES: Dict[str, frozenset] = {
    "gsm8k": frozenset({"40_original", "82_original", "95_original", "111_original", "189_original", "105_original"}),
    "math500": frozenset(),
}


def is_disputed(instance_id: str) -> bool:
    """True when the instance's gold is disputed on the selected benchmark; such instances are never scored."""
    return instance_id in DISPUTED_INSTANCES.get(BENCHMARK, frozenset())
JUDGE_MODEL = "openai/gpt-oss-120b"  # grades ESV diagnoses only; never sees solver prompts it could influence

SOLVER_MAX_TOKENS = 4000  # default reply cap for the solvers and the ESV checker; see solver_max_tokens()
NUM_SAMPLES = 4           # initial solves per instance for DQA, CoT and ESV stage 0
SAMPLE_TEMPERATURE = 0.7  # > 0 so the 4 samples can differ; ESV stages and the retry control use temperature 0
NUMERIC_TOLERANCE = Decimal("0.01")

ENV_FILE_NAMES = (".env", "env")  # Windows Explorer can save the file without its leading dot


def find_env_file() -> Optional[Path]:
    """RAEN_ENV_FILE if set, else the nearest .env/env file: in the experiments folder itself, then each parent folder
    (e.g. C:\\...\\2026\\env)."""
    if os.getenv("RAEN_ENV_FILE"):
        return Path(os.environ["RAEN_ENV_FILE"])
    folders = (EXPERIMENT_DIR, *EXPERIMENT_DIR.parents)
    return next((folder / name for folder in folders for name in ENV_FILE_NAMES if (folder / name).is_file()), None)


ENV_FILE = find_env_file()
if ENV_FILE:
    load_dotenv(ENV_FILE)

# OpenRouter routing for every call: cheapest provider first (budget: the whole experiment should stay around $10),
# but never a 4-bit (fp4/int4) version of a model released at higher precision, since those can be noticeably worse
# at math. gpt-oss is released in 4-bit (MXFP4), so its native format is allowed. Cheaper providers are slower.
NO_4BIT_QUANTIZATIONS = ["int8", "fp6", "fp8", "fp16", "bf16", "fp32", "unknown"]
NATIVE_4BIT_MODELS = {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}


def provider_prefs(model: str) -> Dict[str, Any]:
    prefs: Dict[str, Any] = {"sort": "price"}
    if model not in NATIVE_4BIT_MODELS:
        prefs["quantizations"] = NO_4BIT_QUANTIZATIONS
    return prefs


MAX_RETRIES = int(os.getenv("JEE_MAX_RETRIES", "3"))
# 429s from a provider's shared pool clear in tens of seconds: back off longer, without using up MAX_RETRIES.
RATE_LIMIT_RETRIES = int(os.getenv("JEE_RATE_LIMIT_RETRIES", "5"))
REQUEST_TIMEOUT = int(os.getenv("JEE_TIMEOUT_SECONDS", "300"))


def make_client():
    """OpenRouter client. Imported lazily so --report-only and offline tests never need a key."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit(f"OPENROUTER_API_KEY is not set (.env used: {ENV_FILE or 'none found'}). "
                 "Set RAEN_ENV_FILE to the path of your .env, or place the .env in a parent folder of this repo.")
    from openai import OpenAI
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=0)


# ---- Small utilities -------------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def value_as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    # On Windows the swap fails while another process (editor, antivirus, indexer) briefly holds the target open.
    for attempt in range(20):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"Warning: could not read {path}.", file=sys.stderr)
        return default


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.split("/")[-1])


def reasoning_setting(value: str) -> Optional[Dict[str, Any]]:
    if value == "default":
        return None  # send nothing: the provider's default behaviour (required for non-reasoning models)
    if value == "off":
        return {"enabled": False}
    return {"effort": value}


def ratio(n: int, d: int) -> Dict[str, Any]:
    return {"n": n, "d": d, "rate": round(n / d, 4) if d else None}


def fmt(r: Optional[Dict[str, Any]]) -> str:
    if not r:
        return "-"
    return "n/a (0/0)" if r["rate"] is None else f"{100 * r['rate']:.1f}% ({r['n']}/{r['d']})"


def fmt_pp(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.1f} pp"


# ---- Tagged output parsing and model calls ---------------------------------------------------------
# Models answer in XML-style tags rather than JSON: LaTeX backslashes break JSON strings, and a cut-off tagged
# reply can still be partially recovered.

def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Fallback for models that answer in JSON despite the tag instructions."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = cleaned[start:end + 1]
    for attempt in (candidate, re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate)):  # 2nd try: escape raw LaTeX
        try:
            value = json.loads(attempt, strict=False)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def parse_tagged(text: str, fields: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Returns ({field: value}, [fields recovered from an unclosed tag]). Takes the last occurrence of each tag."""
    if not isinstance(text, str) or not text.strip():
        return {}, []
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    out: Dict[str, str] = {}
    for name in fields:
        matches = re.findall(rf"<{name}>(.*?)</{name}>", text, re.DOTALL | re.IGNORECASE)
        if matches and matches[-1].strip():
            out[name] = matches[-1].strip()
    partial = []
    # A reply cut off mid-field leaves the last tag unclosed; keep what was written.
    next_tag = re.compile(r"<(?:%s)>" % "|".join(fields), re.IGNORECASE)
    for name in fields:
        if name in out:
            continue
        openings = list(re.finditer(rf"<{name}>", text, re.IGNORECASE))
        if openings:
            tail = text[openings[-1].end():]
            following = next_tag.search(tail)
            value = (tail[:following.start()] if following else tail).strip()
            if value:
                out[name] = value
                partial.append(name)
    if not out:
        parsed_json = parse_json_object(text)
        if parsed_json:
            out = {name: value_as_text(parsed_json[name]) for name in fields if value_as_text(parsed_json.get(name))}
    return out, partial


TRUNCATION_NUDGE = "\n\nYour previous reply was cut off before all tags were written. Be much more concise this time and output ALL the tags."
FORMAT_NUDGE = "\n\nYour previous reply was missing required tags. Output ALL the tags listed above, each opened and closed."


def call_stage(
    client,
    model: str,
    stage: str,
    prompt: str,
    fields: List[str],
    required: List[str],
    reasoning: Optional[Dict[str, Any]] = None,
    max_tokens: int = 4000,
    temperature: Optional[float] = 0.0,
    seed: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Calls the model, retrying on API errors, missing tags and truncation. Token usage is summed over attempts.

    A reply truncated by max_tokens is retried once with the cap doubled; a reply missing tags gets a format nudge.
    temperature None sends no temperature (the provider default), as the pilot did for the generator and verifiers.
    """
    cap, suffix = max_tokens, ""
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    latency_total, errors = 0.0, []
    last: Optional[Dict[str, Any]] = None
    api = client.with_options(timeout=timeout) if timeout and hasattr(client, "with_options") else client
    attempt, rate_limited = 0, 0
    while attempt < MAX_RETRIES:
        attempt += 1
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt + suffix}],
            "max_tokens": cap,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed
        extra_body: Dict[str, Any] = {"provider": provider_prefs(model)}
        if reasoning is not None:
            extra_body["reasoning"] = reasoning
        kwargs["extra_body"] = extra_body
        started = time.perf_counter()
        try:
            response = api.chat.completions.create(**kwargs)
        except Exception as exc:  # API/provider errors are recorded, not silently swallowed.
            latency_total += time.perf_counter() - started
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if "401" in str(exc) or "expired" in str(exc).lower():
                break  # an invalid key will not fix itself on retry
            if ("429" in str(exc) or "rate-limited" in str(exc)) and rate_limited < RATE_LIMIT_RETRIES:
                rate_limited += 1
                attempt -= 1  # a rate limit is not the request's fault: it does not use up an attempt
                time.sleep(min(10 * 2 ** (rate_limited - 1), 90))
                continue
            time.sleep(min(3 * attempt, 10))
            continue
        latency_total += time.perf_counter() - started
        if not getattr(response, "choices", None):
            errors.append(f"attempt {attempt}: empty choices")
            continue
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = getattr(response, "usage", None)
        for key in usage_total:
            value = getattr(usage, key, None) if usage else None
            if isinstance(value, (int, float)):
                usage_total[key] += value
        parsed, partial = parse_tagged(content, fields)
        finish_reason = getattr(choice, "finish_reason", None)
        last = {"raw_response": content, "parsed": parsed, "partial_fields": partial, "finish_reason": finish_reason, "max_tokens": cap}
        if all(parsed.get(name) for name in required) and not any(name in partial for name in required):
            break
        if finish_reason == "length":
            cap = min(cap * 2, max_tokens * 2)
            suffix = TRUNCATION_NUDGE
        else:
            suffix = FORMAT_NUDGE
    if last is None:
        status = "api_error"
        last = {"raw_response": "", "parsed": {}, "partial_fields": [], "finish_reason": None, "max_tokens": cap}
    elif all(last["parsed"].get(name) for name in required):
        status = "completed" if not any(name in last["partial_fields"] for name in required) else "partial"
    else:
        status = "truncated" if last["finish_reason"] == "length" else "missing_fields"
    return {
        "stage": stage,
        "model": model,
        "status": status,
        **last,
        "temperature": temperature,
        "seed": seed,
        "attempts": attempt,
        "errors": errors,
        "usage": usage_total,
        "latency_seconds": round(latency_total, 3),
    }


def bool_value(value: Any) -> Optional[bool]:
    words = re.findall(r"[a-z]+", str(value or "").lower())
    if not words:
        return None
    if words[0] in {"true", "yes"}:
        return True
    if words[0] in {"false", "no"}:
        return False
    return None


def verdict_value(value: Any, labels: Tuple[str, ...]) -> Optional[str]:
    """First matching label; list longer labels first ('INCORRECT' before 'CORRECT')."""
    text = str(value or "").upper()
    for label in labels:
        if re.search(rf"\b{label}\b", text):
            return label
    return None


def to_confidence(value: Any) -> float:
    try:
        text = str(value).strip()
        return float(text[:-1]) / 100 if text.endswith("%") else float(text)
    except (TypeError, ValueError):
        return 0.0


# ---- Answer formats and scoring --------------------------------------------------------------------
# answer_format is derived from the question type (JEEBench's names; GSM8K problems are Integer; MATH-500 problems
# are Integer, Numeric or Expression) and is the same for a problem and its perturbation:
#   MCQ -> "single" (exactly one option), MCQ(multiple) -> "multiple", Integer -> "integer", Numeric -> "numeric",
#   Expression -> "expression" (a LaTeX value: a fraction, radical, tuple, interval, polynomial or word; scored by
#   expression_answers.py, the only place that knows how to compare those).

ANSWER_FORMATS = {"MCQ": "single", "MCQ(multiple)": "multiple", "Integer": "integer", "Numeric": "numeric",
                  "Expression": "expression"}


def answer_format_of(question_type: str) -> str:
    return ANSWER_FORMATS[question_type]


def is_choice_format(answer_format: str) -> bool:
    return answer_format in ("single", "multiple")


def answer_format_note(answer_format: str) -> str:
    if answer_format == "integer":
        return "Answer format: the answer is a single non-negative integer. final_answer must be just the integer."
    if answer_format == "numeric":
        return ("Answer format: the answer is a single number. If it is not an integer, give it correct to two decimal "
                "places. final_answer must be just the number.")
    if answer_format == "expression":
        return ("Answer format: the answer is a single exact mathematical value written in LaTeX - a number, "
                "fraction, radical, multiple of pi, coordinate tuple, interval, set, vector, polynomial or a single "
                "word. final_answer must be just that value in simplest exact form, with no words of explanation, no "
                "units and no '='. Keep it exact: write \\frac{14}{3} or 3\\sqrt{13}, never a decimal approximation.")
    if answer_format == "single":
        return "Answer format: multiple choice, exactly one option is correct. final_answer must be the option letter, e.g. B."
    return ("Answer format: multiple choice, one or more options may be correct. Select every correct option. "
            "final_answer must be the letter(s) of all correct options, e.g. B or ACD.")


def strip_markup(text: str) -> str:
    text = re.sub(r"\\(?:boxed|text|mathrm|textbf)\s*\{([^{}]*)\}", r"\1", text)
    return text.replace("$", "").replace("**", "").strip()


CHOICE_PREFIX = re.compile(r"^\s*(?:the\s+)?(?:final\s+)?(?:correct\s+)?(?:answer|options?|choices?)?\s*(?:is|are)?\s*[:=\-]?\s*", re.IGNORECASE)
CHOICE_LETTER = re.compile(r"\(?\s*([A-D])\s*\)?(?![A-Za-z0-9])")
CHOICE_SEPARATOR = re.compile(r"\s*(?:,|;|&|/|\band\b|\s)\s*(?:options?\s*)?", re.IGNORECASE)


def normalize_choice(value: Any) -> Optional[str]:
    """'C) -2 tan θ' -> 'C', 'B and C (h(0)=16)' -> 'BC', '(A), (B), (C)' -> 'ABC', 'AC' -> 'AC', 'NONE' -> None."""
    text = strip_markup(str(value or ""))
    compact = re.sub(r"[\s,;&()\[\]{}.*]+", "", text)
    if re.fullmatch(r"[A-Da-d]{1,4}", compact):
        return "".join(sorted(set(compact.upper())))
    # Read a leading run of option letters, stopping at the first non-letter content (an option's value).
    position = CHOICE_PREFIX.match(text).end()
    letters = []
    while True:
        match = CHOICE_LETTER.match(text, position)
        if not match:
            break
        letters.append(match.group(1))
        position = match.end()
        separator = CHOICE_SEPARATOR.match(text, position)
        if not separator:
            break
        position = separator.end()
    if letters:
        return "".join(sorted(set(letters)))
    markers = re.findall(r"(?:option\s*|\()([A-D])(?:\)|\b)", text, flags=re.IGNORECASE)
    return "".join(sorted(set(m.upper() for m in markers))) if markers else None


def normalize_number(value: Any) -> Optional[Decimal]:
    text = strip_markup(str(value or "")).replace(",", "")
    text = re.sub(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    if "=" in text:
        text = text.rsplit("=", 1)[1]
    fraction = re.search(r"\(?\s*([-+]?\d+(?:\.\d+)?)\s*\)?\s*/\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?", text)
    if fraction:
        denominator = Decimal(fraction.group(2))
        return Decimal(fraction.group(1)) / denominator if denominator else None
    number = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not number:
        return None
    try:
        return Decimal(number.group(0))
    except InvalidOperation:
        return None


def normalize_answer(value: Any, answer_format: str) -> Optional[str]:
    """Canonical string form of an answer (option letters, the number, or the LaTeX value), or None if unreadable."""
    if answer_format == "expression":
        return expression_answers.normalize(value)
    if is_choice_format(answer_format):
        letters = normalize_choice(value)
        if letters and answer_format == "single" and len(letters) != 1:
            return letters  # kept so a multi-letter reply to a single-answer MCQ is scored wrong, not unreadable
        return letters
    number = normalize_number(value)
    return None if number is None else format(number.normalize(), "f")


def answers_match(a: Optional[str], b: Optional[str], answer_format: str) -> bool:
    if a is None or b is None:
        return False
    if answer_format == "expression":
        return expression_answers.match(a, b)
    if is_choice_format(answer_format):
        return a == b
    tolerance = NUMERIC_TOLERANCE if answer_format == "numeric" else Decimal("1e-6")
    try:
        return abs(Decimal(a) - Decimal(b)) <= tolerance + Decimal("1e-9")
    except InvalidOperation:
        return False


def score_answer(prediction: Any, gold: Any, answer_format: str) -> Dict[str, Any]:
    """An empty or unreadable answer is scored as incorrect (a failure to answer is a failure), and flagged."""
    predicted, expected = normalize_answer(prediction, answer_format), normalize_answer(gold, answer_format)
    return {"normalized_answer": predicted, "normalized_gold": expected,
            "correct": answers_match(predicted, expected, answer_format), "answer_readable": predicted is not None}


def extract_answer(call: Dict[str, Any]) -> str:
    answer = (call.get("parsed") or {}).get("final_answer", "")
    if answer:
        return answer
    # Last resort for replies that ignore the tag format: an explicit "final answer: ..." line only.
    matches = re.findall(r"final\s+answer\s*(?:is)?\s*[:=]\s*(.+)", call.get("raw_response") or "", flags=re.IGNORECASE)
    return matches[-1].strip() if matches else ""


def majority_answer(normalized: List[Optional[str]], answer_format: str) -> Optional[str]:
    """Most frequent readable answer; numeric answers within tolerance are grouped; ties go to the earliest sample."""
    groups: List[List[Any]] = []  # [representative, count, first_index]
    for index, value in enumerate(normalized):
        if value is None:
            continue
        for group in groups:
            if answers_match(group[0], value, answer_format):
                group[1] += 1
                break
        else:
            groups.append([value, 1, index])
    if not groups:
        return None
    return max(groups, key=lambda g: (g[1], -g[2]))[0]


def modal_index(normalized: List[Optional[str]], answer_format: str) -> int:
    """Index of the first sample giving the majority answer (sample 0 if every answer is unreadable)."""
    top = majority_answer(normalized, answer_format)
    if top is None:
        return 0
    return next(i for i, v in enumerate(normalized) if answers_match(v, top, answer_format))


def answer_counts(normalized: List[Optional[str]]) -> Dict[str, int]:
    return dict(Counter(v if v is not None else "UNREADABLE" for v in normalized))


# ---- Statistics and final-answer scoring -----------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> Optional[float]:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return None
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def final_fields(gold: str, answer_format: str, initial: Dict[str, Any], final_answer: Optional[str]) -> Dict[str, Any]:
    """Scores a final answer against gold, only here. An empty final answer keeps the initial one."""
    final = final_answer if final_answer else initial["answer"]
    score = score_answer(final, gold, answer_format)
    return {"final_answer": final, "final_normalized": score["normalized_answer"], "final_correct": score["correct"],
            "changed": not answers_match(score["normalized_answer"], initial["normalized"], answer_format)}
