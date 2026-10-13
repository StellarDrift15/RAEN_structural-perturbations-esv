import os
import sys
import json
import re
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Make helpers/ and prompts/ importable regardless of the working directory
try:
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SRC_DIR = os.path.abspath("src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Works with src/helpers + src/prompts subfolders, or with all files flat in src/
try:
    from helpers.math_prompt_formatter import create_inference_prompt
    from prompts.prompt_functions import generate_prompt_a, generate_prompt_b, A_FIELDS, B_FIELDS
except ModuleNotFoundError:
    from math_prompt_formatter import create_inference_prompt
    from prompt_functions import generate_prompt_a, generate_prompt_b, A_FIELDS, B_FIELDS

# Fetch the API key from src/.env:
load_dotenv(os.path.join(SRC_DIR, ".env"))
GPT_KEY = os.environ.get('OPENROUTER_API_KEY')

# Load the dataset
jeebench = load_dataset("json", data_files=os.path.join(SRC_DIR, "dataset.json"))

# Data Preprocessing and Filtering
jeebench_df = pd.DataFrame(jeebench["train"])

math_jeebench_df = jeebench_df[jeebench_df["subject"] == "math"].copy()
math_jeebench_df['formatted_prompt'] = math_jeebench_df.apply(create_inference_prompt, axis=1)
math_jeebench_df = math_jeebench_df.reset_index(drop=True)

print(f"Math problems loaded: {len(math_jeebench_df)}")
print("=== SAMPLE MODEL-READY PROMPT ===\n")
print(math_jeebench_df['formatted_prompt'].iloc[0])


# Stage 1: Create the perturbation of the original problem using model A
# Stage 2: Verify the perturbation of the original problem using 2 other models
# Stage 3: Independently solve the new perturbed problem using 2 other models
# Stage 4: Freeze the (P, P') in the frozen dataset for all three methods: Direct, CoT, ESV

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=GPT_KEY,
    timeout=120,
    max_retries=1,  # one quick SDK retry (covers 429/5xx blips); parse retries are handled by call_tagged
)

# Models Setup:

model_a = "qwen/qwen3-30b-a3b-instruct-2507"
model_b = "deepseek/deepseek-v3.2"
model_c = "qwen/qwen3.5-9b"

CONFIDENCE_THRESHOLD = 0.6
NUM_PROBLEMS = 20
MAX_ATTEMPTS = 3  # perturbation attempts per original problem
REQUIRED_BREAK_VOTES = 1  # verifiers (of 2) that must find the original strategy broken
REQUIRE_A_AGREEMENT = False  # if False, only Models B and C (the independent solvers) must agree on answer_prime
MAX_CALL_RETRIES = 3  # retries per model call on API/parse failure
MAX_TOKENS = 3500  # output cap per call, includes reasoning tokens
MAX_WORKERS = 4  # original problems processed in parallel (lower this if you see 429 rate-limit errors)

REASONING = {"enabled": False}
PROVIDER_SORT = "throughput"


def call_model(prompt, model, label=""):
  extra_body = {"reasoning": REASONING}
  if PROVIDER_SORT:
    extra_body["provider"] = {"sort": PROVIDER_SORT}
  t0 = time.perf_counter()
  response = client.chat.completions.create(
      model=model,
      messages=[{"role": "user", "content": prompt}],
      max_tokens=MAX_TOKENS,
      extra_body=extra_body,
  )
  choice = response.choices[0]
  tokens = getattr(response.usage, "completion_tokens", "?") if response.usage else "?"
  tqdm.write(f"    [{label or model}] {time.perf_counter() - t0:.0f}s, {tokens} tok, finish={choice.finish_reason}")
  return choice.message.content or "", choice.finish_reason


def parse_tagged(text, fields):
  """Extracts <field>...</field> values from a model response. Takes the last occurrence of each tag."""
  if not isinstance(text, str):
    return None
  text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
  out = {}
  for f in fields:
    matches = re.findall(rf"<{f}>(.*?)</{f}>", text, re.DOTALL | re.IGNORECASE)
    if matches:
      out[f] = matches[-1].strip()
  return out or None


def call_tagged(prompt, model, label, fields, accept):
  """Calls the model and returns a parsed dict of tagged fields.

  accept(parsed) -> bool decides whether the response has everything we need.
  Retries on API errors / missing tags; a response truncated by max_tokens is retried at most once
  with a 'be brief' instruction (repeating the same prompt would just truncate again).
  """
  suffix = ""
  length_failures = 0
  for attempt in range(1, MAX_CALL_RETRIES + 1):
    try:
      raw, finish_reason = call_model(prompt + suffix, model, label)
    except Exception as e:
      tqdm.write(f"  {label}: API error (try {attempt}/{MAX_CALL_RETRIES}): {e}")
      time.sleep(2 * attempt)
      continue
    parsed = parse_tagged(raw, fields)
    if parsed is not None and accept(parsed):
      return parsed
    tqdm.write(f"  {label}: missing/incomplete tags (try {attempt}/{MAX_CALL_RETRIES}, finish_reason={finish_reason}); "
               f"got tags {list(parsed) if parsed else []}; response starts: {raw[:120]!r}")
    if finish_reason == "length":
      length_failures += 1
      if length_failures >= 2:
        break
      suffix = "\n\nYour previous reply was cut off. Keep the derivation under 60 words, do not reconsider, and output ALL the tags."
  return None


def accept_a(d):
  status = d.get("status", "").upper()
  if "CANDIDATE" in status:
    needed = ["P_prime", "option_A", "option_B", "option_C", "option_D", "naive_answer", "answer_prime", "confidence_level"]
    return all(d.get(k) for k in needed)
  return "REJECT" in status


def accept_b(d):
  return all(d.get(k) for k in ("valid_problem", "naive_answer", "answer_prime", "confidence_level"))


def to_confidence(value):
  try:
    text = str(value).strip()
    return float(text[:-1]) / 100 if text.endswith("%") else float(text)
  except (TypeError, ValueError):
    return 0.0


def is_true(value):
  return value is True or str(value).strip().lower() == "true"


def clean_choice(value):
  """Standardizes an answer to sorted capital option letters (e.g. 'a, c' -> 'AC'); None if it isn't letters A-D."""
  text = re.sub(r"[\s,;&.()\[\]{}*]|and", "", str(value))
  if not re.fullmatch(r"[A-Da-d]+", text):
    return None
  return "".join(sorted(set(text.upper())))


def standardize(solution):
  solution["answer_prime_raw"] = solution.get("answer_prime")
  solution["answer_prime"] = clean_choice(solution.get("answer_prime"))
  solution["naive_answer"] = clean_choice(solution.get("naive_answer"))


def strategy_broken(verifier):
  """True if the verifier's status says so, or applying the original strategy gives a different answer."""
  status_broken = str(verifier.get("strategy_status", "")).strip().upper() in ("INVALID", "INSUFFICIENT")
  return status_broken or verifier.get("naive_answer") != verifier.get("answer_prime")


def build_feedback(reason, details):
  """Reason for rejection plus the verifiers' explanations, to guide Model A's next attempt."""
  notes = [str(s.get("reason")) for s in details[1:] if isinstance(s, dict) and s.get("reason")]
  return reason + (". Verifier notes: " + " | ".join(notes) if notes else "")


def has_options(problem_text):
  return all(re.search(rf"\(\s*{letter}\s*\)|(?:^|\n)\s*{letter}\s*[.)]", problem_text, re.IGNORECASE) for letter in "ABCD")


def assemble_problem(candidate):
  """Returns the P' stem with its (A)-(D) options appended in a standard format, or None if options are missing."""
  stem = str(candidate["P_prime"]).strip()
  if has_options(stem):
    return stem
  options = candidate.get("options")
  if not isinstance(options, dict) or not all(str(options.get(letter, "")).strip() for letter in "ABCD"):
    return None
  return stem + "\n\n" + "\n\n".join(f"({letter}) {str(options[letter]).strip()}" for letter in "ABCD")


NOT_RUN = object()


def describe_models(a, b, c):
  """Formats confidence, validity and answer_prime for models A, B and C."""
  lines = []
  for name, s in (("A", a), ("B", b), ("C", c)):
    if s is NOT_RUN:
      lines.append(f"    Model {name}: not run")
      continue
    if not s:
      lines.append(f"    Model {name}: no parsable response")
      continue
    validity = ""
    if name != "A":
      validity = f" | valid_problem={s.get('valid_problem')} | strategy_status={s.get('strategy_status')}"
    raw = s.get("answer_prime_raw")
    raw_note = f" (raw: {raw!r})" if raw is not None and raw != s.get("answer_prime") else ""
    lines.append(f"    Model {name}: confidence={s.get('confidence_level')}{validity} | naive_answer={s.get('naive_answer')} | answer_prime={s.get('answer_prime')}{raw_note}")
  return "\n".join(lines)


def log_result(tag, verdict, details):
  tqdm.write(f"[{tag}] {verdict}\n{describe_models(*details)}")


def run_attempt(problem, gold, feedback, tag):
  """Runs one perturbation attempt; returns (rejection_reason or None, (A, B, C) results, P')."""
  # Stage 1: Create the perturbation
  a = call_tagged(generate_prompt_a(problem, gold, feedback), model_a, f"{tag} Model A", A_FIELDS, accept_a)
  if a:
    standardize(a)
    a["options"] = {letter: a.get(f"option_{letter}", "") for letter in "ABCD"}
  if not a or "CANDIDATE" not in str(a.get("status", "")).upper() or not a.get("P_prime"):
    return "no valid candidate from model A", (a, NOT_RUN, NOT_RUN), None
  perturbed_problem = assemble_problem(a)
  if perturbed_problem is None:
    return "P' is missing the (A)-(D) options", (a, NOT_RUN, NOT_RUN), None

  # Stage 2/3: Verify the strategy and independently solve P' using Models B and C (in parallel)
  verifier_prompt = generate_prompt_b(problem, perturbed_problem, a.get("original_strategy"), a.get("changed_condition"))
  with ThreadPoolExecutor(2) as ex:
    fb = ex.submit(call_tagged, verifier_prompt, model_b, f"{tag} Model B", B_FIELDS, accept_b)
    fc = ex.submit(call_tagged, verifier_prompt, model_c, f"{tag} Model C", B_FIELDS, accept_b)
    b, c = fb.result(), fc.result()
  for s in (b, c):
    if s:
      standardize(s)
  details = (a, b, c)
  if not b or not c:
    return "verifier response could not be parsed", details, None

  # Both verifiers must deem P' well-posed, and enough of them must find the original strategy broken
  if not all(is_true(v.get("valid_problem")) for v in (b, c)):
    return "a verifier judged P' ill-posed or ambiguous", details, None
  solvers = details if REQUIRE_A_AGREEMENT else (b, c)
  if any(s["answer_prime"] is None for s in solvers):
    return "a solver found no valid option letters for P' (P' likely has no correct option)", details, None
  if sum(strategy_broken(v) for v in (b, c)) < REQUIRED_BREAK_VOTES:
    return "the original strategy still works on P' (naive_answer equals answer_prime)", details, None

  # Independent solvers must agree on the letter answer_prime, and be confident
  if len({s["answer_prime"] for s in solvers}) != 1:
    return "answer_prime mismatch between solvers", details, None
  if not all(to_confidence(s.get("confidence_level")) >= CONFIDENCE_THRESHOLD for s in solvers):
    return "confidence below threshold", details, None
  return None, details, perturbed_problem


def process_problem(i):
  """Tries up to MAX_ATTEMPTS perturbations for problem i. Returns (i, row dict or None)."""
  problem = math_jeebench_df['formatted_prompt'].iloc[i]
  gold = math_jeebench_df['gold'].iloc[i]
  row = math_jeebench_df.iloc[i]

  feedback = None
  for attempt in range(1, MAX_ATTEMPTS + 1):
    tag = f"{i} attempt {attempt}/{MAX_ATTEMPTS}"
    try:
      reason, details, perturbed_problem = run_attempt(problem, gold, feedback, tag)
    except Exception as e:  # never let one problem kill the whole run
      tqdm.write(f"[{tag}] unexpected error: {type(e).__name__}: {e}")
      reason, details, perturbed_problem = f"unexpected error: {e}", (NOT_RUN, NOT_RUN, NOT_RUN), None
    if reason is None:
      a, b, c = details
      result = {
          "source": row["description"],
          "source_index": int(row["index"]),
          "type": row["type"],
          "original_problem": problem,
          "original_gold": gold,
          "perturbed_problem": perturbed_problem,
          "answer_prime": b["answer_prime"],
          "attempts_used": attempt,
          "original_strategy": a.get("original_strategy"),
          "changed_condition": a.get("changed_condition"),
          "alternative_strategy": a.get("alternative_strategy"),
          "validation": {
              name: {
                  "confidence": to_confidence(s.get("confidence_level")),
                  "answer_prime": s["answer_prime"],
                  "naive_answer": s["naive_answer"],
                  **({"valid_problem": is_true(s.get("valid_problem")), "strategy_status": s.get("strategy_status")} if name != "model_a" else {}),
              }
              for name, s in (("model_a", a), ("model_b", b), ("model_c", c))
          },
      }
      log_result(tag, "accepted", details)
      return i, result
    log_result(tag, f"rejected: {reason}", details)
    feedback = build_feedback(reason, details)
  return i, None


def save(results, total):
  """Writes accepted pairs (ordered by source problem) to frozen_dataset.json."""
  frozen_dataset = []
  for i in sorted(results):
    if results[i] is not None:
      frozen_dataset.append({"id": len(frozen_dataset), **results[i]})
  output_path = os.path.join(SRC_DIR, "frozen_dataset.json")
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "models": {"a": model_a, "b": model_b, "c": model_c},
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "problems_processed": len(results),
            "problems_requested": total,
            "num_pairs": len(frozen_dataset),
            "pairs": frozen_dataset,
        },
        f, indent=2, ensure_ascii=False, default=str,
    )
  print(f"\nFrozen dataset size: {len(frozen_dataset)} (saved to {output_path})")


# Core Loop Logic (problems run in parallel; results are saved even if you Ctrl+C)
total = min(NUM_PROBLEMS, len(math_jeebench_df))
results = {}
start_time = time.perf_counter()
executor = ThreadPoolExecutor(MAX_WORKERS)
try:
  futures = [executor.submit(process_problem, i) for i in range(total)]
  progress = tqdm(as_completed(futures), total=total, desc="Original problems", unit="problem")
  for fut in progress:
    i, row = fut.result()
    results[i] = row
    progress.set_postfix(accepted=sum(r is not None for r in results.values()))
except KeyboardInterrupt:
  print("\nInterrupted: saving what has finished so far...")
  for f in futures:
    f.cancel()
finally:
  executor.shutdown(wait=False, cancel_futures=True)
  save(results, total)
  print(f"Total time: {time.perf_counter() - start_time:.0f}s")