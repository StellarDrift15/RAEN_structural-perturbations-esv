import os
import sys
import json
import re
import time
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
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
    from prompts.prompt_functions import (
        generate_prompt_a, generate_prompt_solve, generate_prompt_strategy,
        A_FIELDS, SOLVE_FIELDS, STRATEGY_FIELDS,
    )
except ModuleNotFoundError:
    from math_prompt_formatter import create_inference_prompt
    from prompt_functions import (
        generate_prompt_a, generate_prompt_solve, generate_prompt_strategy,
        A_FIELDS, SOLVE_FIELDS, STRATEGY_FIELDS,
    )

# Fetch the API key from src/.env:
load_dotenv(os.path.join(SRC_DIR, ".env"))
GPT_KEY = os.environ.get('OPENROUTER_API_KEY')

# Pipeline (unchanged):
# Stage 1: Create the perturbation of the original problem using model A
# Stage 2: Verify the perturbation of the original problem using 2 other models
# Stage 3: Independently solve the new perturbed problem using 2 other models
# Stage 4: Freeze the (P, P') in the frozen dataset for all three methods: Direct, CoT, ESV
#
# Added safeguards (each one is a switch below):
#   - cheap deterministic checks on Model A's output before any verifier is called
#   - Models B/C first solve P' WITHOUT seeing P or the strategy, then apply the strategy in a second call
#   - "NONE" is a legal answer when no option is correct, and it is a rejection
#   - optional freeze-time audit: one stronger/reasoning model must independently reach the same answer

# ---- Run-time controls --------------------------------------------------------------------------
# FAST_MODE: low reasoning effort, smaller token caps, 2 attempts per problem, more parallel workers, shorter timeouts.
# TIME_BUDGET_SECONDS: hard wall-clock limit. When it is reached no new calls are started, everything that has
#   finished is saved to frozen_dataset.json, and the script exits (problems still in progress are dropped).
FAST_MODE = False
TIME_BUDGET_SECONDS = None  # 9 minutes; set to None for no limit
CALL_TIMEOUT = 100 if FAST_MODE else 200  # seconds per API call

STOP = threading.Event()  # set when the time budget is reached
DEADLINE = None


def out_of_time():
  return STOP.is_set() or (DEADLINE is not None and time.time() > DEADLINE)


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=GPT_KEY,
    timeout=CALL_TIMEOUT,
    max_retries=1,  # one quick SDK retry (covers 429/5xx blips); parse retries are handled by call_tagged
)

# Models Setup:

# One model per family, so B and C agreeing is meaningful (their mistakes are less correlated).
model_a = "deepseek/deepseek-v4-flash-0731"  # generator: needs reasoning to design a perturbation and recompute the options
model_b = "qwen/qwen3.5-35b-a3b"             # independent solver / strategy checker
model_c = "openai/gpt-oss-120b"              # independent solver / strategy checker (reasoning cannot be fully switched off)

CONFIDENCE_THRESHOLD = 0.6
NUM_PROBLEMS = 20
MAX_ATTEMPTS = 2  # perturbation attempts per original problem
REQUIRED_BREAK_VOTES = 2  # verifiers (of 2) whose application of the original strategy must visibly fail (was 1)
REQUIRE_A_AGREEMENT = False  # if False, only Models B and C (the independent solvers) must agree on answer_prime
NONE_COUNTS_AS_BROKEN = False  # naive_answer NONE ("S cannot be carried out") counts as a break vote only if True AND status is INVALID/INSUFFICIENT
REJECT_UNCHANGED_OPTIONS = True  # reject P' whose four options are identical to the original's
MAX_CALL_RETRIES = 2  # retries per model call on API/parse failure
MAX_TOKENS = 3500  # default output cap per call (includes reasoning tokens); per-model caps below override it
MAX_TOKENS_CEILING = 32000  # a response truncated by the cap is retried once with the cap doubled, up to this
MAX_WORKERS = 4  # original problems processed in parallel (lower this if you see 429 rate-limit errors)

# Freeze-time audit: a (preferably stronger) model solves P' alone with reasoning ON; its answer must equal answer_prime.
# The audit model is deliberately from a different family than the solvers B and C.
ENABLE_AUDIT = False
AUDIT_MODEL = model_a
AUDIT_REASONING = {"effort": "high"}
AUDIT_MAX_TOKENS = 24000
AUDIT_TIMEOUT = 400  # seconds

# Speed / quality knobs (passed to OpenRouter):
#   REASONING: {"enabled": False} = fastest. Try {"effort": "low"} (and raise MAX_TOKENS to ~6000)
#              if acceptance is low; reasoning tokens do not pollute the tagged answer.
#   PROVIDER_SORT: "throughput" routes to the fastest providers. Set None to use OpenRouter's default.
REASONING = {"enabled": False}  # fallback for any model not listed in MODEL_REASONING
PROVIDER_SORT = "throughput"

# Per-model settings. Reasoning tokens count toward max_tokens, so reasoning models need large caps.
# Effort levels are OpenRouter's normalized values; DeepSeek V4 Flash supports "high"/"xhigh" (or thinking off).
MODEL_REASONING = {
    model_a: {"effort": "high"},
    model_b: {"effort": "medium"},
    model_c: {"effort": "medium"},
}
MODEL_MAX_TOKENS = {
    model_a: 16000,
    model_b: 12000,
    model_c: 12000,
}

if FAST_MODE:  # trade some yield for speed; set FAST_MODE = False above for the slower, higher-yield settings
  # If OpenRouter rejects an effort value for a model, use {"enabled": False} for that model instead.
  MODEL_REASONING = {model_a: {"effort": "low"}, model_b: {"effort": "low"}, model_c: {"effort": "low"}}
  MODEL_MAX_TOKENS = {model_a: 6000, model_b: 6000, model_c: 6000}
  MAX_ATTEMPTS = 2
  MAX_WORKERS = 10
  MAX_CALL_RETRIES = 2


def call_model(prompt, model, label="", reasoning=None, max_tokens=None, timeout=None):
  extra_body = {"reasoning": reasoning if reasoning is not None else MODEL_REASONING.get(model, REASONING)}
  if PROVIDER_SORT:
    extra_body["provider"] = {"sort": PROVIDER_SORT}
  api = client.with_options(timeout=timeout) if timeout else client
  t0 = time.perf_counter()
  response = api.chat.completions.create(
      model=model,
      messages=[{"role": "user", "content": prompt}],
      max_tokens=max_tokens or MODEL_MAX_TOKENS.get(model, MAX_TOKENS),
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


def call_tagged(prompt, model, label, fields, accept, **call_kwargs):
  """Calls the model and returns a parsed dict of tagged fields.

  accept(parsed) -> bool decides whether the response has everything we need.
  Retries on API errors / missing tags. A response truncated by max_tokens (hidden reasoning can eat the whole
  budget and leave the visible answer empty) is retried once with the cap doubled; repeating the same cap would
  just truncate again.
  """
  suffix = ""
  length_failures = 0
  cap = call_kwargs.get("max_tokens") or MODEL_MAX_TOKENS.get(model, MAX_TOKENS)
  for attempt in range(1, MAX_CALL_RETRIES + 1):
    kwargs = {**call_kwargs, "max_tokens": cap}
    try:
      raw, finish_reason = call_model(prompt + suffix, model, label, **kwargs)
    except Exception as e:
      tqdm.write(f"  {label}: API error (try {attempt}/{MAX_CALL_RETRIES}): {e}")
      time.sleep(2 * attempt)
      continue
    parsed = parse_tagged(raw, fields)
    if parsed is not None and accept(parsed):
      return parsed
    tqdm.write(f"  {label}: missing/incomplete tags (try {attempt}/{MAX_CALL_RETRIES}, finish_reason={finish_reason}, max_tokens={cap}); "
               f"got tags {list(parsed) if parsed else []}; response starts: {raw[:120]!r}")
    if finish_reason == "length":
      length_failures += 1
      if length_failures >= 2:
        break
      cap = min(cap * 2, MAX_TOKENS_CEILING)
      suffix = "\n\nYour previous reply was cut off. Keep the derivation under 60 words, do not reconsider, and output ALL the tags."
  return None


def accept_a(d):
  status = d.get("status", "").upper()
  if "CANDIDATE" in status:
    needed = ["P_prime", "option_A", "option_B", "option_C", "option_D", "naive_answer", "answer_prime", "confidence_level"]
    return all(d.get(k) for k in needed)
  return "REJECT" in status


def accept_solve(d):
  return all(d.get(k) for k in ("valid_problem", "answer_prime", "confidence_level"))


def accept_strategy(d):
  return all(d.get(k) for k in ("strategy_status", "naive_answer"))


def to_confidence(value):
  try:
    text = str(value).strip()
    return float(text[:-1]) / 100 if text.endswith("%") else float(text)
  except (TypeError, ValueError):
    return 0.0


def is_true(value):
  return value is True or str(value).strip().lower() == "true"


def clean_choice(value):
  """Standardizes an answer to sorted capital option letters (e.g. 'a, c' -> 'AC'); None if it isn't letters A-D (incl. 'NONE')."""
  text = re.sub(r"[\s,;&.()\[\]{}*]|and", "", str(value))
  if not re.fullmatch(r"[A-Da-d]+", text):
    return None
  return "".join(sorted(set(text.upper())))


def standardize(solution):
  solution["answer_prime_raw"] = solution.get("answer_prime")
  solution["answer_prime"] = clean_choice(solution.get("answer_prime"))
  solution["naive_answer"] = clean_choice(solution.get("naive_answer"))


def strategy_broken(verifier):
  """True if applying the original strategy visibly fails on P'.

  naive_answer letters that differ from answer_prime = broken. naive_answer NONE means S could not be carried out:
  that is 'unknown' by default (a weak model may simply have failed), and only counts as broken when
  NONE_COUNTS_AS_BROKEN is set and the verifier's own status is INVALID/INSUFFICIENT.
  """
  naive, answer = verifier.get("naive_answer"), verifier.get("answer_prime")
  if naive is None:
    status = str(verifier.get("strategy_status", "")).strip().upper()
    return NONE_COUNTS_AS_BROKEN and status in ("INVALID", "INSUFFICIENT")
  return naive != answer


def build_feedback(reason, details):
  """Reason for rejection plus the verifiers' explanations, to guide Model A's next attempt."""
  notes = []
  for s in details[1:]:
    if isinstance(s, dict):
      notes += [str(s[k]) for k in ("reason", "strategy_reason") if s.get(k)]
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

  def fmt(text):
    t = str(text).strip()
    return t if "$" in t or "\\" not in t else f"${t}$"  # bare LaTeX like \frac{7\pi}{9} -> $\frac{7\pi}{9}$
  return stem + "\n\n" + "\n\n".join(f"({letter}) {fmt(options[letter])}" for letter in "ABCD")


def parse_options(text):
  """Returns {'A': 'text', ...} (whitespace/$ stripped, lowercased) from '(A) ... (B) ...' text, or None if not all four found."""
  options = {}
  for m in re.finditer(r"\(\s*([A-D])\s*\)\s*(.*?)(?=\n\s*\(\s*[A-D]\s*\)|\Z)", str(text), re.DOTALL):
    options[m.group(1)] = re.sub(r"[\s$]+", "", m.group(2)).lower()
  return options if len(options) == 4 else None


def options_unchanged(problem, perturbed_problem):
  """True if P' has exactly the same four option texts (in the same positions) as the original problem."""
  original, perturbed = parse_options(problem), parse_options(perturbed_problem)
  return bool(original and perturbed and all(original[l] == perturbed[l] for l in "ABCD"))


NOT_RUN = object()


def describe_models(a, b, c, audit=None):
  """Formats confidence, validity and answer_prime for models A, B, C (and the audit model)."""
  lines = []
  for name, s in (("A", a), ("B", b), ("C", c), ("Audit", audit)):
    if name == "Audit" and audit is None:
      continue
    if s is NOT_RUN:
      lines.append(f"    Model {name}: not run")
      continue
    if not s:
      lines.append(f"    Model {name}: no parsable response")
      continue
    validity = ""
    if name != "A":
      validity = f" | valid_problem={s.get('valid_problem')}"
      if name != "Audit":
        validity += f" | strategy_status={s.get('strategy_status')}"
    raw = s.get("answer_prime_raw")
    raw_note = f" (raw: {raw!r})" if raw is not None and raw != s.get("answer_prime") else ""
    naive = "" if name == "Audit" else f" | naive_answer={s.get('naive_answer')}"
    lines.append(f"    Model {name}: confidence={s.get('confidence_level')}{validity}{naive} | answer_prime={s.get('answer_prime')}{raw_note}")
  return "\n".join(lines)


def log_result(tag, verdict, details, audit=None):
  tqdm.write(f"[{tag}] {verdict}\n{describe_models(*details, audit=audit)}")


def run_attempt(problem, gold, feedback, tag):
  """Runs one perturbation attempt; returns (rejection_reason or None, (A, B, C) results, P', audit result)."""
  if out_of_time():
    return "time budget reached", (NOT_RUN, NOT_RUN, NOT_RUN), None, None
  # Stage 1: Create the perturbation
  a = call_tagged(generate_prompt_a(problem, gold, feedback), model_a, f"{tag} Model A", A_FIELDS, accept_a)
  if a:
    standardize(a)
    a["options"] = {letter: a.get(f"option_{letter}", "") for letter in "ABCD"}
  if not a or "CANDIDATE" not in str(a.get("status", "")).upper() or not a.get("P_prime"):
    return "no valid candidate from model A", (a, NOT_RUN, NOT_RUN), None, None
  perturbed_problem = assemble_problem(a)
  if perturbed_problem is None:
    return "P' is missing the (A)-(D) options", (a, NOT_RUN, NOT_RUN), None, None

  # Cheap deterministic checks before spending any verifier calls
  if a.get("naive_answer") is not None and a["naive_answer"] == a.get("answer_prime"):
    return "Model A's own naive_answer equals its answer_prime (the perturbation does not break the strategy)", (a, NOT_RUN, NOT_RUN), None, None
  if REJECT_UNCHANGED_OPTIONS and options_unchanged(problem, perturbed_problem):
    return "P' reuses the original four options unchanged; recompute the options for P'", (a, NOT_RUN, NOT_RUN), None, None

  if out_of_time():
    return "time budget reached", (a, NOT_RUN, NOT_RUN), None, None

  # Stage 2/3 (a): Models B and C independently solve P' alone (they never see P or the strategy)
  solve_prompt = generate_prompt_solve(perturbed_problem)
  with ThreadPoolExecutor(2) as ex:
    fb = ex.submit(call_tagged, solve_prompt, model_b, f"{tag} Model B", SOLVE_FIELDS, accept_solve)
    fc = ex.submit(call_tagged, solve_prompt, model_c, f"{tag} Model C", SOLVE_FIELDS, accept_solve)
    b, c = fb.result(), fc.result()
  for s in (b, c):
    if s:
      standardize(s)
  details = (a, b, c)
  if not b or not c:
    return "verifier response could not be parsed", details, None, None

  # Both solvers must deem P' well-posed, find a correct option, agree on it, and be confident
  if not all(is_true(v.get("valid_problem")) for v in (b, c)):
    return "a verifier judged P' ill-posed, ambiguous or without a correct option", details, None, None
  solvers = details if REQUIRE_A_AGREEMENT else (b, c)
  if any(s["answer_prime"] is None for s in solvers):
    return "a solver found no valid option letters for P' (P' likely has no correct option)", details, None, None
  if len({s["answer_prime"] for s in solvers}) != 1:
    return "answer_prime mismatch between solvers", details, None, None
  if not all(to_confidence(s.get("confidence_level")) >= CONFIDENCE_THRESHOLD for s in solvers):
    return "confidence below threshold", details, None, None

  if out_of_time():
    return "time budget reached", details, None, None

  # Stage 2/3 (b): Models B and C apply the original strategy to P' and compare with the solved answer
  with ThreadPoolExecutor(2) as ex:
    fsb = ex.submit(call_tagged, generate_prompt_strategy(problem, perturbed_problem, a.get("original_strategy"), a.get("changed_condition"), b["answer_prime"]),
                    model_b, f"{tag} Model B (strategy)", STRATEGY_FIELDS, accept_strategy)
    fsc = ex.submit(call_tagged, generate_prompt_strategy(problem, perturbed_problem, a.get("original_strategy"), a.get("changed_condition"), c["answer_prime"]),
                    model_c, f"{tag} Model C (strategy)", STRATEGY_FIELDS, accept_strategy)
    sb, sc = fsb.result(), fsc.result()
  if not sb or not sc:
    return "strategy-check response could not be parsed", details, None, None
  for v, sv in ((b, sb), (c, sc)):
    v["strategy_status"] = sv.get("strategy_status")
    v["strategy_reason"] = sv.get("strategy_reason")
    v["naive_answer_raw"] = sv.get("naive_answer")
    v["naive_answer"] = clean_choice(sv.get("naive_answer"))
  if sum(strategy_broken(v) for v in (b, c)) < REQUIRED_BREAK_VOTES:
    return "the original strategy still works on P' (naive_answer equals answer_prime) or could not be shown to fail", details, None, None

  # Freeze-time audit: a stronger model, reasoning on, solves P' alone and must land on the same answer
  audit = None
  if ENABLE_AUDIT:
    audit = call_tagged(solve_prompt, AUDIT_MODEL, f"{tag} Audit", SOLVE_FIELDS, accept_solve,
                        reasoning=AUDIT_REASONING, max_tokens=AUDIT_MAX_TOKENS, timeout=AUDIT_TIMEOUT)
    if audit:
      standardize(audit)
    if not audit:
      return "audit response could not be parsed", details, None, audit
    if not is_true(audit.get("valid_problem")) or audit["answer_prime"] is None:
      return "audit judged P' ill-posed or found no correct option among the four", details, None, audit
    if audit["answer_prime"] != b["answer_prime"]:
      return f"audit's independent solution ({audit['answer_prime']}) differs from answer_prime ({b['answer_prime']})", details, None, audit
  return None, details, perturbed_problem, audit


def process_problem(i, df):
  """Tries up to MAX_ATTEMPTS perturbations for problem i of df. Returns (i, row dict or None)."""
  problem = df['formatted_prompt'].iloc[i]
  gold = df['gold'].iloc[i]
  row = df.iloc[i]

  feedback = None
  for attempt in range(1, MAX_ATTEMPTS + 1):
    tag = f"{i} attempt {attempt}/{MAX_ATTEMPTS}"
    if out_of_time():
      return i, None
    try:
      reason, details, perturbed_problem, audit = run_attempt(problem, gold, feedback, tag)
    except Exception as e:  # never let one problem kill the whole run
      tqdm.write(f"[{tag}] unexpected error: {type(e).__name__}: {e}")
      reason, details, perturbed_problem, audit = f"unexpected error: {e}", (NOT_RUN, NOT_RUN, NOT_RUN), None, None
    if reason is None:
      a, b, c = details
      validation = {
          name: {
              "confidence": to_confidence(s.get("confidence_level")),
              "answer_prime": s["answer_prime"],
              "naive_answer": s["naive_answer"],
              **({"valid_problem": is_true(s.get("valid_problem")), "strategy_status": s.get("strategy_status")} if name != "model_a" else {}),
          }
          for name, s in (("model_a", a), ("model_b", b), ("model_c", c))
      }
      if ENABLE_AUDIT and audit:
        validation["audit"] = {
            "model": AUDIT_MODEL,
            "confidence": to_confidence(audit.get("confidence_level")),
            "answer_prime": audit["answer_prime"],
            "valid_problem": is_true(audit.get("valid_problem")),
        }
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
          "validation": validation,
      }
      log_result(tag, "accepted", details, audit)
      return i, result
    if reason == "time budget reached":
      return i, None
    log_result(tag, f"rejected: {reason}", details, audit)
    feedback = build_feedback(reason, details)
  return i, None


def save(results, total):
  """Writes accepted pairs (ordered by source problem) to frozen_dataset.json."""
  frozen_dataset = []
  for i in sorted(results):
    if results[i] is not None:
      frozen_dataset.append({"id": len(frozen_dataset), **results[i]})
  output_path = os.path.join(SRC_DIR, "frozen_dataset.json")
  models = {"a": model_a, "b": model_b, "c": model_c}
  if ENABLE_AUDIT:
    models["audit"] = AUDIT_MODEL
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "models": models,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "problems_processed": len(results),
            "problems_requested": total,
            "num_pairs": len(frozen_dataset),
            "pairs": frozen_dataset,
        },
        f, indent=2, ensure_ascii=False, default=str,
    )
  print(f"\nFrozen dataset size: {len(frozen_dataset)} (saved to {output_path})")


def main():
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

  # Core Loop Logic (problems run in parallel; results are saved on Ctrl+C or when the time budget runs out)
  global DEADLINE
  total = min(NUM_PROBLEMS, len(math_jeebench_df))
  results = {}
  start_time = time.perf_counter()
  DEADLINE = time.time() + TIME_BUDGET_SECONDS if TIME_BUDGET_SECONDS else None
  executor = ThreadPoolExecutor(MAX_WORKERS)
  futures = []
  timed_out = False
  try:
    futures = [executor.submit(process_problem, i, math_jeebench_df) for i in range(total)]
    progress = tqdm(as_completed(futures, timeout=(TIME_BUDGET_SECONDS or None)), total=total, desc="Original problems", unit="problem")
    for fut in progress:
      i, row = fut.result()
      results[i] = row
      progress.set_postfix(accepted=sum(r is not None for r in results.values()))
  except FuturesTimeout:
    timed_out = True
    STOP.set()
    print(f"\nTime budget ({TIME_BUDGET_SECONDS}s) reached: saving what has finished so far...")
    for f in futures:  # collect anything that finished while we were timing out
      if f.done() and not f.cancelled() and not f.exception():
        i, row = f.result()
        results.setdefault(i, row)
  except KeyboardInterrupt:
    print("\nInterrupted: saving what has finished so far...")
    STOP.set()
    for f in futures:
      f.cancel()
  finally:
    executor.shutdown(wait=False, cancel_futures=True)
    save(results, total)
    print(f"Total time: {time.perf_counter() - start_time:.0f}s")
    if timed_out:  # worker threads may still be inside an API call; exit without waiting for them
      sys.stdout.flush()
      os._exit(0)


if __name__ == "__main__":
  main()