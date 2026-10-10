import os
import sys
import json
import re
import pandas as pd
from datasets import load_dataset
from openai import OpenAI

# Make helpers/ and prompts/ importable regardless of the working directory
try:
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SRC_DIR = os.path.abspath("src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pilot_experiment.src.helpers.math_prompt_formatter import create_inference_prompt
from pilot_experiment.src.prompts.prompt_functions import generate_prompt_a, generate_prompt_b

# Fetch the GPT KEY from the .env file:
GPT_KEY = os.environ.get('OPENROUTER_API_KEY')

# Load the dataset
jeebench = load_dataset("json", data_files=os.path.join(SRC_DIR, "dataset.json"))

# Data Preprocessing and Filtering
jeebench_df = pd.DataFrame(jeebench["train"])

math_jeebench_df = jeebench_df[jeebench_df["subject"] == "math"].copy()
math_jeebench_df['formatted_prompt'] = math_jeebench_df.apply(create_inference_prompt, axis=1)
math_jeebench_df = math_jeebench_df.reset_index(drop=True)

print(math_jeebench_df[:200])


print("=== SAMPLE MODEL-READY PROMPT ===\n")
print(math_jeebench_df['formatted_prompt'].iloc[0])



# Stage 1: Create the perturbation of the original problem using model A
# Stage 2: Verify the perturbation of the original problem using 2 other models
# Stage 3: Independently solve the new perturbed problem using 2 other models
# Stage 4: Freeze the (P, P') in the frozen dataset for all three methods: Direct, CoT, ESV

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=GPT_KEY,
)

# Models Setup:

model_a = "google/gemini-2.5-flash"
model_b = "google/gemini-2.5-flash"
model_c = "google/gemini-2.5-flash"

CONFIDENCE_THRESHOLD = 0.7
NUM_PROBLEMS = 21


def call_model(prompt, model):
  response = client.chat.completions.create(
      model=model,
      messages=[{"role": "user", "content": prompt}],
  )
  return response.choices[0].message.content


def parse_json_response(text):
  """Parses a JSON object from a model response, tolerating ```json fences and surrounding text."""
  if not isinstance(text, str):
    return None
  text = text.strip()
  fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
  if fenced:
    text = fenced.group(1).strip()
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
      return None
    try:
      return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
      return None


def to_confidence(value):
  try:
    return float(value)
  except (TypeError, ValueError):
    return 0.0


def is_true(value):
  return value is True or str(value).strip().lower() == "true"


def normalize_answer(value):
  return re.sub(r"\s+", "", str(value)).lower()


# Perturbation Generator
def perturbation_generator(generate_prompt, model):
  try:
    return parse_json_response(call_model(generate_prompt, model))
  except Exception as e:
    print(f"Error generating perturbation: {e}")
    return None

# Perturbation Strategy Verifier
def perturbation_strategy_verifier(strategy_verifier_prompt, model):
  try:
    return parse_json_response(call_model(strategy_verifier_prompt, model))
  except Exception as e:
    print(f"Error verifying perturbation strategy: {e}")
    return None



# Frozen Dataset that stores pairs of (P, P')
frozen_dataset = []

# Core Loop Logic
for i in range(min(NUM_PROBLEMS, len(math_jeebench_df))):
  problem = math_jeebench_df['formatted_prompt'].iloc[i]
  gold = math_jeebench_df['gold'].iloc[i]

  # Stage 1: Create the perturbation
  perturbed_solution = perturbation_generator(generate_prompt_a(problem, gold), model_a)
  if not perturbed_solution or str(perturbed_solution.get("status", "")).upper() != "CANDIDATE" or not perturbed_solution.get("P_prime"):
    print(f"[{i}] rejected: no valid candidate from model A")
    continue
  perturbed_problem = perturbed_solution["P_prime"]

  # Stage 2/3: Verify the strategy and independently solve P' using Models B and C
  strategy_verifier_prompt = generate_prompt_b(problem, perturbed_problem)
  strategy_verifier_solution_b = perturbation_strategy_verifier(strategy_verifier_prompt, model_b)
  strategy_verifier_solution_c = perturbation_strategy_verifier(strategy_verifier_prompt, model_c)
  if not strategy_verifier_solution_b or not strategy_verifier_solution_c:
    print(f"[{i}] rejected: verifier response could not be parsed")
    continue

  verifiers = [strategy_verifier_solution_b, strategy_verifier_solution_c]

  # Both verifiers must deem P' well-posed and the original strategy broken
  if not all(is_true(v.get("valid_problem")) and str(v.get("strategy_status", "")).upper() in ("INVALID", "INSUFFICIENT") for v in verifiers):
    print(f"[{i}] rejected: verifiers did not confirm a valid strategy-breaking perturbation")
    continue

  # All three models must be confident and agree on answer_prime
  solutions = [perturbed_solution] + verifiers
  if not all(to_confidence(s.get("confidence_level")) > CONFIDENCE_THRESHOLD for s in solutions):
    print(f"[{i}] rejected: confidence below threshold")
    continue
  answers = {normalize_answer(s.get("answer_prime")) for s in solutions}
  if len(answers) != 1 or "" in answers:
    print(f"[{i}] rejected: answer_prime mismatch")
    continue

  # Stage 4: Include the pair (P, P') in the frozen dataset
  frozen_dataset.append((problem, perturbed_problem))
  print(f"[{i}] accepted")


# Display the frozen dataset
print(f"\nFrozen dataset size: {len(frozen_dataset)}")
for pair in frozen_dataset:
  print(pair)
