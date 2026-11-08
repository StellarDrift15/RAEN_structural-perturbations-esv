"""Prompt builders for the two-stage perturbation pipeline (adapted from pilot_experiment/src/prompts/prompt_functions.py).

Two perturbation modes, balanced roughly 50/50 by build_dataset.py:
  breaking    applying the original strategy S unchanged to P' gives a wrong answer or cannot be completed
  preserving  P' is a genuinely new instance (the answer changes and must be recomputed), but S still works unchanged

P' keeps the original's answer format (single-correct MCQ, multiple-correct MCQ, integer, numeric, or - on MATH-500 -
an exact LaTeX expression). The pilot turned every P' into a 4-option MCQ, which made original vs perturbed accuracy
not like-for-like.

Verification is split into two independent calls, as in the pilot:
  1. generate_prompt_solve    -> sees ONLY P' (no original problem, no strategy) -> avoids anchoring
  2. generate_prompt_strategy -> sees P, P', S and the solved answer -> mechanically applies S
"""

# What the <answer_prime> / <naive_answer> tag placeholders say, per answer format. The wording for the four older
# formats is unchanged, so JEEBench and GSM8K prompts stay byte-identical to the frozen runs.
ANSWER_TAG_HINT = {"single": "option letters", "multiple": "option letters", "integer": "number", "numeric": "number",
                   "expression": "LaTeX value"}
SOLVE_TARGET = {"single": "select the correct option(s)", "multiple": "select the correct option(s)",
                "integer": "give the final numerical answer", "numeric": "give the final numerical answer",
                "expression": "give the final exact answer"}

A_FIELDS = [
    "status", "P_prime", "option_A", "option_B", "option_C", "option_D",
    "original_strategy", "changed_condition", "failure_reason", "alternative_strategy",
    "derivation", "naive_answer", "answer_prime", "confidence_level",
]
SOLVE_FIELDS = ["valid_problem", "reason", "derivation", "answer_prime", "confidence_level"]
STRATEGY_FIELDS = ["strategy_status", "strategy_reason", "naive_answer"]

TAG_RULES = r"""
    Format rules (strict):
    - Output ONLY the tagged fields shown below, in the listed order, each with an opening and a closing tag. No JSON, no code fences, no text outside the tags. Write LaTeX normally with single backslashes (e.g. \frac{1}{2}); do not escape anything.
    - Keep every field to one or two sentences unless stated otherwise.
"""

SOLVE_FORMAT_RULES = r"""
    - In derivation (and P_prime), you may be longer, but solve step by step and check the result BEFORE committing to an answer, keeping derivation under 120 words: no digressions, re-checking or second-guessing, and never leave it unfinished.
    - confidence_level must be a plain number between 0 and 1 reflecting your actual certainty.
"""

COMMON_RULES = TAG_RULES + SOLVE_FORMAT_RULES

# What the answer of P' must look like, per answer format (same format as the original problem).
FORMAT_RULES_A = {
    "single": r"""
    Answer format of P' (same as the original): multiple choice with exactly four options A, B, C, D and EXACTLY ONE correct option. Put only the question stem in <P_prime> and each option's text in its own <option_X> tag; none may be empty.
    - answer_prime and naive_answer must each be a single capital letter (e.g. B). Never put a number or expression there.
    - The correct option can be any of A, B, C, D; never default to a particular letter. Recompute every option for P': the correct option must be the value you actually derived, and no other option may also be correct. Do NOT reuse the original four options unchanged.""",
    "multiple": r"""
    Answer format of P' (same as the original): multiple choice with exactly four options A, B, C, D, where one or more options are correct. Put only the question stem in <P_prime> and each option's text in its own <option_X> tag; none may be empty.
    - answer_prime and naive_answer must contain ONLY capital letters of options in alphabetical order, with no spaces or separators (e.g. B or ACD). Never put a number or expression there.
    - Recompute and re-check EVERY option for P'; the correct set must be exactly the options you verified. Do NOT reuse the original four options unchanged.""",
    "integer": r"""
    Answer format of P' (same as the original): the answer is a single non-negative integer, with NO options. Leave all four option tags empty.
    - answer_prime and naive_answer must each be just the integer (e.g. 7). Never put letters or words there. Choose the change so that the correct answer of P' is a non-negative integer.""",
    "numeric": r"""
    Answer format of P' (same as the original): the answer is a single number, with NO options. Leave all four option tags empty. If the answer is not an integer, the question must ask for it correct to two decimal places.
    - answer_prime and naive_answer must each be just the number (e.g. 2.25), rounded to two decimal places if not an integer. Never put letters or words there.""",
    "expression": r"""
    Answer format of P' (same as the original): the answer is a single exact mathematical value written in LaTeX, with NO options. Leave all four option tags empty.
    - answer_prime and naive_answer must each contain ONLY the value itself, in LaTeX with single backslashes: a number, fraction, radical, multiple of pi, coordinate tuple, interval, set, vector, polynomial or a single word. No words of explanation, no units, no "=" and no surrounding sentence.
    - Keep the SAME kind of answer as the original problem: a fraction stays an exact fraction, a radical stays an exact radical, a coordinate tuple stays a tuple of the same length, an interval stays an interval. Give the value in simplest exact form and never as a decimal approximation of an exact value (write \frac{14}{3}, not 4.67).
    - Choose the change so that the answer of P' is still a single, well-defined value of that kind.""",
}

SOLVE_RULES = {
    "single": r"""
    Additional rules:
    - answer_prime must be the single capital letter of the correct option (e.g. B), or the single word NONE.
    - If no option or more than one option is correct, or the question is ambiguous or ill-posed, set answer_prime to NONE and valid_problem to false. Never pick the closest option.""",
    "multiple": r"""
    Additional rules:
    - answer_prime must contain ONLY the capital letters of ALL correct options in alphabetical order, with no spaces or separators (e.g. B or ACD), or the single word NONE.
    - If no option is correct, or the question is ambiguous or ill-posed, set answer_prime to NONE and valid_problem to false.""",
    "integer": r"""
    Additional rules:
    - answer_prime must be just the integer answer (e.g. 7), or the single word NONE.
    - If the answer is not a well-defined non-negative integer, or the question is ambiguous or ill-posed, set answer_prime to NONE and valid_problem to false.""",
    "numeric": r"""
    Additional rules:
    - answer_prime must be just the number, rounded to two decimal places if not an integer (e.g. 2.25), or the single word NONE.
    - If the answer is not a single well-defined number, or the question is ambiguous or ill-posed, set answer_prime to NONE and valid_problem to false.""",
    "expression": r"""
    Additional rules:
    - answer_prime must be just the final value in LaTeX, in simplest exact form and nothing else (e.g. \frac{14}{3}, 3\sqrt{13}, \frac{\pi}{2}, (6,31,-1), (-\infty,2)\cup(3,\infty), x^2-4x+1), or the single word NONE. No units, no "=" and no explanation. Never give a decimal approximation of an exact value.
    - If the answer is not a single well-defined value, or the question is ambiguous or ill-posed, set answer_prime to NONE and valid_problem to false.""",
}
SOLVE_RULES_COMMON = """
    - valid_problem is false only if the question is ambiguous, ill-posed or has no correct answer of the required form."""

STRATEGY_ANSWER_RULE = {
    "single": "naive_answer must be the capital option letter S yields (e.g. B), or NONE if S cannot be carried out or yields a value that matches no option.",
    "multiple": "naive_answer must contain ONLY capital letters of the options S yields, in alphabetical order (e.g. B or ACD), or NONE if S cannot be carried out.",
    "integer": "naive_answer must be just the integer S yields, or NONE if S cannot be carried out.",
    "numeric": "naive_answer must be just the number S yields (two decimal places if not an integer), or NONE if S cannot be carried out.",
    "expression": r"naive_answer must be just the value S yields, in LaTeX in simplest exact form (e.g. \frac{14}{3}, 3\sqrt{13}, (6,31,-1)) and nothing else, or NONE if S cannot be carried out.",
}

BREAKING_TASK = """
    Create a perturbation P' of this math problem such that the original solution strategy no longer works and a different strategy is needed.

    Original problem: {problem}
    Reference answer: {answer}

    Steps:
    1. State the original strategy S as the key step(s) of the standard solution actually used to reach the reference answer (e.g. a substitution, a symmetry argument, a direct formula, a monotonicity or sign assumption).
    2. Change a structural condition so that applying S mechanically to P' gives a WRONG answer or cannot be completed. Useful levers: weaken or remove a domain/sign/monotonicity/continuity assumption so that cases must be split; change a quantifier or constraint so a symmetry no longer holds; add a degenerate or boundary case that S silently ignores; change the object so a standard formula no longer applies; reverse what is given and what is asked.
    3. Cosmetic changes (new constants, renamed variables, reordered options, extra data that does not change the method, excluding a point that does not matter) do NOT count.
    4. Write P' as a complete, unambiguous, solvable question, and solve it with a different strategy.
    5. naive_answer is the answer you get by applying S unchanged to P'. It MUST differ from answer_prime{naive_visible}.
    6. Before finalizing, check what your change does to the solution set or optimum. If a value you remove, add or restrict is itself a solution, the answer changes. If it does not affect the result at all, the change is cosmetic, so choose a different change.
    In changed_condition state the structural change; in failure_reason state exactly why S fails on P'; in alternative_strategy state the strategy that works on P'.
    If no valid perturbation exists, output status REJECT (you may leave the other tags empty).
"""

PRESERVING_TASK = """
    Create a perturbation P' of this math problem that is a genuinely NEW instance of the same problem type, such that the original solution strategy STILL WORKS unchanged on P'.

    Original problem: {problem}
    Reference answer: {answer}

    Steps:
    1. State the original strategy S as the key step(s) of the standard solution actually used to reach the reference answer (e.g. a substitution, a symmetry argument, a direct formula, a monotonicity or sign assumption).
    2. Change the problem's data or a non-structural condition (values, coefficients, the given function or curve within the same family, the quantity asked for) so that the answer of P' is DIFFERENT from the original answer and must be recomputed, while every condition S relies on still holds. Applying S step by step to P' must reach the correct answer of P'.
    3. Not allowed: changes that break S (removing a sign/domain/monotonicity assumption, adding a degenerate case, breaking a symmetry S uses); purely cosmetic changes (renaming variables, rewording, reordering options) that leave the answer unchanged.
    4. Write P' as a complete, unambiguous, solvable question, and solve it by applying S.
    5. naive_answer is the answer you get by applying S unchanged to P'. It MUST EQUAL answer_prime. The value of the answer must differ from the original reference answer{value_change_note}.
    In changed_condition state what was changed; in failure_reason write "none: S still applies" followed by why every condition S needs still holds; in alternative_strategy repeat S as applied to P'.
    If no valid perturbation exists, output status REJECT (you may leave the other tags empty).
"""


def generate_prompt_a(problem: str, answer: str, mode: str, answer_format: str, feedback: str = None) -> str:
    feedback_text = ""
    if feedback:
        feedback_text = ("\n    A previous attempt was rejected because: " + str(feedback) +
                         "\n    Produce a fundamentally different perturbation that fixes this.\n")
    choice = answer_format in ("single", "multiple")
    if mode == "breaking":
        task = BREAKING_TASK.format(problem=problem, answer=answer,
                                    naive_visible=", and one wrong option of P' should be the naive_answer so the failure is visible" if choice else "")
    else:
        task = PRESERVING_TASK.format(problem=problem, answer=answer,
                                      value_change_note=" (for multiple choice: the option values must be recomputed, not reused)" if choice else "")
    return f"""{task}{COMMON_RULES}{FORMAT_RULES_A[answer_format]}
{feedback_text}
    Output exactly this tagged format and nothing else:
    <status>CANDIDATE or REJECT</status>
    <P_prime>question stem only, without options</P_prime>
    <option_A>{'...' if choice else '(empty)'}</option_A>
    <option_B>{'...' if choice else '(empty)'}</option_B>
    <option_C>{'...' if choice else '(empty)'}</option_C>
    <option_D>{'...' if choice else '(empty)'}</option_D>
    <original_strategy>...</original_strategy>
    <changed_condition>...</changed_condition>
    <failure_reason>...</failure_reason>
    <alternative_strategy>...</alternative_strategy>
    <derivation>...</derivation>
    <naive_answer>{ANSWER_TAG_HINT[answer_format]}</naive_answer>
    <answer_prime>{ANSWER_TAG_HINT[answer_format]}</answer_prime>
    <confidence_level>number between 0 and 1</confidence_level>
    """


def generate_prompt_solve(problem_perturbed: str, answer_format: str) -> str:
    """Independent solve of P' only (Models B and C). Never sees P, the strategy or the mode."""
    return f"""
    Solve this math problem independently by any valid method and {SOLVE_TARGET[answer_format]}.

    Problem: {problem_perturbed}
    {COMMON_RULES}{SOLVE_RULES[answer_format]}{SOLVE_RULES_COMMON}
    Output exactly this tagged format and nothing else:
    <valid_problem>true or false</valid_problem>
    <reason>...</reason>
    <derivation>...</derivation>
    <answer_prime>{ANSWER_TAG_HINT[answer_format]} or NONE</answer_prime>
    <confidence_level>number between 0 and 1</confidence_level>
    """


def generate_prompt_strategy(problem: str, problem_perturbed: str, strategy: str, changed_condition: str,
                             answer_prime: str, answer_format: str) -> str:
    """Mechanically applies the claimed original strategy S to P' and compares with the solved answer.

    The prompt is identical for both modes, so the verifier is never told which outcome is expected.
    """
    return f"""
    Check whether an original solution strategy still works on a perturbed math problem.

    Original problem P: {problem}
    Perturbed problem P': {problem_perturbed}
    Claimed original strategy S: {strategy}
    Claimed changed condition: {changed_condition}
    Reference answer for P' (already established by an independent solver): {answer_prime}

    Do the following:
    1. Apply S to P' step by step exactly as it was used for P, without adapting it. Report the answer it yields as naive_answer, or NONE if S cannot be carried out.
    2. strategy_status: INVALID if a step of S fails, cannot be carried out, or S gives a result different from the reference answer; INSUFFICIENT if S works only partially and extra reasoning is needed; STILL_VALID if S applied unchanged reaches the reference answer; UNCERTAIN otherwise. Judge by actually applying S, not by whether some method might still work.
    {TAG_RULES}
    Additional rules:
    - {STRATEGY_ANSWER_RULE[answer_format]}
    - Report honestly what S yields when applied unchanged. Do not adapt S so that it reaches the reference answer, and do not avoid it either.

    Output exactly this tagged format and nothing else:
    <strategy_status>INVALID or INSUFFICIENT or STILL_VALID or UNCERTAIN</strategy_status>
    <strategy_reason>...</strategy_reason>
    <naive_answer>{ANSWER_TAG_HINT[answer_format]} or NONE</naive_answer>
    """
