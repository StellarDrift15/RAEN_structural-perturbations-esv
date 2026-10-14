"""Prompt builders for the perturbation pipeline.

Models answer in simple XML-style tags instead of JSON, so LaTeX can be written normally
(single backslashes). The same field lists are imported by main.py, so prompt and parser
can never drift apart.

Verification is split into two independent calls:
  1. generate_prompt_solve    -> sees ONLY P' (no original problem, no strategy) -> avoids anchoring
  2. generate_prompt_strategy -> sees P, P', S and the solved answer -> mechanically applies S
generate_prompt_solve is also used for the optional freeze-time audit by a stronger model.
"""

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

# Added for prompts that include a derivation and a confidence level (A prompt and the solve prompt).
SOLVE_FORMAT_RULES = r"""
    - In derivation (and P_prime), you may be longer, but solve step by step and check each option BEFORE committing to an answer, keeping derivation under 120 words: no digressions, re-checking or second-guessing, and never leave it unfinished.
    - confidence_level must be a plain number between 0 and 1 reflecting your actual certainty.
"""

COMMON_RULES = TAG_RULES + SOLVE_FORMAT_RULES

A_RULES = r"""
    Additional rules for P':
    - P' is a multiple-choice question in the style of the original with exactly four options A, B, C, D, even if the original problem is numeric or integer-type (then make the numeric answer one option and add plausible distractors). One or more options may be correct. Put only the question stem in <P_prime> and each option's text in its own <option_X> tag; none of the four options may be empty.
    - answer_prime and naive_answer must contain ONLY capital letters of options in alphabetical order, with no spaces, separators or other text (e.g. B or ACD). Never put a number or expression there.
    - The correct option can be any of A, B, C, D; never default to a particular letter.
    - Recompute every option for P'. The correct option(s) must be values you actually derived for P'. If your result matches no option, change the options until it does, or output status REJECT; never choose the closest option. Do NOT reuse the original four options unchanged.
    - Before finalizing, check what your change does to the solution set or optimum. If a value you remove, add or restrict is itself a solution, the answer changes, so update the options. If it does not affect the result at all, the change is cosmetic, so choose a different change.
"""

SOLVE_RULES = r"""
    Additional rules:
    - answer_prime must contain ONLY the capital letters of the correct options in alphabetical order, with no spaces or separators (e.g. B or ACD), or the single word NONE.
    - If your result matches none of the four options, or the question is ambiguous or ill-posed, set answer_prime to NONE and valid_problem to false. Never pick the closest option.
    - valid_problem is false only if the question is ambiguous, ill-posed or has no correct option.
"""

STRATEGY_RULES = r"""
    Additional rules:
    - naive_answer must contain ONLY capital letters of options in alphabetical order (e.g. B or ACD), or the single word NONE if S cannot be carried out or yields a value that matches no option.
    - Report honestly what S yields when applied unchanged. Do not adapt S so that it reaches the reference answer.
"""


def generate_prompt_a(problem, answer, feedback=None):
    feedback_text = ""
    if feedback:
        feedback_text = (
            "\n    A previous attempt was rejected because: " + str(feedback) +
            "\n    Produce a fundamentally different perturbation that fixes this.\n"
        )
    template = """
    Create a perturbation P' of this math problem such that the original solution strategy no longer works and a different strategy is needed.

    Original problem: {problem}
    Reference answer: {answer}

    Steps:
    1. State the original strategy S as the key step(s) of the standard solution actually used to reach the reference answer (e.g. a substitution, a symmetry argument, a direct formula, a monotonicity or sign assumption).
    2. Change a structural condition so that applying S mechanically to P' gives a WRONG answer or cannot be completed. Useful levers: weaken or remove a domain/sign/monotonicity/continuity assumption so that cases must be split; change a quantifier or constraint so a symmetry no longer holds; add a degenerate or boundary case that S silently ignores; change the object so a standard formula no longer applies; reverse what is given and what is asked.
    3. Cosmetic changes (new constants, renamed variables, reordered options, extra data that does not change the method, excluding a point that does not matter) do NOT count.
    4. Write P' as a complete, unambiguous, solvable question, and solve it with a different strategy. Recompute every option for P'; place the correct option(s) at varied positions.
    5. naive_answer is the option letter(s) you get by applying S unchanged to P'. It MUST differ from answer_prime, and one wrong option of P' should be the naive_answer so the failure is visible.
    If no valid perturbation exists, output status REJECT (you may leave the other tags empty).
    {common_rules}{a_rules}{feedback_text}
    Output exactly this tagged format and nothing else:
    <status>CANDIDATE or REJECT</status>
    <P_prime>question stem only, without options</P_prime>
    <option_A>...</option_A>
    <option_B>...</option_B>
    <option_C>...</option_C>
    <option_D>...</option_D>
    <original_strategy>...</original_strategy>
    <changed_condition>...</changed_condition>
    <failure_reason>...</failure_reason>
    <alternative_strategy>...</alternative_strategy>
    <derivation>...</derivation>
    <naive_answer>option letters</naive_answer>
    <answer_prime>option letters</answer_prime>
    <confidence_level>number between 0 and 1</confidence_level>
    """
    return template.format(problem=problem, answer=answer, common_rules=COMMON_RULES, a_rules=A_RULES, feedback_text=feedback_text)


def generate_prompt_solve(problem_perturbed):
    """Independent solve of P' only. Used by Models B and C, and by the optional audit model."""
    template = """
    Solve this multiple-choice math problem independently by any valid method and select the correct option(s).

    Problem: {P_prime}
    {common_rules}{solve_rules}
    Output exactly this tagged format and nothing else:
    <valid_problem>true or false</valid_problem>
    <reason>...</reason>
    <derivation>...</derivation>
    <answer_prime>option letters or NONE</answer_prime>
    <confidence_level>number between 0 and 1</confidence_level>
    """
    return template.format(P_prime=problem_perturbed, common_rules=COMMON_RULES, solve_rules=SOLVE_RULES)


def generate_prompt_strategy(problem, problem_perturbed, strategy, changed_condition, answer_prime):
    """Mechanically applies the claimed original strategy S to P' and compares with the solved answer."""
    template = """
    Check whether an original solution strategy still works on a perturbed math problem.

    Original problem P: {P}
    Perturbed problem P': {P_prime}
    Claimed original strategy S: {strategy}
    Claimed changed condition: {changed_condition}
    Reference answer for P' (already established by an independent solver): {answer_prime}

    Do the following:
    1. Apply S to P' step by step exactly as it was used for P, without adapting it. Report the option letter(s) it yields as naive_answer, or NONE if S cannot be carried out.
    2. strategy_status: INVALID if a step of S fails, cannot be carried out, or S gives a result different from the reference answer; INSUFFICIENT if S works only partially and extra reasoning is needed; STILL_VALID if S applied unchanged reaches the reference answer; UNCERTAIN otherwise. Judge by actually applying S, not by whether some method might still work.
    {common_rules}{strategy_rules}
    Output exactly this tagged format and nothing else:
    <strategy_status>INVALID or INSUFFICIENT or STILL_VALID or UNCERTAIN</strategy_status>
    <strategy_reason>...</strategy_reason>
    <naive_answer>option letters or NONE</naive_answer>
    """
    return template.format(
        P=problem, P_prime=problem_perturbed, strategy=strategy, changed_condition=changed_condition,
        answer_prime=answer_prime, common_rules=TAG_RULES, strategy_rules=STRATEGY_RULES,
    )