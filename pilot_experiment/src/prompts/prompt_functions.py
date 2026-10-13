"""Prompt builders for the perturbation pipeline.

Models now answer in simple XML-style tags instead of JSON. LaTeX can be written
normally (single backslashes), which removes the escaping/parsing failures and the
whitespace loops that JSON mode caused.
"""

# Field lists are shared with main.py so the prompt and the parser can never drift apart.
A_FIELDS = [
    "status", "P_prime", "option_A", "option_B", "option_C", "option_D",
    "original_strategy", "changed_condition", "failure_reason", "alternative_strategy",
    "derivation", "naive_answer", "answer_prime", "confidence_level",
]
B_FIELDS = [
    "valid_problem", "strategy_status", "reason", "derivation",
    "naive_answer", "answer_prime", "confidence_level",
]

FORMAT_RULES = r"""
    Format rules (strict):
    - P' is a multiple-choice question in the style of the original with exactly four options A, B, C, D, even if the original problem is numeric or integer-type (then make the numeric answer one option and add plausible distractors). One or more options may be correct. Put only the question stem in <P_prime> and each option's text in its own <option_X> tag; none of the four options may be empty.
    - answer_prime must contain ONLY the capital letters of the correct options in alphabetical order, with no spaces, separators or other text (e.g. B or ACD). Never put a number or expression in answer_prime.
    - confidence_level must be a plain number between 0 and 1 reflecting your actual certainty.
    - The correct option can be any of A, B, C, D; never default to a particular letter, and do not copy placeholder values from the template.
    - Output ONLY the tagged fields shown below, in the listed order, each with an opening and a closing tag. No JSON, no code fences, no text outside the tags. Write LaTeX normally with single backslashes (e.g. \frac{1}{2}); do not escape anything.
    - Keep every field to one or two sentences except P_prime and derivation. In derivation, solve P' step by step and check each option BEFORE committing to answers, but keep it under 120 words: no digressions, re-checking or second-guessing, and never leave it unfinished. If no option matches your result, do not keep searching: commit to the closest option, lower confidence_level, and finish all the tags.
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
    1. State the original strategy S as the key step(s) of the standard solution (e.g. a substitution, a symmetry argument, a direct formula, a monotonicity or sign assumption).
    2. Change a structural condition so that applying S mechanically to P' gives a WRONG answer or cannot be completed. Useful levers: weaken or remove a domain/sign/monotonicity/continuity assumption so that cases must be split; change a quantifier or constraint so a symmetry no longer holds; add a degenerate or boundary case that S silently ignores; change the object so a standard formula no longer applies; reverse what is given and what is asked.
    3. Cosmetic changes (new constants, renamed variables, reordered options, extra data that does not change the method) do NOT count.
    4. Write P' as a complete, unambiguous, solvable question, and solve it with a different strategy. Recompute every option for P'; do not reuse the original options unless they are still meaningful, and place the correct option(s) at varied positions.
    5. naive_answer is the option letter(s) you get by applying S unchanged to P'. It MUST differ from answer_prime, and one wrong option of P' should be the naive_answer so the failure is visible.
    If no valid perturbation exists, output status REJECT (you may leave the other tags empty).
    {format_rules}{feedback_text}
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
    return template.format(problem=problem, answer=answer, format_rules=FORMAT_RULES, feedback_text=feedback_text)


def generate_prompt_b(problem, problem_perturbed, strategy, changed_condition):
    template = """
      Independently verify this proposed math perturbation.

      Original problem P: {P}
      Perturbed problem P': {P_prime}
      Claimed original strategy S: {strategy}
      Claimed changed condition: {changed_condition}

      Do the following:
      1. Solve P' independently by any valid method and select the correct option(s) -> answer_prime.
      2. Apply S to P' step by step exactly as it was used for P, without adapting it. Report the option letter(s) it yields as naive_answer, or NONE if S cannot be carried out.
      3. strategy_status: INVALID if a step of S fails or S gives a different result than answer_prime; INSUFFICIENT if S works only partially and extra reasoning is needed; STILL_VALID if S applied unchanged reaches answer_prime; UNCERTAIN otherwise. Judge by actually applying S, not by whether some method might still work.
      4. valid_problem is false only if P' is ambiguous, ill-posed or has no correct option.
      Include a confidence level in [0,1] for answer_prime.
      {format_rules}
      Output exactly this tagged format and nothing else:
      <valid_problem>true or false</valid_problem>
      <strategy_status>INVALID or INSUFFICIENT or STILL_VALID or UNCERTAIN</strategy_status>
      <reason>...</reason>
      <derivation>...</derivation>
      <naive_answer>option letters or NONE</naive_answer>
      <answer_prime>option letters</answer_prime>
      <confidence_level>number between 0 and 1</confidence_level>
    """
    return template.format(
        P=problem, P_prime=problem_perturbed, strategy=strategy,
        changed_condition=changed_condition, format_rules=FORMAT_RULES,
    )