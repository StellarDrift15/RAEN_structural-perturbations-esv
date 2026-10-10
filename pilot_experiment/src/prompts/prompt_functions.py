
def generate_prompt_a(problem, answer):
    final_prompt = """
    Create one valid, strategy-breaking perturbation of this math problem.

    Original problem: {problem}
    Reference answer: {answer}

    Identify the original solution strategy and its required condition. Change one structural condition so the original strategy becomes invalid or insufficient. Provide a complete, unambiguous, solvable P', explain the failure briefly, propose an alternative strategy, and solve P'. Include a confidence level between [0,1] for your new answer to perturbed problem. Changing only numbers or wording is insufficient. If no valid perturbation is found, return status REJECT.

    Return JSON only:
    {{"status":"CANDIDATE/REJECT","P_prime":"...","original_strategy":"...","changed_condition":"...","failure_reason":"...","alternative_strategy":"...","answer_prime":"...","confidence_level":"...", "derivation":"..."}}
    """.format(problem=problem, answer=answer)
    return final_prompt


def generate_prompt_b(problem, problem_perturbed):
    final_prompt = """
      Independently verify this proposed math perturbation.

      Original problem P: {P}
      Perturbed problem P': {P_prime}

      Solve both. Identify the strategy used for P and its required conditions. Determine whether that strategy remains valid and sufficient for P'. Solve P' independently. Include a confidence level between [0,1] for your new answer to perturbed problem. Check well-posedness and ambiguity. Do not assume the perturbation is valid.

      Return JSON only:
      {{"valid_problem":true,"original_strategy":"...","required_condition":"...","strategy_status":"INVALID/INSUFFICIENT/STILL_VALID/UNCERTAIN","reason":"...","alternative_strategy":"...","answer_prime":"...","confidence_level":"...", "derivation":"..."}}
    """.format(P=problem, P_prime=problem_perturbed)
    return final_prompt