import pandas as pd


def create_inference_prompt(row):
    """
    Standardizes the math question details into a clean, structured text prompt
    suitable for passing directly to an LLM.
    """
    # 1. Start with the main question body
    question_text = str(row.get('question', '')).strip()

    # 2. Extract options if this is a Multiple Choice Question and options exist in the row
    options_text = ""
    if 'options' in row and not isinstance(row['options'], (list, dict)) and pd.isna(row['options']):
        pass  # no options for this row
    elif 'options' in row:
        options = row['options']
        if isinstance(options, list):
            options_text = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in options])
        elif isinstance(options, dict):
            options_text = "\nOptions:\n" + "\n".join([f"{k}: {v}" for k, v in options.items()])

    # 3. Clean up common raw LaTeX inconsistencies
    question_text = question_text.replace('\\mathrm{~mL}', ' mL').replace('\\mathrm', '')

    # 4. Construct the structured prompt template
    prompt = (
        f"You are an expert mathematics AI assistant. Solve the following Joint Entrance Examination (JEE) Advanced level question step-by-step, then provide the final answer clearly.\n\n"
        f"### Category: {str(row.get('subject', 'math')).upper()} - {row.get('type', 'Question')}\n"
        f"### Source: {row.get('description', 'JEE Benchmark')}\n\n"
        f"### Question:\n{question_text}\n"
        f"{options_text}\n\n"
    )
    return prompt