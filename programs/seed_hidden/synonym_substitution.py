FAMILY = "synonym_substitution"
MECHANISM = "LLM replaces 1-3 common (non-technical) words in the question with synonyms"

PROMPT = """In the question below, replace one to three common, non-technical words with exact synonyms.
Do not change technical terms, names, numbers, formulas or negations. Change nothing else.
Output only the modified question.

Question: {q}"""


def mutate(seed, llm, rng):
    q = llm(PROMPT.format(q=seed["question"]), temperature=0.7).strip()
    return {"question": q, "choices": list(seed["choices"])}


def check(seed, mutated):
    q0, q1 = seed["question"], mutated["question"]
    if q0 == q1:
        return "no change"
    if numbers_in(q0) != numbers_in(q1):
        return "numbers changed"
    for w in NEGATIONS & words(q0):
        if w not in words(q1):
            return f"dropped negation '{w}'"
    if missing := contains_all(named_entities_in(q0), q1):
        return f"named entity changed: {missing[0]}"
    if edit_ratio(q0, q1) > 0.30:
        return "too many edits"
    return True
