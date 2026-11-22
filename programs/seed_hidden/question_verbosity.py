FAMILY = "question_verbosity"
MECHANISM = "LLM restates the question in a longer, more elaborate style; options untouched"

STYLES = ["formal academic", "conversational", "very verbose and wordy", "exam-instruction style"]
PROMPT = """Restate the following multiple-choice question stem in a {style} style. It may be longer, but it must ask exactly the same thing.
Rules: keep every number, formula, name, unit and negation; do not add hints or new information.
Output only the restated question, nothing else.

Question: {q}"""


def mutate(seed, llm, rng):
    q = llm(PROMPT.format(style=rng.choice(STYLES), q=seed["question"]), temperature=0.8).strip()
    return {"question": q, "choices": list(seed["choices"])}


def check(seed, mutated):
    q0, q1 = seed["question"], mutated["question"]
    if not q1:
        return "empty question"
    for n in numbers_in(q0):
        if n not in q1:
            return f"dropped number {n}"
    for w in NEGATIONS & words(q0):
        if w not in words(q1):
            return f"dropped negation '{w}'"
    if not 0.8 <= len(q1) / max(len(q0), 1) <= 4:
        return "length ratio out of range"
    if similarity(q0, q1) < 0.70:
        return "too different in meaning"
    return True
