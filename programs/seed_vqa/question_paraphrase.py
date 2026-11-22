FAMILY = "question_paraphrase"
MECHANISM = "LLM rewords the question stem in different words; image and options untouched"

PROMPT = """Rewrite the following multiple-choice question stem so that it asks exactly the same thing using different wording.
The question is about an image you cannot see, so keep every reference to the image (e.g. "shown", "in the picture").
Rules: keep every number, name, unit and negation; do not add hints; do not change what is being asked.
Output only the rewritten question, nothing else.

Question: {q}"""


def mutate(seed, llm, rng):
    q = llm(PROMPT.format(q=seed["question"]), temperature=0.8).strip()
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
    if not 0.5 <= len(q1) / max(len(q0), 1) <= 3:
        return "length changed too much"
    if similarity(q0, q1) < 0.80:
        return "too different in meaning"
    return True
