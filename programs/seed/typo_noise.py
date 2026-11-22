FAMILY = "typo_noise"
MECHANISM = "swap adjacent letters inside a few long, non-numeric words of the question"


def mutate(seed, llm, rng):
    toks = seed["question"].split(" ")
    cand = [i for i, t in enumerate(toks) if len(t) > 4 and t.isalpha() and t.lower() not in NEGATIONS]
    if not cand:
        raise ValueError("no candidate words")
    for i in rng.sample(cand, max(1, len(cand) // 10)):
        t = toks[i]; j = rng.randrange(1, len(t) - 2)
        toks[i] = t[:j] + t[j + 1] + t[j] + t[j + 2:]
    return {"question": " ".join(toks), "choices": list(seed["choices"])}


def check(seed, mutated):
    q0, q1 = seed["question"], mutated["question"]
    if numbers_in(q0) != numbers_in(q1):
        return "numbers changed"
    if edit_ratio(q0, q1) > 0.10:
        return "too many edits"
    return True
