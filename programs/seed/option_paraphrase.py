FAMILY = "option_paraphrase"
MECHANISM = "LLM rewords each answer option (short/numeric options are left unchanged); question untouched"

PROMPT = """Reword each of the following answer options so it means exactly the same thing in different words.
Keep every number, formula, name and negation. Output exactly {n} lines, in the same order, one option per line, with no labels.

{opts}"""


def _rewordable(c):
    return len(c.split()) >= 3 and not re.fullmatch(r"[\d\s.,/%$-]+", c)


def mutate(seed, llm, rng):
    ch = list(seed["choices"])
    idx = [i for i, c in enumerate(ch) if _rewordable(c)]
    if not idx:
        raise ValueError("no rewordable options")
    out = llm(PROMPT.format(n=len(idx), opts="\n".join(ch[i] for i in idx)), temperature=0.7)
    lines = [re.sub(r"^\s*(?:[-*]|\(?[A-Da-d1-4][.)])\s*", "", l).strip() for l in out.strip().splitlines() if l.strip()]
    if len(lines) != len(idx):
        raise ValueError(f"expected {len(idx)} lines, got {len(lines)}")
    for i, l in zip(idx, lines):
        ch[i] = l
    return {"question": seed["question"], "choices": ch}


def check(seed, mutated):
    old, new = seed["choices"], mutated["choices"]
    if len(new) != 4 or len(set(c.lower() for c in new)) != 4:
        return "options not 4 distinct strings"
    for o, n in zip(old, new):
        if o == n:
            continue
        for num in numbers_in(o):
            if num not in n:
                return f"option dropped number {num}"
        for w in NEGATIONS & words(o):
            if w not in words(n):
                return f"option dropped negation '{w}'"
        if similarity(o, n) < 0.75:
            return f"option changed meaning: '{o}' -> '{n}'"
    return True
