FAMILY = "option_permutation"
MECHANISM = "shuffle the order of the four options; expected label follows the permutation"


def mutate(seed, llm, rng):
    perm = list(range(4))
    while perm == list(range(4)):
        rng.shuffle(perm)
    ch = [seed["choices"][j] for j in perm]           # new position i holds old option perm[i]
    label_map = {"ABCD"[perm[i]]: "ABCD"[i] for i in range(4)}
    return {"question": seed["question"], "choices": ch, "label_map": label_map}


def check(seed, mutated):
    if sorted(mutated["choices"]) != sorted(seed["choices"]):
        return "options are not a permutation"
    old = seed["choices"]["ABCD".index(seed["answer"])]
    new = mutated["choices"]["ABCD".index(mutated["label_map"][seed["answer"]])]
    if old != new:
        return "label_map points to wrong option"
    return True
