FAMILY = "option_permutation"
MECHANISM = "shuffle the order of the options; expected label follows the permutation"


def mutate(seed, llm, rng):
    n = len(seed["choices"])
    L = "ABCDE"[:n]
    perm = list(range(n))
    while perm == list(range(n)):
        rng.shuffle(perm)
    ch = [seed["choices"][j] for j in perm]           # new position i holds old option perm[i]
    label_map = {L[perm[i]]: L[i] for i in range(n)}
    return {"question": seed["question"], "choices": ch, "label_map": label_map}


def check(seed, mutated):
    L = "ABCDE"
    if sorted(mutated["choices"]) != sorted(seed["choices"]):
        return "options are not a permutation"
    for old_label, new_label in mutated["label_map"].items():   # every option must sit where label_map says
        if seed["choices"][L.index(old_label)] != mutated["choices"][L.index(new_label)]:
            return "label_map points to wrong option"
    return True
