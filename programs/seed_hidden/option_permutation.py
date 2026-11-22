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
    for old_label, new_label in mutated["label_map"].items():   # every option must sit where label_map says
        if seed["choices"]["ABCD".index(old_label)] != mutated["choices"]["ABCD".index(new_label)]:
            return "label_map points to wrong option"
    return True
