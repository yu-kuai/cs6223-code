FAMILY = "image_grayscale"
MECHANISM = "convert the image to grayscale; only allowed when the question does not depend on colour"


def mutate(seed, llm, rng):
    return {"question": seed["question"], "choices": list(seed["choices"]),
            "image": ImageOps.grayscale(seed["image"]).convert("RGB")}


def check(seed, mutated):
    text = " ".join([seed["question"], seed.get("hint", "")] + seed["choices"])
    if COLOR_WORDS & words(text):
        return "question mentions colour"
    if "highlighted" in text.lower() or "map" in words(text):
        return "maps/highlighted regions often rely on colour"
    return True
