FAMILY = "image_brightness_contrast"
MECHANISM = "slightly change image brightness and contrast"


def mutate(seed, llm, rng):
    img = ImageEnhance.Brightness(seed["image"]).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    return {"question": seed["question"], "choices": list(seed["choices"]), "image": img}


def check(seed, mutated):
    text = " ".join([seed["question"], seed.get("hint", "")] + seed["choices"])
    if COLOR_WORDS & words(text):
        return "question is about colour or shade"
    if ssim(seed["image"], mutated["image"]) < 0.85:
        return "image changed too much"
    return True
