FAMILY = "image_blur"
MECHANISM = "apply a light Gaussian blur to the image"


def mutate(seed, llm, rng):
    img = seed["image"].filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.2)))
    return {"question": seed["question"], "choices": list(seed["choices"]), "image": img}


def check(seed, mutated):
    if min(seed["image"].size) < 200:
        return "image too small, blur may erase details"
    if ssim(seed["image"], mutated["image"]) < 0.80:
        return "blur too strong"
    return True
