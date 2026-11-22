FAMILY = "image_rescale"
MECHANISM = "downscale the image and scale it back up (lower effective resolution)"


def mutate(seed, llm, rng):
    img = seed["image"]
    f = rng.uniform(0.6, 0.85)
    small = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.BILINEAR)
    return {"question": seed["question"], "choices": list(seed["choices"]),
            "image": small.resize(img.size, Image.BILINEAR)}


def check(seed, mutated):
    if min(seed["image"].size) < 150:
        return "image too small to rescale safely"
    if ssim(seed["image"], mutated["image"]) < 0.80:
        return "too much detail lost"
    return True
