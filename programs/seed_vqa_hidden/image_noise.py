FAMILY = "image_noise"
MECHANISM = "add mild Gaussian pixel noise to the image; text untouched"


def mutate(seed, llm, rng):
    img = seed["image"]
    sigma = rng.uniform(4, 10)
    a = np.asarray(img, dtype=np.float64)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, sigma, a.shape)
    out = Image.fromarray(np.clip(a + noise, 0, 255).astype("uint8"))
    return {"question": seed["question"], "choices": list(seed["choices"]), "image": out}


def check(seed, mutated):
    if not same_size(seed["image"], mutated["image"]):
        return "image size changed"
    if psnr(seed["image"], mutated["image"]) < 28:
        return "noise too strong"
    return True
