FAMILY = "image_jpeg"
MECHANISM = "re-encode the image as a lower-quality JPEG (compression artifacts)"


def mutate(seed, llm, rng):
    buf = io.BytesIO()
    seed["image"].save(buf, format="JPEG", quality=rng.randint(25, 50))
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return {"question": seed["question"], "choices": list(seed["choices"]), "image": img}


def check(seed, mutated):
    if not same_size(seed["image"], mutated["image"]):
        return "image size changed"
    if psnr(seed["image"], mutated["image"]) < 26:
        return "compression too lossy"
    return True
