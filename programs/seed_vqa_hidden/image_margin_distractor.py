FAMILY = "image_margin_distractor"
MECHANISM = "add a blank margin to one side of the image and draw an unrelated shape in it; original pixels kept"

SHAPES = ["circle", "square", "triangle"]


def mutate(seed, llm, rng):
    img = seed["image"]
    w, h = img.size
    m = max(40, int(0.2 * min(w, h)))
    side = rng.choice(["right", "bottom"])
    canvas = Image.new("RGB", (w + m, h) if side == "right" else (w, h + m), "white")
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    s = int(m * 0.6)
    x0, y0 = (w + (m - s) // 2, rng.randint(0, max(0, h - s))) if side == "right" else (rng.randint(0, max(0, w - s)), h + (m - s) // 2)
    color = tuple(rng.randint(0, 200) for _ in range(3))
    shape = rng.choice(SHAPES)
    if shape == "circle":
        d.ellipse([x0, y0, x0 + s, y0 + s], fill=color)
    elif shape == "square":
        d.rectangle([x0, y0, x0 + s, y0 + s], fill=color)
    else:
        d.polygon([(x0 + s // 2, y0), (x0, y0 + s), (x0 + s, y0 + s)], fill=color)
    return {"question": seed["question"], "choices": list(seed["choices"]), "image": canvas}


def check(seed, mutated):
    a, b = seed["image"], mutated["image"]
    if b.width < a.width or b.height < a.height:
        return "image shrank"
    if pixel_change(a, b.crop((0, 0, a.width, a.height)), tol=0) > 0:
        return "original pixels were modified"
    text = " ".join([seed["question"], seed.get("hint", "")] + seed["choices"]).lower()
    for w in ("how many", "shape", "circle", "square", "triangle", "count"):
        if w in text:
            return f"question involves '{w}', a drawn shape could change the answer"
    return True
