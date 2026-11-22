"""Helper library exposed to mutation programs (inside the sandbox).
Everything here is cheap and deterministic, except `similarity`, which is an RPC to the parent
(the parent hosts a small sentence-embedding model)."""
import math, re

NEGATIONS = {"not", "no", "never", "none", "except", "neither", "nor", "without", "cannot", "isn't",
             "aren't", "doesn't", "don't", "didn't", "won't", "wasn't", "weren't", "false", "incorrect", "least"}
QUANTIFIERS = ["at least", "at most", "all", "every", "only", "always", "most", "some", "exactly", "more than", "less than"]

_rpc = None  # set by sandbox worker: _rpc(fn_name, **kwargs) -> value


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()))


def numbers_in(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?(?:/\d+)?", text)


def math_spans_in(text: str) -> list[str]:
    """LaTeX $...$ spans and compact formula-like tokens (x^2, 3x+2, f(x))."""
    spans = re.findall(r"\$[^$]+\$", text)
    spans += re.findall(r"\b\w*[\^_=+*/]\w+[\w^_=+*/()]*", text)
    return spans


def named_entities_in(text: str) -> list[str]:
    """Rough: capitalized word sequences not at sentence start, plus ALL-CAPS acronyms."""
    ents = re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text)
    ents += re.findall(r"\b[A-Z]{2,}\b", text)
    return list(dict.fromkeys(ents))


def edit_distance(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:  # pure-python fallback
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def edit_ratio(a: str, b: str) -> float:
    """Normalized edit distance in [0, 1]."""
    return edit_distance(a, b) / max(len(a), len(b), 1)


def similarity(a: str, b: str) -> float:
    """Cosine similarity of sentence embeddings (all-MiniLM-L6-v2), via parent process."""
    return float(_rpc("similarity", a=a, b=b))


def contains_all(needles, haystack: str) -> list[str]:
    """Return the needles that are MISSING from haystack (case-insensitive)."""
    h = haystack.lower()
    return [n for n in needles if n.lower() not in h]


# ---------------------------------------------------------------- image helpers (VQA task)
# Images are PIL.Image (RGB). numpy/PIL are imported lazily so the text task does not pay for them.
COLOR_WORDS = {"red", "orange", "yellow", "green", "blue", "purple", "violet", "pink", "brown", "black", "white",
               "gray", "grey", "color", "colour", "colored", "coloured", "shade", "darker", "lighter", "bright"}
SPATIAL_WORDS = {"left", "right", "top", "bottom", "above", "below", "north", "south", "east", "west", "upper",
                 "lower", "front", "behind", "next", "between", "highlighted", "shown", "corner", "side"}


def _arr(img):
    import numpy as np
    return np.asarray(img.convert("RGB"), dtype=np.float64)


def same_size(a, b) -> bool:
    return a.size == b.size


def psnr(a, b) -> float:
    """Peak signal-to-noise ratio in dB (higher = more similar; >=30 is visually near-identical).
    Images of different size are compared after resizing b to a's size."""
    import numpy as np
    if a.size != b.size:
        b = b.resize(a.size)
    mse = float(np.mean((_arr(a) - _arr(b)) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def ssim(a, b) -> float:
    """Structural similarity of the grayscale images (7x7 windows), in [-1, 1]; 1 = identical.
    >=0.9 is usually 'same content'. b is resized to a's size if needed."""
    import numpy as np
    if a.size != b.size:
        b = b.resize(a.size)
    x = np.asarray(a.convert("L"), dtype=np.float64); y = np.asarray(b.convert("L"), dtype=np.float64)
    if min(x.shape) < 7:
        return 1.0 if np.allclose(x, y) else 0.0

    def box(z, k=7):  # mean over k x k windows via integral image
        c = np.pad(z, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
    mx, my = box(x), box(y)
    vx, vy, cxy = box(x * x) - mx ** 2, box(y * y) - my ** 2, box(x * y) - mx * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return float(s.mean())


def pixel_change(a, b, tol: int = 16) -> float:
    """Fraction of pixels whose max channel difference exceeds tol (0 = untouched). b resized if needed."""
    import numpy as np
    if a.size != b.size:
        b = b.resize(a.size)
    return float((np.abs(_arr(a) - _arr(b)).max(axis=2) > tol).mean())


def vlm(prompt: str, image=None, temperature: float = 0.7, max_tokens: int = 512) -> str:
    """Ask the fuzzer model (a vision-language model) about an image. Counts toward the budget like llm()."""
    import base64, io
    b64 = None
    if image is not None:
        buf = io.BytesIO(); image.convert("RGB").save(buf, format="PNG"); b64 = base64.b64encode(buf.getvalue()).decode()
    return _rpc("vlm", prompt=prompt, image=b64, temperature=temperature, max_tokens=max_tokens)


EXPORTS = ["NEGATIONS", "QUANTIFIERS", "words", "numbers_in", "math_spans_in", "named_entities_in",
           "edit_distance", "edit_ratio", "similarity", "contains_all",
           "COLOR_WORDS", "SPATIAL_WORDS", "same_size", "psnr", "ssim", "pixel_change", "vlm"]
