"""VQA (ScienceQA-IMG) MCQ prompt formatting + answer parsing. Like mcq.py but 2-5 options and an image.

The image is sent inline as a base64 data URL so the same messages work with vllm.LLM.chat (offline)
and with the vLLM OpenAI server (fuzz loop)."""
import base64, io, re

LETTERS = "ABCDE"


def system(n: int) -> str:
    ls = "A or B" if n == 2 else ", ".join(LETTERS[:n - 1]) + f", or {LETTERS[n - 1]}"
    return ("You are answering a multiple-choice question about an image. You may reason briefly, "
            f"but you must end your reply with a final line of the form 'Answer: X' where X is {ls}.")


def format_question(question: str, choices: list[str], hint: str = "") -> str:
    opts = "\n".join(f"({l}) {c}" for l, c in zip(LETTERS, choices))
    ctx = f"Context: {hint.strip()}\n\n" if hint and hint.strip() else ""
    return f"{ctx}{question.strip()}\n\n{opts}\n\nAnswer:"


def image_url(img) -> str:
    """img: data URL (returned as is), file path, or PIL.Image -> data URL (PNG)."""
    if isinstance(img, str) and img.startswith("data:"):
        return img
    if isinstance(img, str):
        with open(img, "rb") as f:
            data = f.read()
    else:
        buf = io.BytesIO(); img.convert("RGB").save(buf, format="PNG"); data = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def messages(question: str, choices: list[str], img, hint: str = "") -> list[dict]:
    return [{"role": "system", "content": system(len(choices))},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url(img)}},
                {"type": "text", "text": format_question(question, choices, hint)}]}]


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def parse_answer(text: str, choices: list[str]) -> str | None:
    """Same priority as mcq.parse_answer (+ last line "(X) option"), letters limited to the number of options."""
    L = LETTERS[:len(choices)]
    final = re.compile(rf"answer\s*(?:is)?\s*[:\-]?\s*\**\s*\(?([{L}])\)?(?![A-Za-z])", re.I)
    bare = re.compile(rf"^\s*\**\(?([{L}])\)?\**\s*[.)]?\s*$")
    lead = re.compile(rf"^\s*\**(?:\(([{L}])\)|([{L}])[.):])\s")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = final.findall(text)
    if m:
        return m[-1].upper()
    m = bare.match(text) or lead.match(text)
    if m:
        return next(g for g in m.groups() if g).upper()
    lines = [x for x in text.strip().splitlines() if x.strip()]
    m = lines and (lead.match(lines[-1] + " ") or bare.match(lines[-1]))  # reasoning then "(D) Asia" as last line
    if m:
        return next(g for g in m.groups() if g).upper()
    hits = [l for l, c in zip(L, choices) if _norm(c) and _norm(c) == _norm(text)]
    return hits[0] if len(hits) == 1 else None
