"""Shared MCQ prompt formatting + answer parsing."""
import re

LETTERS = "ABCD"
SYSTEM = ("You are answering a multiple-choice question. You may reason briefly, "
          "but you must end your reply with a final line of the form 'Answer: X' where X is A, B, C, or D.")


def format_question(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"({l}) {c}" for l, c in zip(LETTERS, choices))
    return f"{question.strip()}\n\n{opts}\n\nAnswer:"


def messages(question: str, choices: list[str]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": format_question(question, choices)}]


_FINAL = re.compile(r"answer\s*(?:is)?\s*[:\-]?\s*\**\s*\(?([ABCD])\)?(?![A-Za-z])", re.I)
_BARE = re.compile(r"^\s*\**\(?([ABCD])\)?\**\s*[.)]?\s*$")
_LEAD = re.compile(r"^\s*\**(?:\(([ABCD])\)|([ABCD])[.):])\s")   # "(B) Buffer-overrun", "B. war"


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def parse_answer(text: str, choices: list[str] | None = None) -> str | None:
    """Priority: last 'Answer: X' > bare letter > leading '(X) ...' > reply == exactly one option's text.
    None if unparseable/truncated."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = _FINAL.findall(text)
    if m:
        return m[-1].upper()
    m = _BARE.match(text) or _LEAD.match(text)
    if m:
        return next(g for g in m.groups() if g).upper()
    if choices:
        hits = [l for l, c in zip(LETTERS, choices) if _norm(c) and _norm(c) == _norm(text)]
        if len(hits) == 1:
            return hits[0]
    return None
