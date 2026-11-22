"""Detect the two 'debatable' mutation styles found in the Claude labeling of v3 cases:
double negation added to the question (e.g. 'NOT NOT', 'NOT the one that is NOT'), and a shredded question
(original words still in order but with filler text inserted inside the question)."""
import re

_W = re.compile(r"[a-z0-9]+")
_NEG = re.compile(r"\b(not|n't|isn't)\b", re.I)


def double_negation(orig_q, new_q):
    return len(_NEG.findall(new_q)) >= len(_NEG.findall(orig_q)) + 2


def shredded(orig_q, new_q):
    o, n = _W.findall(orig_q.lower()), _W.findall(new_q.lower())
    if len(n) < len(o) + 15 or " ".join(o) in " ".join(n):
        return False                      # short addition, or original kept as one block (prefix/suffix context)
    i = gaps = 0                          # original words appear in order, interrupted several times
    for w in n:
        if i < len(o) and w == o[i]:
            i += 1
        elif 0 < i < len(o):
            gaps += 1
    return i == len(o) and gaps >= 10


def debatable(orig_q, new_q):
    if double_negation(orig_q, new_q):
        return "double_negation"
    if shredded(orig_q, new_q):
        return "shredded"
    return None
