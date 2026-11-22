"""Prompts for the judge (triage) and for program synthesis / refinement by the fuzzer."""
import json, re

from mcq import format_question

# ----------------------------------------------------------------------------- judge
JUDGE = """You are auditing a metamorphic test of a question-answering model.
A test generator modified an original multiple-choice question. The modification is supposed to preserve
the meaning, so that the correct answer of the modified question is the EXPECTED label below.
The model under test answered the modified question with a DIFFERENT label.

The generator is ALLOWED to make changes that do not affect the answer, for example: rewording, changing style or
length, adding background sentences that are irrelevant to the answer, reordering options (the expected label already
accounts for this), rewording options, typos, synonyms. Such changes alone do NOT make a test invalid.

Decide which of these is true:
- VALID: the modified question still has exactly the same meaning and the expected label is still clearly the
  only correct answer. The model under test made a real mistake.
- INVALID: the modification changed the meaning, introduced an error or ambiguity, leaked or removed information,
  or otherwise made the expected label no longer clearly correct. This is a broken test, not a model mistake.

The generator's stated mechanism for this test: {mechanism}

### Original question (correct answer: {gold})
{orig}

### Modified question (expected answer: {expected})
{mut}

### Model under test's reply (it answered {got})
{reply}

Respond with a JSON object only, no other text. Write the reason first, then the verdict that follows from it:
{{"reason": "<one sentence: does the modified question have the same correct answer, and why>",
  "verdict": "VALID" or "INVALID",
  "pattern": "<if VALID: one short sentence, in your own words, describing the specific feature of THIS modification that most likely caused the model's mistake (what changed, and why it could mislead). Do not copy wording from these instructions. If INVALID: empty>"}}"""


def judge_prompt(seed, mutated, expected, got, reply, mechanism=""):
    return JUDGE.format(mechanism=mechanism or "unknown", gold=seed["answer"], orig=format_question(seed["question"], seed["choices"]),
                        expected=expected, mut=format_question(mutated["question"], mutated["choices"]),
                        got=got, reply=reply[-1500:])


JUDGE_VQA = """You are auditing a metamorphic test of a vision-language model.
A test generator modified an original multiple-choice question about an image (the question text, the options, the
image, or several of these). The modification is supposed to preserve the meaning, so that the correct answer of the
modified question is the EXPECTED label below. The model under test answered the modified question with a DIFFERENT label.

The FIRST image above is the original image, the SECOND image is the modified image.

The generator is ALLOWED to make changes that do not affect the answer, for example: rewording, adding irrelevant
background text, reordering options (the expected label already accounts for this), typos; mild noise, blur,
compression, brightness/contrast changes, resizing, adding margins or unrelated marks that cover nothing relevant,
grayscale when colour does not matter. Such changes alone do NOT make a test invalid.

Decide which of these is true:
- VALID: everything needed to answer is still present and clear, and the expected label is still clearly the only
  correct answer. The model under test made a real mistake.
- INVALID: the modification changed the meaning, hid, removed, covered, recoloured or distorted information needed to
  answer, introduced an error or ambiguity, or otherwise made the expected label no longer clearly correct.
  This is a broken test, not a model mistake.

The generator's stated mechanism for this test: {mechanism}

### Original question (correct answer: {gold})
{orig}

### Modified question (expected answer: {expected})
{mut}

### Model under test's reply (it answered {got})
{reply}

Respond with a JSON object only, no other text. Write the reason first, then the verdict that follows from it:
{{"reason": "<one sentence: does the modified question have the same correct answer, and why>",
  "verdict": "VALID" or "INVALID",
  "pattern": "<if VALID: one short sentence, in your own words, describing the specific feature of THIS modification that most likely caused the model's mistake (what changed, and why it could mislead). Do not copy wording from these instructions. If INVALID: empty>"}}"""


def judge_messages_vqa(seed, mutated, expected, got, reply, mechanism, orig_img_url, mut_img_url):
    """Chat messages for the VQA judge: both images, then the text. Image args are data URLs."""
    from vqa import format_question as fq
    text = JUDGE_VQA.format(mechanism=mechanism or "unknown", gold=seed["answer"],
                            orig=fq(seed["question"], seed["choices"], seed.get("hint", "")), expected=expected,
                            mut=fq(mutated["question"], mutated["choices"], mutated.get("hint", "")),
                            got=got, reply=reply[-1500:])
    return [{"role": "user", "content": [
        {"type": "text", "text": "Original image:"}, {"type": "image_url", "image_url": {"url": orig_img_url}},
        {"type": "text", "text": "Modified image:"}, {"type": "image_url", "image_url": {"url": mut_img_url}},
        {"type": "text", "text": text}]}]


def parse_judge(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            v = str(d.get("verdict", "")).upper()
            if v in ("VALID", "INVALID"):
                return {"verdict": v, "reason": str(d.get("reason", ""))[:500], "pattern": str(d.get("pattern", ""))[:300]}
        except Exception:
            pass
    v = re.search(r"\b(INVALID|VALID)\b", text.upper())
    return {"verdict": v.group(1) if v else "UNPARSED", "reason": text[:300], "pattern": ""}


# ----------------------------------------------------------------------------- program API (shared)
API = """A mutation program is a Python module that defines:

FAMILY = "<short snake_case name>"
MECHANISM = "<one sentence: what the mutation does>"

def mutate(seed, llm, rng) -> dict:
    # seed = {"question": str, "choices": [4 strings], "answer": "A"|"B"|"C"|"D"}
    # llm(prompt: str, temperature=0.7, max_tokens=512) -> str   calls a language model (costs budget; use at most 2 calls)
    # rng = random.Random instance (use it for all randomness)
    # return {"question": str, "choices": [4 strings], "label_map": {...}}   (label_map optional)
    # label_map maps each old label to its new label; omit it if option order is unchanged.
    # raise an exception if the seed is unsuitable for this mutation.

def check(seed, mutated) -> True | str:
    # validity constraints: return True if the mutation provably preserves the correct answer,
    # otherwise return a short string explaining the rejection. Must NOT call llm.

Available without import: re, math, random, string, and these helpers:
  words(text)->set of lowercase words; numbers_in(text)->list of numbers; math_spans_in(text)->list of formula spans;
  named_entities_in(text)->list; edit_distance(a,b)->int; edit_ratio(a,b)->float in [0,1];
  similarity(a,b)->float cosine similarity of sentence embeddings (cheap); contains_all(needles, text)->list of missing needles;
  NEGATIONS (set of negation words); QUANTIFIERS (list of quantifier phrases).
Do not redefine these helper names (they are provided and will override your definitions).
Allowed imports: re, math, random, string, itertools, collections, json, functools, unicodedata, textwrap.
No file, network or OS access.

The goal of a mutation is to produce a question that a competent reader would answer EXACTLY the same way
(answer-preserving), but that might nevertheless confuse a language model."""

API_TEXT = API

API_VQA = """A mutation program is a Python module that defines:

FAMILY = "<short snake_case name>"
MECHANISM = "<one sentence: what the mutation does>"

def mutate(seed, llm, rng) -> dict:
    # seed = {"question": str, "choices": [2 to 5 strings], "answer": "A".."E", "hint": str (text context, may be ""),
    #         "image": PIL.Image (RGB), "subject": str, "topic": str}
    # llm(prompt: str, temperature=0.7, max_tokens=512) -> str   calls a language model, text only (costs budget)
    # vlm(prompt: str, image=None, temperature=0.7, max_tokens=512) -> str   asks a vision-language model about an image
    #     (costs budget). Use at most 2 llm/vlm calls in total.
    # rng = random.Random instance (use it for all randomness)
    # return {"question": str, "choices": [same number of strings], "image": PIL.Image, "hint": str, "label_map": {...}}
    #   "image" and "hint" are optional (omit to keep the originals); label_map maps each old label to its new label,
    #   omit it if option order is unchanged.
    # raise an exception if the seed is unsuitable for this mutation.

def check(seed, mutated) -> True | str:
    # validity constraints: return True if the mutation provably preserves the correct answer,
    # otherwise return a short string explaining the rejection. Must NOT call llm or vlm.
    # seed["image"] and mutated["image"] are both PIL.Image here.

The question is about the image: the information needed to answer is usually IN the image (maps, diagrams, charts,
photos, text in the image). An image change is answer-preserving only if every detail needed to answer is still
clearly visible and unchanged in meaning (no cropped/covered/recoloured relevant content, no mirrored maps or text).

Available without import: re, math, random, string, io, np (numpy), Image, ImageDraw, ImageEnhance, ImageFilter,
ImageFont, ImageOps (from PIL), and these helpers:
  words(text)->set of lowercase words; numbers_in(text)->list of numbers; named_entities_in(text)->list;
  edit_distance(a,b)->int; edit_ratio(a,b)->float in [0,1]; similarity(a,b)->float cosine similarity of sentence
  embeddings (cheap); contains_all(needles, text)->list of missing needles; NEGATIONS (set); QUANTIFIERS (list);
  COLOR_WORDS (set of colour/shade words); SPATIAL_WORDS (set of position/direction words);
  same_size(img_a, img_b)->bool; psnr(img_a, img_b)->dB (>=30 near-identical); ssim(img_a, img_b)->[-1,1] (>=0.9 same
  content); pixel_change(img_a, img_b, tol=16)->fraction of pixels changed.
Do not redefine these helper names (they are provided and will override your definitions).
Allowed imports: re, math, random, string, itertools, collections, json, functools, unicodedata, textwrap, io, PIL, numpy.
No file, network or OS access.

The goal of a mutation is to produce a question (text and/or image) that a competent person would answer EXACTLY the
same way (answer-preserving), but that might nevertheless confuse a vision-language model."""

TASK = "mcq"


API_TEXT_HIDDEN = API_TEXT.replace(
    '# seed = {"question": str, "choices": [4 strings], "answer": "A"|"B"|"C"|"D"}',
    '# seed = {"question": str, "choices": [4 strings]}   (you are NOT told which option is correct)').replace(
    "(answer-preserving), but that might nevertheless confuse a language model.",
    "(answer-preserving), but that might nevertheless confuse a language model.\n"
    "You do not know which option is correct, so the mutation must preserve the answer WHICHEVER option is correct:\n"
    "treat all options the same way, never weaken or strengthen individual options, and never hint at any option.\n"
    "If you reorder options, return label_map so every option can be followed to its new letter.\n"
    "The test harness also applies fixed checks after your check(): options must stay 4 distinct strings, every number\n"
    "and negation word of the question must be kept, no new 'except/false/incorrect/least' framing may be added, and\n"
    "the question must not newly mention any option's text.")


API_VQA_HIDDEN = API_VQA.replace(
    '# seed = {"question": str, "choices": [2 to 5 strings], "answer": "A".."E", "hint": str (text context, may be ""),',
    '# seed = {"question": str, "choices": [2 to 5 strings], "hint": str (text context, may be ""),').replace(
    '#         "image": PIL.Image (RGB), "subject": str, "topic": str}',
    '#         "image": PIL.Image (RGB), "subject": str, "topic": str}   (you are NOT told which option is correct)').replace(
    "same way (answer-preserving), but that might nevertheless confuse a vision-language model.",
    "same way (answer-preserving), but that might nevertheless confuse a vision-language model.\n"
    "You do not know which option is correct, so the mutation must preserve the answer WHICHEVER option is correct:\n"
    "treat all options the same way, never weaken or strengthen individual options, and never hint at any option.\n"
    "If you reorder options, return label_map so every option can be followed to its new letter.\n"
    "The test harness also applies fixed checks after your check(): options must stay distinct strings, every number\n"
    "and negation word of the question must be kept, every number of the hint must be kept, no new\n"
    "'except/false/incorrect/least' framing may be added, and the question must not newly mention any option's text.")
assert API_VQA_HIDDEN.count("NOT told") == 1 and "WHICHEVER" in API_VQA_HIDDEN


V4_NOTE = """
Additional fixed checks (v4), applied after your check(): the question must not gain any negation word
(not, no, never, none, neither, nor, cannot, without, n't), not even in added context; the original question must stay
in one piece (you may add text before or after it, never inside it); no option may become much longer than the
original option (no added explanations or commentary inside options)."""


def configure(task: str, hide_answer: bool = False, v4: bool = False):
    """Switch prompts between the text MCQ task and the VQA task (and the hidden-answer program API)."""
    global API, TASK
    TASK = task
    if task == "vqa":
        API = API_VQA_HIDDEN if hide_answer else API_VQA
    else:
        API = API_TEXT_HIDDEN if hide_answer else API_TEXT
    if v4:
        API = API + "\n" + V4_NOTE.strip("\n")


# repair prompt v2: judge reasons for broken tests + the program's own real failures (what to keep)
REFINE2 = """{api}

The following mutation program produced INVALID tests: an auditor found that its mutations changed the meaning
of the question, so the model under test was blamed for a mistake that was really the program's fault.

### Program
```python
{code}
```

### Invalid tests it produced (fix these)
{cases}

### Real failures it found (keep this ability: these tests were judged fair)
{good}

Rewrite the program so it stops producing this class of invalid mutation while still finding real failures:
1. Tighten check(): add general code rules that would reject these invalid cases and similar ones (not rules that only
   match these exact examples). Make sure check() cannot crash and does not reject almost everything.
2. Where a problem cannot be detected by code, improve mutate() (e.g. the instructions in its LLM prompt) instead.
Keep FAMILY, keep the same overall mutation idea, keep the function signatures.
Your revision will be tested on the same questions before it is accepted.
Output the complete revised program in one ```python block."""


def refine_prompt2(code, fp_cases, tp_cases):
    def fmt(cs, with_reason=True):
        out = []
        for i, c in enumerate(cs, 1):
            out.append(f"[{i}] Original question: {c['orig_q'][:600]}\n    Mutated question: {c['mut_q'][:600]}\n"
                       f"    Original options: {c['orig_choices']}\n    Mutated options: {c['mut_choices']}\n"
                       f"    Auditor's reason: {c['reason']}")
        return "\n".join(out)
    return REFINE2.format(api=API, code=code, cases=fmt(fp_cases),
                          good=fmt(tp_cases) if tp_cases else "(none yet)")


EXAMPLE_HEADER = "### Example program\n```python\n{code}\n```"


def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        code = max(blocks, key=len)
    else:   # unclosed fence (e.g. output cut at max_tokens): take everything after the opening fence
        m = re.search(r"```(?:python)?\s*\n(.*)", text, flags=re.S)
        code = m.group(1) if m else text
    return code if "def mutate" in code and "def check" in code else None


# ----------------------------------------------------------------------------- refinement (FP -> tighten)
REFINE = """{api}

The following mutation program produced INVALID tests: an auditor found that its mutations changed the meaning
of the question, so the model under test was blamed for a mistake that was really the program's fault.

### Program
```python
{code}
```

### Invalid tests it produced
{cases}

Rewrite the program so it stops producing this class of invalid mutation:
1. Tighten check(): add general code rules that would reject these cases and similar ones (not rules that only match these exact examples).
2. Where a problem cannot be detected by code, improve the instructions in mutate()'s LLM prompt instead.
Keep FAMILY, keep the same overall mutation idea, keep the function signatures.
Output the complete revised program in one ```python block."""


def refine_prompt(code, fp_cases):
    cases = []
    for i, c in enumerate(fp_cases, 1):
        cases.append(f"[{i}] Original question: {c['orig_q'][:600]}\n    Mutated question: {c['mut_q'][:600]}\n"
                     f"    Original options: {c['orig_choices']}\n    Mutated options: {c['mut_choices']}\n"
                     f"    Auditor's reason: {c['reason']}")
    return REFINE.format(api=API, code=code, cases="\n".join(cases))


# ----------------------------------------------------------------------------- new programs (exploration)
EXPLORE_TP = """{api}

{example}

The testing campaign so far has found these KINDS of real model failures (already known, do NOT target them again):
{known}

Here is a recently found real failure, for inspiration about what confuses the model:
{tp}

Write a NEW mutation program that probes a DIFFERENT weakness than the known ones above: a different kind of change,
or a different part of the question. It must still be answer-preserving and must have a strict check().
Existing program families (do not duplicate them): {families}
Give it a new FAMILY name. Output the complete program in one ```python block."""

# v3 (--bold): build on the program that finds the most real failures, with explicit strength feedback
EXPLORE_EXPLOIT = """{api}

{example}

This program is currently the most successful one: it found real model failures (judged fair by an auditor) in
{rate} of its tests. For comparison, newly written programs in this campaign find real failures in {avg} of tests.
Real failures it found:
{tps}

Write a STRONGER VARIANT of this program: keep the core idea that works, but make the change bolder or apply it in a
more effective way (e.g. more of it, a harder-to-ignore version, a better place in the question), so that it changes
the model's answer more often. It must remain answer-preserving: a careful human must still give the same answer.
Do not copy the program unchanged. Give it a new FAMILY name. Output the complete program in one ```python block."""

STRENGTH_NOTE = """
Feedback on strength: programs written so far change the model's answer in only {avg} of their tests, while the best
program reaches {best}. Programs that almost never change the answer are retired. Make a change bold enough to
actually confuse the model, while keeping the answer the same for a careful human."""

EXPLORE_INCONSISTENCY = """{api}

{example}

The program above changed the model's answer in {rate} of its tests. Write a NEW variant mutation program that is
likely to change the model's answer even MORE often. It should still be answer-preserving.
Existing program families: {families}
Give it a new FAMILY name. Output the complete program in one ```python block."""

EXPLORE_BLIND = """{api}

{example}

Write a NEW answer-preserving mutation program, different from the example. It must have a strict check().
Existing program families (do not duplicate them): {families}
Give it a new FAMILY name. Output the complete program in one ```python block."""


def tp_text(c):
    return (f"Original: {c['orig_q'][:500]}\nModified: {c['mut_q'][:500]}\n"
            f"Correct answer {c['expected']}, model answered {c['got']}. Auditor: {c['pattern'] or c['reason']}")
