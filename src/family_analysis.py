"""Mutation reliability by family and magnitude (proposal RQ1), over all runs with an external-judge file.
Family = program name (seed families keep their names; synthesized programs are mapped to a coarse category by keywords).
Magnitude = word-length ratio and character similarity (difflib) between original and mutated question+options.
Validity proxy = external 32B referee verdict AND no label-mapping error (to be calibrated against the manual audit).
usage: python src/family_analysis.py [--out results/analysis/families_v2.txt]"""
import argparse, difflib, glob, json, os, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/analysis/families_v2.txt")
a = ap.parse_args()
EXT = {"mcq": "qwen3-32b", "vqa": "qwen3-vl-32b"}
CATS = [("option_order", r"permut|reorder|shuffl|relabel"),
        ("option_text", r"option_(synonym|replacement|paraphras|rephras|elaborat)|synonym"),
        ("negation_framing", r"negat|reversal|dependency|framing|quantif"),
        ("info_removal", r"hint_removal|content_removal|explanation_removal|context_removal|structure_removal"),
        ("concept_shift", r"conceptual|definition_shift|context_shift|false_attribute|conflicting|false"),
        ("paraphrase", r"paraphras|rephras|verbos|question_structure|grammar"),
        ("irrelevant_context", r"irrelevant|add_|distract"),
        ("image_text", r"image_text|text_overlay|image_context_replacement"),
        ("image_quality", r"image_(noise|jpeg|blur|rescale)|distorti"),
        ("image_colour", r"gray|grey|bright|contrast"),
        ("image_geometry", r"mirror|flip|rotat"),
        ("image_marker", r"margin|marker"),
        ("typo", r"typo|noise")]
seeds = {}
for p in ("data/mmlu_seeds.jsonl", "data/scienceqa_img_seeds.jsonl"):
    for l in open(p):
        d = json.loads(l); seeds[d["id"]] = d


def category(prog):
    """primary mechanism = category whose keyword appears first in the name; '+' marks compound programs."""
    fam = re.sub(r"_v\d+$", "", prog.split("_", 1)[1].lower())
    hits = [(m.start(), name) for name, pat in CATS for m in [re.search(pat, fam)] if m]
    prim = min(hits)[1] if hits else "other"
    return prim + ("+" if ("_with_" in fam or "_and_" in fam or len(fam) > 100) else "")


def _n(t):
    return " ".join(str(t).lower().split())


def label_err(e, s):
    """gold option text present exactly once but at a letter other than expected (broken label_map)."""
    gold = s.get("answer_letter") or s["answer"]
    ch = [_n(c) for c in e["mutated"]["choices"]]
    gt = _n(s["choices"]["ABCDE".index(gold)])
    return ch.count(gt) == 1 and ch.index(gt) != "ABCDE".index(e["expected"])


def text_of(q, ch, hint=""):
    return " ".join([hint or "", q] + [str(c) for c in ch])


stats = defaultdict(lambda: defaultdict(list))   # (task, cat) -> field -> values
for run in sorted(glob.glob("results/fuzz*/*/*")):
    task = "vqa" if "fuzz_vqa" in run else "mcq"
    jp = os.path.join(run, f"judge_ext_{EXT[task]}.jsonl")
    if "smoke" in run or not os.path.exists(jp):
        continue
    ext = {json.loads(l)["line"]: json.loads(l) for l in open(jp)}
    for i, l in enumerate(open(os.path.join(run, "events.jsonl"))):
        if i not in ext:
            continue
        e = json.loads(l); s = seeds[e["seed_id"]]; m = e["mutated"]
        o_t = text_of(s["question"], s["choices"], s.get("hint", ""))
        m_t = text_of(m["question"], m["choices"], m.get("hint", s.get("hint", "")))
        st = stats[(task, category(e["program"]))]
        bad = label_err(e, s)
        ok = ext[i]["ext"] == "VALID" and not bad   # label errors count as invalid tests
        st["ext_valid"].append(ok)
        st["label_err"].append(bad)
        st["self_valid"].append(e["judge"]["verdict"] == "VALID")
        st["len_ratio"].append(len(m_t.split()) / max(len(o_t.split()), 1))
        st["char_sim"].append(difflib.SequenceMatcher(None, o_t, m_t, autojunk=False).ratio())
        if task == "vqa":
            st["img_changed"].append(bool(m.get("image")))
        # magnitude buckets for the validity-vs-magnitude curve
        stats[(task, "~sim<0.6" if st["char_sim"][-1] < 0.6 else "~sim0.6-0.9" if st["char_sim"][-1] < 0.9 else "~sim>=0.9")]["ext_valid"].append(ok)

mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
out = [f"{'task':5s}{'category':20s}{'n':>6s}{'ext_valid':>11s}{'lab_err':>9s}{'self_valid':>11s}{'len_ratio':>11s}{'char_sim':>10s}{'img_chg':>9s}"]
for (task, cat) in sorted(stats):
    d = stats[(task, cat)]
    out.append(f"{task:5s}{cat:20s}{len(d['ext_valid']):>6d}{mean(d['ext_valid']):>11.2f}{mean(d['label_err']) if d['label_err'] else float('nan'):>9.2f}{mean(d['self_valid']):>11.2f}"
               f"{mean(d['len_ratio']):>11.2f}{mean(d['char_sim']):>10.2f}{mean(d['img_changed']) if d['img_changed'] else float('nan'):>9.2f}")
open(a.out, "w").write("\n".join(out) + "\n")
print("\n".join(out))
