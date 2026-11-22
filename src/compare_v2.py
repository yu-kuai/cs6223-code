"""Old vs new (v2) text runs, all graded by the same independent judge (default: DeepSeek, judge_ext_deepseek.jsonl).
Per run: tests asked, violations, label errors (gold text present once at another letter), real failures
(judge VALID and no label error), broken tests (the rest). Then mean ± sd per (model, setup) over seeds.
usage: python src/compare_v2.py [--tag deepseek] [--out results/analysis/compare_v2.txt]"""
import argparse, glob, json, os, statistics
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="deepseek")
ap.add_argument("--out", default=None)
ap.add_argument("--task", choices=["mcq", "vqa"], default="mcq")
ap.add_argument("--strict", action="store_true", help="do not count double-negation / shredded-question failures as real (src/debatable.py)")
a = ap.parse_args()
a.out = a.out or ("results/analysis/compare_v2" + ("_vqa" if a.task == "vqa" else "") + ("_strict" if a.strict else "") + ".txt")
import sys; sys.path.insert(0, "src")
from debatable import debatable  # noqa: E402
seeds = {json.loads(l)["id"]: json.loads(l) for l in open("data/mmlu_seeds.jsonl")}
for l in open("data/scienceqa_img_seeds.jsonl"):
    d = json.loads(l); d["answer"] = d["answer_letter"]; seeds[d["id"]] = d
n = lambda t: " ".join(str(t).lower().split())
SETUPS = [("Static", lambda d: d.startswith("fixed")),
          ("Self-Judge", lambda d: d in ("full", "full_s1", "full_s2")),
          ("Self-Judge + letter check", lambda d: d.startswith("full_guard")),
          ("External-Judge", lambda d: d.startswith("full_v2") and "smoke" not in d and "ansvis" not in d),
          ("External-Judge, answer visible", lambda d: d.startswith("full_v2_ansvis")),
          ("Greedy", lambda d: d.startswith("full_v3")),
          ("Guarded", lambda d: d.startswith("full_v4"))]


def label_err(e):
    s = seeds[e["seed_id"]]
    gt = n(s["choices"]["ABCDE".index(s["answer"])]); ch = [n(x) for x in e["mutated"]["choices"]]
    return ch.count(gt) == 1 and ch.index(gt) != "ABCDE".index(e["expected"])


rows = []
for run in sorted(glob.glob("results/fuzz/qwen3-*_self/*" if a.task == "mcq" else "results/fuzz_vqa/qwen3-vl-*_self/*")):
    d = os.path.basename(run)
    setup = None if "smoke" in d else next((name for name, f in SETUPS if f(d)), None)
    jp = os.path.join(run, f"judge_ext_{a.tag}.jsonl")
    if setup is None or not os.path.exists(jp):
        continue
    ext = {json.loads(l)["line"]: json.loads(l)["ext"] for l in open(jp)}
    c = defaultdict(int); qs = set()
    for i, l in enumerate(open(os.path.join(run, "events.jsonl"))):
        e = json.loads(l)
        c[e["status"]] += 1
        if e["status"] != "violation":
            continue
        if label_err(e):
            c["label_err"] += 1
        elif ext.get(i) == "VALID" and a.strict and debatable(seeds[e["seed_id"]]["question"], e["mutated"]["question"]):
            c["broken"] += 1; c["debatable"] += 1
        elif ext.get(i) == "VALID":
            c["real"] += 1; qs.add(e["seed_id"])
        elif ext.get(i) in ("INVALID", "UNPARSED"):
            c["broken"] += 1
        else:
            c["ungraded"] += 1
    c["questions"] = len(qs)
    model = run.split("/")[2].replace("_self", "")
    rows.append((model, setup, d, c))

out = [f"Graded by: {a.tag} (label errors counted as broken, separately listed)" + ("; STRICT: double-negation and shredded-question failures counted as broken" if a.strict else ""), "",
       f"{'model':12s}{'setup':31s}{'runs':>5s}{'asked':>11s}{'viol':>8s}{'real':>12s}{'broken':>12s}{'label_err':>12s}{'real %':>8s}{'per 1k asked':>13s}{'diff. questions':>17s}"]
groups = defaultdict(list)
for model, setup, d, c in rows:
    groups[(model, setup)].append(c)


def ms(xs):
    return f"{statistics.mean(xs):.0f}±{statistics.stdev(xs):.0f}" if len(xs) > 1 else f"{xs[0]:.0f}"


for model in (("qwen3-1.7b", "qwen3-8b", "qwen3-32b") if a.task == "mcq" else ("qwen3-vl-2b", "qwen3-vl-8b", "qwen3-vl-32b")):
    for setup, _ in SETUPS:
        cs = groups.get((model, setup))
        if not cs:
            continue
        asked = [c["pass"] + c["violation"] for c in cs]
        real = [c["real"] for c in cs]; broken = [c["broken"] for c in cs]; le = [c["label_err"] for c in cs]
        viol = [c["violation"] for c in cs]
        pct = 100 * sum(real) / max(sum(viol), 1)
        per1k = 1000 * sum(real) / max(sum(asked), 1)
        out.append(f"{model:12s}{setup:31s}{len(cs):>5d}{ms(asked):>11s}{ms(viol):>8s}{ms(real):>12s}{ms(broken):>12s}"
                   f"{ms(le):>12s}{pct:>7.0f}%{per1k:>13.1f}{ms([c['questions'] for c in cs]):>17s}")
    out.append("")
ung = sum(c["ungraded"] for _, _, _, c in rows)
out.append(f"ungraded violations (missing judge line): {ung}")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(a.out, "w").write("\n".join(out) + "\n")
print("\n".join(out))
